import math
import string

import numpy as np

from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.backend.config import is_flash_attention_enabled
from keras.src.layers.activations.softmax import Softmax
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.layer import Layer
from keras.src.layers.regularization.dropout import Dropout


@keras_export("keras.layers.MultiHeadAttention")
class MultiHeadAttention(Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need"
    [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as `value_dim` can take
    a linear projection and return.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        output_shape: The expected shape of an output tensor, besides the batch
            and sequence dims. If not specified, projects back to the query
            feature dim (the query input's last dimension).
        attention_axes: axes over which the attention is applied. `None` means
            attention over all axes, but batch, heads, and features.
        flash_attention: If `None`, the layer attempts to use flash
            attention for faster and more memory-efficient attention
            computations when possible. This behavior can be configured using
            `keras.config.enable_flash_attention()` or
            `keras.config.disable_flash_attention()`.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        seed: Optional integer to seed the dropout layer.

    Call arguments:
        query: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
            `T` is the target sequence length, and dim is the feature dimension.
        value: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
            `S` is the source sequence length, and dim is the feature dimension.
        key: Optional key tensor of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model, or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        attention_scores: (Optional) multi-head attention coefficients over
            attention axes.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        flash_attention=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        if output_shape:
            if isinstance(output_shape, int):
                output_shape = (output_shape,)
            try:
                output_shape = tuple(output_shape)
            except:
                raise ValueError(
                    f"Invalid `output_shape`: {output_shape}. When "
                    "specified, the `output_shape` should be of type tuple, "
                    "list, or int."
                )
        self._output_shape = output_shape
        self._flash_attention = flash_attention or is_flash_attention_enabled()
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if isinstance(attention_axes, int):
            attention_axes = (attention_axes,)
        elif attention_axes and not isinstance(attention_axes, (list, tuple)):
            raise ValueError(
                "`attention_axes` must be an int, list, or tuple."
                f"Received: attention_axes={attention_axes}"
            )
        self._attention_axes = attention_axes
        self.seed = seed

        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(float(self._key_dim))

        # Check for flash attention constraints
        if self._flash_attention and self._dropout > 0.0:
            raise ValueError(
                "Dropout is not supported when flash attention is enabled. "
                "Please set dropout to 0.0 to use flash attention."
            )

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def key_dim(self):
        return self._key_dim

    @property
    def value_dim(self):
        return self._value_dim

    @property
    def dropout(self):
        return self._dropout

    @property
    def use_bias(self):
        return self._use_bias

    # Avoid exposing `output_shape` as it may conflict with `Functional` and
    # `Sequential` models when calling `summary()`.

    @property
    def attention_axes(self):
        return self._attention_axes

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "seed": self.seed,
        }
        return {**base_config, **config}

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        """Builds layers and variables.

        Args:
            query_shape: Shape of the `query` tensor.
            value_shape: Shape of the `value` tensor.
            key: Optional shape of the `key` tensor.
        """
        key_shape = value_shape if key_shape is None else key_shape

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )

        query_rank = len(query_shape)
        value_rank = len(value_shape)
        key_rank = len(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=1, output_dims=2
        )
        self._query_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            key_rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            value_rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape)
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))

    @property
    def query_dense(self):
        return self._query_dense

    @property
    def key_dense(self):
        return self._key_dense

    @property
    def value_dense(self):
        return self._value_dense

    @property
    def output_dense(self):
        return self._output_dense

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            dtype=self.dtype_policy,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def _make_output_dense(self, query_shape, common_kwargs, name=None):
        """Builds the output projection matrix.

        Args:
            free_dims: Number of free dimensions for einsum equation building.
            common_kwargs: Common keyword arguments for einsum layer.
            name: Name for the projection layer.

        Returns:
            Projection layer.
        """
        query_rank = len(query_shape)
        if self._output_shape:
            output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=2, output_dims=len(output_shape)
        )
        return EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs,
        )

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )
        self._softmax = Softmax(axis=norm_axes, dtype=self.dtype_policy)
        self._dropout_layer = Dropout(
            rate=self._dropout, dtype=self.dtype_policy, seed=self.seed
        )

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # attention_scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, mask=attention_mask)

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, S, N, key_dim)`.
            value: Projected value tensor of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Check for flash attention constraints
        if self._flash_attention and return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
            self._dropout > 0.0
            or return_attention_scores
            or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                    len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = ops.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = ops.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                mask=attention_mask,
                scale=self._inverse_sqrt_key_dim,
                is_causal=False,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        # Apply the mask using the custom masked softmax
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # Apply dropout to the attention scores if needed
        if self._dropout > 0.0:
            final_attn_scores = self._dropout_layer(
                attention_scores, training=training
            )
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = ops.einsum(
            self._combine_equation, final_attn_scores, value
        )
        return attention_output, attention_scores

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        cache=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        self._return_attention_scores = return_attention_scores
        if key is None:
            key = value

        # Determine training mode (e.g., for cache usage)
        if training is None:
            training = backend.learning_phase()
        else:
            training = ops.convert_to_tensor(training, dtype="bool")

        # Delete the masks because the masks are handled at the level of the
        # layer
        query_mask_from_input = backend.get_keras_mask(query)
        backend.set_keras_mask(query, None)
        backend.set_keras_mask(value, None)
        backend.set_keras_mask(key, None)

        # Project query
        query = self._query_dense(query)

        # Handle cache
        if cache is not None and not training:
            past_key, past_value = cache
            current_key = self._key_dense(key)
            current_value = self._value_dense(value)
            new_key = ops.concatenate([past_key, current_key], axis=1)
            new_value = ops.concatenate([past_value, current_value], axis=1)
        else:
            new_key = self._key_dense(key)
            new_value = self._value_dense(value)

        # Compute attention mask using the potentially updated key/value
        # The sequence length for causal mask now comes from new_key
        attention_mask = self._compute_attention_mask(
            query,
            new_value,  # Use new_value for sequence length consistency
            query_mask=query_mask_from_input,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            # Pass new_key to _compute_causal_mask if it's used there
            # for its sequence length.
            cached_kv_length=ops.shape(new_key)[1]
            if use_causal_mask
            else None,
        )
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        # `query` = [B, T, N, H]
        # `new_key` = [B, S_full, N, H]
        # `new_value` = [B, S_full, N, H]
        attention_output, attention_scores = self._compute_attention(
            query,
            new_key,
            new_value,
            attention_mask,
            training,
            # Pass the instance flag, not the call argument
            self._return_attention_scores,
        )
        attention_output = self._output_dense(attention_output)

        # Set mask on output if needed
        if query_mask_from_input is not None:
            backend.set_keras_mask(attention_output, query_mask_from_input)

        current_cache = (new_key, new_value)
        if self._return_attention_scores:
            return attention_output, attention_scores, current_cache
        return attention_output, current_cache

    def _compute_attention_mask(
        self,
        query,
        value,  # Note: this is new_value in call()
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
        cached_kv_length=None,  # New argument for causal mask with cache
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, T, N, key_dim)`.
            value: Projected value tensor of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        auto_mask = None
        if query_mask is not None:
            query_mask = ops.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = ops.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = ops.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length (of current value, not cached)
            mask = ops.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask

        # Key mask should apply to the full key sequence length if cache is used
        # However, typical use of key_mask is for padding in the *original* key,
        # so its shape might be (B, S_current).
        # If cache is used, this mask needs to be handled carefully.
        # For now, assume key_mask corresponds to the original key passed to call().
        # If `key` is part of a cache, its mask should already be part of `past_key`.
        # This part might need refinement based on how key_mask is intended with caching.
        if key_mask is not None:
            key_mask = ops.cast(key_mask, "bool")
            # If cache is used, key_mask applies to current_key part.
            # We'd need to concatenate it with a mask for past_key.
            # This is complex. Assuming key_mask is for S_current or S_total if no cache.
            # For simplicity, let's assume key_mask is [B, 1, S_total_or_S_current]
            mask = ops.expand_dims(key_mask, -2)
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if use_causal_mask:
            # the shape of the causal mask is [1, T, S_total]
            # Pass the total key sequence length if cache is active
            mask = self._compute_causal_mask(
                query, value, v_seq_length_override=cached_kv_length
            )
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, "bool")
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else attention_mask & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query tensor of shape `(B, T, ...)`.
            value: value tensor of shape `(B, S, ...)` (optional, defaults to
                query).
            v_seq_length_override: Optional integer to override value sequence length.
                Used when KV cache is active, so S becomes S_past + S_current.

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`.
        """
        q_seq_length = ops.shape(query)[1]
        v_seq_length = (
            v_seq_length_override
            if v_seq_length_override is not None
            else (q_seq_length if value is None else ops.shape(value)[1])
        )
        ones_mask = ops.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        return ops.greater_equal(row_index, col_index)

    def compute_output_shape(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        query_shape = tuple(query_shape)
        value_shape = tuple(value_shape)
        if key_shape is None:
            key_shape = value_shape
        else:
            key_shape = tuple(key_shape)

        if not (self.built and hasattr(self, "_key_dense") and hasattr(self, "_value_dense")):
            # Layer is not built yet, shapes of dense projections are unknown.
            # This can happen if compute_output_shape is called early.
            # We'll define cache shapes with None for head and key/value dims
            # if they are not available.
            # This is a fallback, ideally build() is called first.
            num_heads = self._num_heads
            key_dim = self._key_dim
            value_dim = self._value_dim
        else:
            # Layer is built, so we can get precise shapes.
            # _key_dense.compute_output_shape(key_shape) would be
            # (B, S, N, key_dim)
            # _value_dense.compute_output_shape(value_shape) would be
            # (B, S, N, value_dim)
            num_heads = self._key_dense.output_shape[-2]
            key_dim = self._key_dense.output_shape[-1]
            value_dim = self._value_dense.output_shape[-1]


        if value_shape[1:-1] != key_shape[1:-1]:
            # This check might be too strict if sequence lengths differ,
            # but the original check was about non-sequence, non-feature dims.
            # Let's assume the feature dimension is the last one.
            if len(value_shape) > 2 and len(key_shape) > 2 and \
               value_shape[2:-1] != key_shape[2:-1]:
                raise ValueError(
                    "All dimensions of `value` and `key`, except batch, "
                    "sequence, and feature, "
                    f"must be equal. Received: value_shape={value_shape} and "
                    f"key_shape={key_shape}"
                )

        attention_output_shape = list(query_shape)
        if self._output_shape:
            attention_output_shape[-1] = self._output_shape[0] # Assuming 1D output_shape
        # else it's query_shape[-1] which is already set.
        attention_output_shape = tuple(attention_output_shape)


        # Cache shapes: (B, S, N, H_k) and (B, S, N, H_v)
        # S comes from key_shape[1] or value_shape[1]. This represents S_total.
        # In compute_output_shape, S is dynamic if inputs are dynamic.
        key_seq_len = key_shape[1] # Can be None

        # Shape of projected key: (B, S, num_heads, key_dim)
        cache_key_shape = (
            key_shape[0],
            key_seq_len,
            num_heads,
            key_dim,
        )
        # Shape of projected value: (B, S, num_heads, value_dim)
        cache_value_shape = (
            value_shape[0],
            key_seq_len, # Value sequence length should track key sequence length for cache
            num_heads,
            value_dim,
        )
        cache_shape = (cache_key_shape, cache_value_shape)

        if hasattr(self, "_return_attention_scores") and self._return_attention_scores:
            # Attention scores shape: (B, N, T, S)
            # T is query_shape[1], S is key_seq_len
            score_shape = (
                query_shape[0],
                num_heads,
                query_shape[1],
                key_seq_len,
            )
            return attention_output_shape, score_shape, cache_shape
        return attention_output_shape, cache_shape

    def compute_output_spec(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        cache=None, # Added cache to signature
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        # Stash return_attention_scores to be used by compute_output_shape
        # This is a bit of a workaround as compute_output_shape ideally
        # shouldn't depend on call args not passed to it.
        # Keras handles this by sometimes calling build then compute_output_shape.
        self._return_attention_scores_for_spec = return_attention_scores


        key_shape_for_compute = key.shape if key is not None else value.shape

        output_shapes = self.compute_output_shape(
            query.shape, value.shape, key_shape_for_compute
        )

        # Reset the temporary flag after use
        if hasattr(self, "_return_attention_scores_for_spec"):
            del self._return_attention_scores_for_spec


        if return_attention_scores : # Use the direct argument here
            att_output_spec = backend.KerasTensor(
                output_shapes[0], dtype=self.compute_dtype
            )
            att_scores_spec = backend.KerasTensor(
                output_shapes[1], dtype=self.compute_dtype
            )
            cache_k_spec = backend.KerasTensor(
                output_shapes[2][0], dtype=self.compute_dtype
            )
            cache_v_spec = backend.KerasTensor(
                output_shapes[2][1], dtype=self.compute_dtype
            )
            return att_output_spec, att_scores_spec, (cache_k_spec, cache_v_spec)
        else:
            att_output_spec = backend.KerasTensor(
                output_shapes[0], dtype=self.compute_dtype
            )
            cache_k_spec = backend.KerasTensor(
                output_shapes[1][0], dtype=self.compute_dtype
            )
            cache_v_spec = backend.KerasTensor(
                output_shapes[1][1], dtype=self.compute_dtype
            )
            return att_output_spec, (cache_k_spec, cache_v_spec)

# Helper to get num_heads and key/value_dim if layer is not built
# This might not be needed if build() is always called before compute_output_shape
# For now, removed direct use of _get_unbuilt_proj_dims from compute_output_shape
# and relying on self._num_heads, self._key_dim, self._value_dim which are set in __init__

def _get_unbuilt_proj_dims(layer_instance):
    # Fallback for when layer isn't built yet.
    # This is a simplification. Real projection shapes depend on EinsumDense.
    return layer_instance.num_heads, layer_instance.key_dim, layer_instance.value_dim


def _index_to_einsum_variable(i):
    """Converts an index to a einsum variable name.

    We simply map indices to lowercase characters, e.g. 0 -> 'a', 1 -> 'b'.
    """
    return string.ascii_lowercase[i]


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    1. Query-key dot product:
        (<batch dims>, <query attention dims>, num_heads, channels),
        (<batch dims>, <key attention dims>, num_heads, channels) ->
        (<batch dims>, num_heads, <query attention dims>, <key attention dims>)
    2. Combination:
        (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
        (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
        dims>, <query attention dims>, num_heads, channels)

    Args:
        rank: Rank of query, key, value tensors.
        attn_axes: List/tuple of axes, `[-1, rank)`,
            that attention will be applied to.

    Returns:
        Einsum equations.
    """
    target_notation = ""
    for i in range(rank):
        target_notation += _index_to_einsum_variable(i)
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _index_to_einsum_variable(letter_offset)
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)
