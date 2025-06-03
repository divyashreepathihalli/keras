import math

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


@keras_export("keras.layers.GroupQueryAttention")
class GroupedQueryAttention(Layer):
    """Grouped Query Attention layer.

    This is an implementation of grouped-query attention introduced by
    [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here
    `num_key_value_heads` denotes number of groups, setting
    `num_key_value_heads` to 1 is equivalent to multi-query attention, and
    when `num_key_value_heads` is equal to `num_query_heads` it is equivalent
    to multi-head attention.

    This layer first projects `query`, `key`, and `value` tensors. Then, `key`
    and `value` are repeated to match the number of heads of `query`.

    Then, the `query` is scaled and dot-producted with `key` tensors. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities and concatenated back to a single
    tensor.

    Args:
        head_dim: Size of each attention head.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key and value attention heads.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
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
        query: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `target_seq_len` is the length of
            target sequence, and `feature_dim` is dimension of feature.
        value: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `source_seq_len` is the length of
            source sequence, and `feature_dim` is dimension of feature.
        key: Optional key tensor of shape
            `(batch_dim, source_seq_len, feature_dim)`. If not given, will use
            `value` for both `key` and `value`, which is most common case.
        attention_mask: A boolean mask of shape
            `(batch_dim, target_seq_len, source_seq_len)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, where 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output
            should be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: Result of the computation, of shape
            `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len`
            is for target sequence length and `feature_dim` is the query input
            last dim.
        attention_scores: (Optional) attention coefficients of shape
            `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        dropout=0.0,
        use_bias=True,
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
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_query_heads` must be divisible by `num_key_value_heads`."
            )
        self.num_repeats = num_query_heads // num_key_value_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self._flash_attention = flash_attention or is_flash_attention_enabled()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seed = seed

        self._inverse_sqrt_head_dim = 1.0 / math.sqrt(float(self.head_dim))
        self._return_attention_scores = False # Will be set in call
        self.built_from_signature = False # Ensure build is properly called

        # Check for flash attention constraints
        if self._flash_attention and self.dropout > 0.0:
            raise ValueError(
                "Dropout is not supported when flash attention is enabled. "
                "Please set dropout to 0.0 to use flash attention."
            )

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        key_shape = value_shape if key_shape is None else key_shape
        self.feature_dim = query_shape[-1]
        self._query_dense = EinsumDense(
            "bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            bias_axes="uh" if self.use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)

        self._key_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)

        self._value_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        self._softmax = Softmax(axis=-1, dtype=self.dtype_policy)
        self._dropout_layer = Dropout(
            rate=self.dropout, dtype=self.dtype_policy, seed=self.seed
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self._output_dense = EinsumDense(
            "bquh,uhm->bqm",
            output_shape=(None, self.feature_dim),
            bias_axes="m" if self.use_bias else None,
            name="attention_output",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype_policy,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self.kernel_initializer.__class__.from_config(
            self.kernel_initializer.get_config()
        )
        bias_initializer = self.bias_initializer.__class__.from_config(
            self.bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        self._return_attention_scores = return_attention_scores # Instance flag
        if key is None:
            key = value

        if training is None:
            training = backend.learning_phase()
        else:
            training = ops.convert_to_tensor(training, dtype="bool")

        query_orig_mask = backend.get_keras_mask(query) # Store original mask

        query = self._query_dense(query) # (B, T, num_query_heads, head_dim)

        # K, V are initially (B, S, num_key_value_heads, head_dim)
        current_key_unrepeated = self._key_dense(key)
        current_value_unrepeated = self._value_dense(value)

        if cache is not None and not training:
            past_key_unrepeated, past_value_unrepeated = cache
            new_key_unrepeated = ops.concatenate(
                [past_key_unrepeated, current_key_unrepeated], axis=1
            )
            new_value_unrepeated = ops.concatenate(
                [past_value_unrepeated, current_value_unrepeated], axis=1
            )
        else:
            new_key_unrepeated = current_key_unrepeated
            new_value_unrepeated = current_value_unrepeated

        # Determine full sequence length for masks
        # This is S_total for causal mask if cache is used
        current_total_kv_seq_len = ops.shape(new_key_unrepeated)[1]

        attention_mask = self._compute_attention_mask(
            query, # Query shape (B, T, ...)
            new_value_unrepeated, # Value shape (B, S_total, num_kv_heads, head_dim) for length calc
            query_mask=query_orig_mask, # Mask for query (B, T)
            value_mask=value_mask, # Mask for current value (B, S_current)
            key_mask=key_mask,     # Mask for current key (B, S_current)
            attention_mask=attention_mask, # External mask (B, T, S_total or S_current)
            use_causal_mask=use_causal_mask,
            # Pass the total key sequence length for causal mask generation
            cached_kv_length=current_total_kv_seq_len if use_causal_mask else None
        )

        # Repeat K, V after potential concatenation with cache
        # Resulting K, V for attention: (B, S_total, num_query_heads, head_dim)
        key_for_attention = ops.repeat(
            new_key_unrepeated, self.num_repeats, axis=2
        )
        value_for_attention = ops.repeat(
            new_value_unrepeated, self.num_repeats, axis=2
        )

        output, scores = self._compute_attention(
            query,
            key_for_attention,
            value_for_attention,
            attention_mask=attention_mask,
            training=training,
        )

        output = self._output_dense(
            output
        )  # (batch_dim, target_seq_len, feature_dim)

        current_cache_unrepeated = (new_key_unrepeated, new_value_unrepeated)
        if self._return_attention_scores:
            return output, scores, current_cache_unrepeated
        return output, current_cache_unrepeated

    def _compute_attention_mask(
        self,
        query,
        value, # This is new_value_unrepeated
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
        cached_kv_length=None, # New argument for causal mask with cache
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
            # This mask should apply to the S dimension. If cache is used,
            # this mask refers to S_current. The final attention_mask will be
            # against S_total. Careful handling of mask concatenation might be
            # needed if value_mask is for S_current and cache is used.
            # For now, assume value_mask is [B, 1, S_current_or_S_total]
            mask = ops.expand_dims(value_mask, -2)
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if key_mask is not None:
            key_mask = ops.cast(key_mask, "bool")
            # Similar to value_mask, if cache is used, this applies to S_current.
            # For now, assume key_mask is [B, 1, S_current_or_S_total]
            mask = ops.expand_dims(key_mask, -2)
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if use_causal_mask:
            # the shape of the causal mask is [1, T, S_total]
            mask = self._compute_causal_mask(
                query, value, v_seq_length_override=cached_kv_length
            )
            auto_mask = mask if auto_mask is None else auto_mask & mask

        # `attention_mask` arg is assumed to be [B, T, S_total_or_S_current]
        # If cache is used, S should be S_total.
        # If S_current, it needs careful concatenation like key/value_mask.
        # Current assumption: external `attention_mask` is correctly shaped for S_total if cache.
        if auto_mask is not None:
            attention_mask = (
                auto_mask
                if attention_mask is None
                else ops.cast(attention_mask, bool) & auto_mask
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
                query). S is S_total_unrepeated for length calculation.
            v_seq_length_override: Optional integer to override value sequence length.

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`. S is S_total.
        """
        q_seq_length = ops.shape(query)[1]
        # value here is new_value_unrepeated for GQA
        # its shape[1] is S_total (unrepeated)
        v_seq_length = (
            v_seq_length_override
            if v_seq_length_override is not None
            else (q_seq_length if value is None else ops.shape(value)[1])
        )
        ones_mask = ops.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        return ops.greater_equal(row_index, col_index)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ): # key and value are already repeated here
        # Check for flash attention constraints
        if self._flash_attention and self._return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
            self.dropout > 0.0
            or self._return_attention_scores
            or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -1 * 2 - 1
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
                scale=self._inverse_sqrt_head_dim,
                is_causal=False,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_head_dim, query.dtype)
        )
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        scores = ops.einsum(
            self._dot_product_equation, query, key
        )  # (batch_dim, query_heads, target_seq_len, source_seq_len)
        scores = self._masked_softmax(scores, attention_mask=attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout > 0.0:
            scores_dropout = self._dropout_layer(scores, training=training)
        else:
            scores_dropout = scores
        output = ops.einsum(self._combine_equation, scores_dropout, value)
        return output, scores

    def _masked_softmax(self, scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -1 * 2 - 1
            for _ in range(len(scores.shape) - len(attention_mask.shape)):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(scores, mask=attention_mask)

    def compute_output_shape(
        self,
        query_shape,
        value_shape,
        key_shape=None,
        # cache_shape is not passed but computed based on inputs
    ):
        if key_shape is None:
            key_shape = value_shape

        query_shape = tuple(query_shape)
        value_shape = tuple(value_shape)
        key_shape = tuple(key_shape)

        # Output shape is same as query_shape, but last dim is feature_dim
        attention_output_shape = query_shape[:-1] + (self.feature_dim,)

        # Cache shapes are for unrepeated K/V.
        # (B, S, num_key_value_heads, head_dim)
        # S comes from key_shape[1] or value_shape[1].
        # This represents S_total in call, or S_current if no cache.
        # For compute_output_shape, it's based on input spec.
        key_seq_len = key_shape[1] # Can be None

        cache_key_shape = (
            key_shape[0], # Batch size
            key_seq_len,  # Sequence length (can be None)
            self.num_key_value_heads,
            self.head_dim,
        )
        cache_value_shape = (
            value_shape[0], # Batch size
            key_seq_len,    # Sequence length (should track key's for cache)
            self.num_key_value_heads,
            self.head_dim,
        )
        cache_shape = (cache_key_shape, cache_value_shape)

        if hasattr(self, "_return_attention_scores") and self._return_attention_scores:
            # Attention scores shape: (B, num_query_heads, T, S)
            # T is query_shape[1], S is key_seq_len (S_total)
            score_shape = (
                query_shape[0],
                self.num_query_heads,
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
        self._return_attention_scores = return_attention_scores

        key_shape_for_compute = key.shape if key is not None else value.shape

        output_shapes = self.compute_output_shape(
            query.shape, value.shape, key_shape_for_compute
        )

        # output_shapes can be (att_out_shape, cache_shape) or (att_out_shape, score_shape, cache_shape)

        if return_attention_scores:
            att_output_spec = backend.KerasTensor(output_shapes[0], dtype=query.dtype)
            att_scores_spec = backend.KerasTensor(output_shapes[1], dtype=query.dtype)
            cache_k_spec = backend.KerasTensor(output_shapes[2][0], dtype=key.dtype if key is not None else value.dtype)
            cache_v_spec = backend.KerasTensor(output_shapes[2][1], dtype=value.dtype)
            return att_output_spec, att_scores_spec, (cache_k_spec, cache_v_spec)
        else:
            att_output_spec = backend.KerasTensor(output_shapes[0], dtype=query.dtype)
            cache_k_spec = backend.KerasTensor(output_shapes[1][0], dtype=key.dtype if key is not None else value.dtype)
            cache_v_spec = backend.KerasTensor(output_shapes[1][1], dtype=value.dtype)
            return att_output_spec, (cache_k_spec, cache_v_spec)

    def get_config(self):
        config = {
            "head_dim": self.head_dim,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "use_bias": self.use_bias,
            "dropout": self.dropout,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
