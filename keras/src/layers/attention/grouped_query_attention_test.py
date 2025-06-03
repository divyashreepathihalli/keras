import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import testing
from keras.src.backend.config import disable_flash_attention
from keras.src.backend.config import enable_flash_attention
from keras.src.backend.config import is_flash_attention_enabled


class GroupedQueryAttentionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Flash attention is a newly introduced feature. We need to disable it
        # for testing purposes.
        disable_flash_attention()

    def tearDown(self):
        enable_flash_attention()
        return super().tearDown()

    def test_basics(self):
        self.assertFalse(is_flash_attention_enabled())
        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "use_bias": False,
                "dropout": 0.5,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    def test_basics_with_flash_attention(self):
        enable_flash_attention()
        init_kwargs = {
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "dtype": "float16",
        }
        input_shape = {
            "query_shape": (2, 8, 16),
            "value_shape": (2, 4, 16),
        }
        expected_output_shape = (2, 8, 16)
        if backend.backend() in ("tensorflow", "numpy"):
            self.skipTest(
                "Flash attention is not supported in tensorflow and numpy "
                "backends."
            )
        elif backend.backend() == "torch":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "PyTorch version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if (
                    "Flash attention is not supported with the provided inputs"
                    in str(e.args[0])
                ):
                    self.assertTrue(
                        (
                            "Flash attention is not supported with the "
                            "provided inputs"
                        )
                        in str(e.args[0])
                    )
        elif backend.backend() == "jax":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if "cuDNN" in str(e.args[0]):
                    self.assertTrue("cuDNN is not detected." in str(e.args[0]))
                elif "Require at least" in str(e.args[0]):
                    self.assertTrue(
                        "Require at least Ampere arch to run" in str(e.args[0])
                    )
                elif "Flash attention" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
                    )

    @parameterized.named_parameters(
        ("without_key_proj_mha", (4, 8), (2, 8), None, 2, 2),
        ("with_key_proj_mha", (4, 8), (2, 8), (2, 3), 2, 2),
        ("without_key_proj_gqa", (4, 8), (2, 8), None, 4, 2),
        ("with_key_proj_gqa", (4, 8), (2, 8), (2, 3), 4, 2),
        ("without_key_value_proj_mqa", (4, 8), (2, 8), None, 4, 1),
        ("with_key_value_proj_mqa", (4, 8), (2, 8), (2, 3), 4, 1),
    )
    def test_compute_output_shape(
        self,
        query_dims,
        value_dims,
        key_dims,
        num_query_heads,
        num_key_value_heads,
    ):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=2,
        )
        batch_size = 7
        query_shape = (batch_size,) + query_dims
        value_shape = (batch_size,) + value_dims
        key_shape = (batch_size,) + key_dims if key_dims else None

        query = np.ones(query_shape)
        value = np.ones(value_shape)
        key = np.ones(key_shape) if key_shape else None
        output = layer(query=query, value=value, key=key)
        comp_output_shape = layer.compute_output_shape(
            query_shape, value_shape, key_shape
        )
        self.assertEqual(output.shape, comp_output_shape)

    @parameterized.named_parameters(
        ("query_value_dim_mismatch", (2, 4, 8), (2, 2, 7), 2),
        ("key_value_dim_mismatch", (2, 4, 8), (2, 2, 8), (2, 1, 7)),
    )
    def test_shape_mismatch_error(self, query_shape, value_shape, key_shape):
        """Test dimension mismatches"""
        layer = layers.GroupedQueryAttention(
            num_query_heads=4,
            num_key_value_heads=4,
            head_dim=2,
        )
        with self.assertRaisesRegex(ValueError, r"must be equal"):
            layer.compute_output_shape(query_shape, value_shape, key_shape)

    def test_initializer(self):
        # Test with a specified initializer.
        layer = layers.GroupedQueryAttention(
            num_query_heads=16,
            num_key_value_heads=16,
            head_dim=64,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        layer.build((2, 4, 8), (2, 4, 8))

        # Make sure the sub layers have different kernel init value.
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._key_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._value_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._output_dense.kernel,
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_query_mask_propagation(self):
        """Test automatic propagation of the query's mask."""
        try:
            layer = layers.GroupedQueryAttention(
                num_query_heads=2, num_key_value_heads=2, head_dim=2
            )
            self.assertTrue(layer.supports_masking)
            query = np.array(
                [[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]]
            )
            masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
            value = np.random.normal(size=(3, 3, 8))
            output = layer(query=masked_query, value=value)
        except RuntimeError as e:
            if e.args[0].startswith(
                "(*bias): last dimension must be contiguous"
            ):
                self.skipTest(
                    "PyTorch errors out on GPU: issue to track bug is here "
                    "https://github.com/keras-team/keras/issues/20459"
                )
        self.assertAllClose(masked_query._keras_mask, output._keras_mask)

    @parameterized.named_parameters(("causal", True), ("not_causal", 0))
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_masking(self, use_causal_mask):
        """Test that the value and causal masks are taken into account."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2
        )
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
        masked_value = layers.Embedding(6, 8, mask_zero=True)(value)
        output = layer(
            query=masked_query,
            value=masked_value,
            use_causal_mask=use_causal_mask,
        )
        mask = np.array(
            [[[1, 1, 0]] * 3 + [[0, 0, 0]] * 2]
            + [[[1, 0, 0]] * 5]
            + [[[1, 1, 1]] + [[0, 0, 0]] * 4]
        ).astype(bool)
        if use_causal_mask:
            mask = mask & np.array(
                [[[1, 0, 0], [1, 1, 0]] + [[1, 1, 1]] * 3]
            ).astype(bool)
        del masked_query._keras_mask
        del masked_value._keras_mask
        output_with_manual_mask = layer(
            query=masked_query, value=masked_value, attention_mask=mask
        )
        self.assertAllClose(output, output_with_manual_mask)

    @parameterized.named_parameters(
        ("disable_flash_attention", False), ("enable_flash_attention", True)
    )
    def test_correctness(self, flash_attention):
        if flash_attention:
            # Let the backend decide whether to use flase attention
            enable_flash_attention()
        dtype = "float16"  # Flash attention only accepts float16/bfloat16
        head_dim = 8  # key_dim % 8 == 0 to enable flash attention
        num_query_heads = num_key_value_heads = 8

        query = np.identity(head_dim)[np.newaxis, ...]
        key = np.identity(head_dim)[np.newaxis, ...]
        value = (
            np.reshape(np.arange(head_dim * head_dim), (1, head_dim, head_dim))
            / 100.0  # Prevent overflow/underflow
        )

        # Setup layer.
        layer = layers.GroupedQueryAttention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            dtype=dtype,
        )
        layer.build(query.shape, key.shape, value.shape)

        # Set layer weights.
        kernel = np.identity(head_dim)
        # To get an identity kernel we need to add a head dim and repeat on it.
        kernel = np.repeat(kernel[:, np.newaxis, :], num_query_heads, axis=1)
        # Zeros for all biases.
        bias = np.zeros((num_query_heads, head_dim))
        output_bias = np.zeros((head_dim,))
        layer.set_weights([kernel, bias] * 3 + [kernel, output_bias])

        # Call layer and assert output.
        expected_output = np.array(
            [2.406, 2.440, 2.473, 2.504, 2.535, 2.568, 2.602, 2.633]
        )
        expected_output = np.tile(
            expected_output[np.newaxis, :, np.newaxis], (1, 1, head_dim)
        )
        expected_score = np.array(
            [
                [0.1187] * 0 + [0.1691] + [0.1187] * 7,
                [0.1187] * 1 + [0.1691] + [0.1187] * 6,
                [0.1187] * 2 + [0.1691] + [0.1187] * 5,
                [0.1187] * 3 + [0.1691] + [0.1187] * 4,
                [0.1187] * 4 + [0.1691] + [0.1187] * 3,
                [0.1187] * 5 + [0.1691] + [0.1187] * 2,
                [0.1187] * 6 + [0.1691] + [0.1187] * 1,
                [0.1187] * 7 + [0.1691] + [0.1187] * 0,
            ]
        )
        expected_score = np.tile(
            expected_score[np.newaxis, np.newaxis, ...], (1, head_dim, 1, 1)
        )
        if flash_attention:
            output = layer(query=query, value=value, key=key)
            self.assertAllClose(output, expected_output, atol=1e-2)
        else:
            output, scores = layer(
                query=query,
                value=value,
                key=key,
                return_attention_scores=True,
            )
            self.assertAllClose(output, expected_output, atol=1e-2)
            self.assertAllClose(scores, expected_score, atol=1e-2)

    def test_flash_attention_with_errors(self):
        if backend.backend() in ("numpy", "tensorflow"):
            pytest.skip(
                reason=(
                    "Flash attention is not supported on tensorflow and numpy."
                )
            )
        # Check `flash_attention=True` and `dropout=0.1`
        with self.assertRaisesRegex(
            ValueError,
            "Dropout is not supported when flash attention is enabled.",
        ):
            layer = layers.GroupedQueryAttention(
                head_dim=2,
                num_query_heads=2,
                num_key_value_heads=2,
                flash_attention=True,
                dropout=0.1,
            )

        # Check `flash_attention=True` and `return_attention_scores=True`
        layer = layers.GroupedQueryAttention(
            head_dim=2,
            num_query_heads=2,
            num_key_value_heads=2,
            flash_attention=True,
        )
        self.assertTrue(layer._flash_attention)
        query = np.random.random((2, 4, 8))
        value = np.random.random((2, 4, 8))
        with self.assertRaisesRegex(
            ValueError,
            "Returning attention scores is not supported when flash "
            "attention is enabled. Please disable flash attention to access"
            " attention scores.",
        ):
            layer(query=query, value=value, return_attention_scores=True)

    def test_kv_cache_usage(self):
        num_query_heads = 4
        num_key_value_heads = 2 # GQA specific
        head_dim = 2
        feature_dim = num_query_heads * head_dim # 8, this is for query's expected last dim
                                                # For K/V, feature_dim is num_key_value_heads * head_dim = 4

        layer = layers.GroupedQueryAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        batch_size = 2
        target_seq_len_1 = 1 # Query for first step
        # For GQA, value/key input feature_dim corresponds to num_key_value_heads * head_dim
        kv_feature_dim = num_key_value_heads * head_dim
        source_seq_len_1 = 3 # Initial KV sequence length

        query1 = backend.random.normal(shape=(batch_size, target_seq_len_1, feature_dim))
        # Value for GQA has feature_dim based on num_key_value_heads
        value1 = backend.random.normal(shape=(batch_size, source_seq_len_1, kv_feature_dim))
        key1 = value1

        # First call (populate cache)
        # training=False is important for cache to be used.
        output1, cache1 = layer(query1, value1, key=key1, cache=None, training=False)

        self.assertEqual(output1.shape, (batch_size, target_seq_len_1, feature_dim))
        self.assertIsInstance(cache1, tuple)
        self.assertEqual(len(cache1), 2)

        # cache1[0] is key_cache, cache1[1] is value_cache (unrepeated)
        # Expected shape: (batch_size, source_seq_len, num_key_value_heads, head_dim)
        self.assertEqual(cache1[0].shape, (batch_size, source_seq_len_1, num_key_value_heads, head_dim))
        self.assertEqual(cache1[1].shape, (batch_size, source_seq_len_1, num_key_value_heads, head_dim))

        # Second call (use cache)
        target_seq_len_2 = 1 # Query for second step (new token)
        source_seq_len_2 = 1 # New KV for this step

        query2 = backend.random.normal(shape=(batch_size, target_seq_len_2, feature_dim))
        value2 = backend.random.normal(shape=(batch_size, source_seq_len_2, kv_feature_dim))
        key2 = value2

        output2, cache2 = layer(query2, value2, key=key2, cache=cache1, training=False)

        self.assertEqual(output2.shape, (batch_size, target_seq_len_2, feature_dim))
        expected_total_kv_len = source_seq_len_1 + source_seq_len_2
        self.assertEqual(cache2[0].shape, (batch_size, expected_total_kv_len, num_key_value_heads, head_dim))
        self.assertEqual(cache2[1].shape, (batch_size, expected_total_kv_len, num_key_value_heads, head_dim))

        # Third call (use cache, return attention scores)
        target_seq_len_3 = 1
        source_seq_len_3 = 1
        query3 = backend.random.normal(shape=(batch_size, target_seq_len_3, feature_dim))
        value3 = backend.random.normal(shape=(batch_size, source_seq_len_3, kv_feature_dim))
        key3 = value3

        output3, scores3, cache3 = layer(
            query3, value3, key=key3, cache=cache1, # Using cache1 and adding value3
            training=False, return_attention_scores=True
        )

        self.assertEqual(output3.shape, (batch_size, target_seq_len_3, feature_dim))

        expected_kv_len_for_scores3 = source_seq_len_1 + source_seq_len_3
        # scores shape: (batch, num_query_heads, target_seq_len, source_seq_len_total_repeated)
        self.assertEqual(scores3.shape, (batch_size, num_query_heads, target_seq_len_3, expected_kv_len_for_scores3))

        self.assertEqual(cache3[0].shape, (batch_size, expected_kv_len_for_scores3, num_key_value_heads, head_dim))
        self.assertEqual(cache3[1].shape, (batch_size, expected_kv_len_for_scores3, num_key_value_heads, head_dim))


    def test_kv_cache_with_causal_mask(self):
        num_query_heads = 4
        num_key_value_heads = 2
        head_dim = 2
        feature_dim = num_query_heads * head_dim
        kv_feature_dim = num_key_value_heads * head_dim

        layer = layers.GroupedQueryAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        batch_size = 2
        target_seq_len_1 = 1
        source_seq_len_1 = 3

        query1 = backend.random.normal(shape=(batch_size, target_seq_len_1, feature_dim))
        value1 = backend.random.normal(shape=(batch_size, source_seq_len_1, kv_feature_dim))

        output1, cache1 = layer(
            query1, value1, key=value1, cache=None, training=False, use_causal_mask=True
        )
        self.assertEqual(output1.shape, (batch_size, target_seq_len_1, feature_dim))
        self.assertEqual(cache1[0].shape, (batch_size, source_seq_len_1, num_key_value_heads, head_dim))
        self.assertEqual(cache1[1].shape, (batch_size, source_seq_len_1, num_key_value_heads, head_dim))

        target_seq_len_2 = 1
        source_seq_len_2 = 1
        query2 = backend.random.normal(shape=(batch_size, target_seq_len_2, feature_dim))
        value2 = backend.random.normal(shape=(batch_size, source_seq_len_2, kv_feature_dim))

        output2, cache2 = layer(
            query2, value2, key=value2, cache=cache1, training=False, use_causal_mask=True
        )
        self.assertEqual(output2.shape, (batch_size, target_seq_len_2, feature_dim))
        expected_total_kv_len = source_seq_len_1 + source_seq_len_2
        self.assertEqual(cache2[0].shape, (batch_size, expected_total_kv_len, num_key_value_heads, head_dim))
        self.assertEqual(cache2[1].shape, (batch_size, expected_total_kv_len, num_key_value_heads, head_dim))

        output3, scores3, cache3 = layer(
            query2, value2, key=value2, cache=cache1,
            training=False, use_causal_mask=True, return_attention_scores=True
        )
        self.assertEqual(output3.shape, (batch_size, target_seq_len_2, feature_dim))
        self.assertEqual(scores3.shape, (batch_size, num_query_heads, target_seq_len_2, expected_total_kv_len))
        self.assertEqual(cache3[0].shape, (batch_size, expected_total_kv_len, num_key_value_heads, head_dim))

    @parameterized.parameters([((1, 2, 8), (1,2,4)), ((2, 3, 16), (2,3,8))]) # query_shape, kv_shape (dim changed)
    def test_symbolic_return_attention_scores_and_cache(self, q_shape_tuple, kv_shape_tuple):
        gqa = layers.GroupedQueryAttention(num_query_heads=4, num_key_value_heads=2, head_dim=int(q_shape_tuple[-1]/4))

        q_input = layers.Input(batch_shape=q_shape_tuple)
        v_input = layers.Input(batch_shape=kv_shape_tuple) # Value/Key have different dim for GQA usually

        # Symbolic cache input
        # Shape of cache elements: (batch_size, seq_len, num_key_value_heads, head_dim)
        cache_k_shape = (q_shape_tuple[0], None, gqa.num_key_value_heads, gqa.head_dim)
        cache_v_shape = (q_shape_tuple[0], None, gqa.num_key_value_heads, gqa.head_dim)

        cache_k_sym = backend.KerasTensor(cache_k_shape, dtype=q_input.dtype)
        cache_v_sym = backend.KerasTensor(cache_v_shape, dtype=v_input.dtype)
        symbolic_cache_input = (cache_k_sym, cache_v_sym)

        result = gqa(q_input, v_input, cache=symbolic_cache_input, return_attention_scores=True)
        self.assertLen(result, 3)
        self.assertEqual(len(result[2]), 2)

        # Eager call
        q_np = np.random.random(q_shape_tuple).astype(backend.floatx())
        v_np = np.random.random(kv_shape_tuple).astype(backend.floatx())

        past_seq_len = 2
        eager_cache_k_shape = (q_shape_tuple[0], past_seq_len, gqa.num_key_value_heads, gqa.head_dim)
        eager_cache_v_shape = (q_shape_tuple[0], past_seq_len, gqa.num_key_value_heads, gqa.head_dim)
        eager_cache_k = backend.ops.zeros(eager_cache_k_shape, dtype=q_np.dtype)
        eager_cache_v = backend.ops.zeros(eager_cache_v_shape, dtype=v_np.dtype)
        eager_cache_input = (eager_cache_k, eager_cache_v)

        out_with_cache = gqa(q_np, v_np, cache=eager_cache_input, return_attention_scores=True, training=False)
        self.assertLen(out_with_cache, 3)
        self.assertEqual(len(out_with_cache[2]), 2)

        self.assertEqual(result[0].shape, out_with_cache[0].shape)
        self.assertEqual(len(result[1].shape), len(out_with_cache[1].shape))
        self.assertEqual(result[1].shape[:-1], out_with_cache[1].shape[:-1])

        self.assertEqual(len(result[2][0].shape), len(out_with_cache[2][0].shape))
        self.assertEqual(result[2][0].shape[0], out_with_cache[2][0].shape[0])
        self.assertEqual(result[2][0].shape[2:], out_with_cache[2][0].shape[2:])
        self.assertEqual(len(result[2][1].shape), len(out_with_cache[2][1].shape))
        self.assertEqual(result[2][1].shape[0], out_with_cache[2][1].shape[0])
        self.assertEqual(result[2][1].shape[2:], out_with_cache[2][1].shape[2:])


    @parameterized.parameters([("return_attention_scores_true", True), ("return_attention_scores_false", False)])
    def test_symbolic_call_structure(self, return_scores):
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 2
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        gqa = layers.GroupedQueryAttention(num_query_heads=num_q_heads, num_key_value_heads=num_kv_heads, head_dim=head_dim)

        shape_q = (2, 5, q_dim)
        shape_kv = (2, 5, kv_dim)
        q_input = layers.Input(batch_shape=shape_q)
        v_input = layers.Input(batch_shape=shape_kv) # Value/Key have different dim for GQA

        cache_k_shape = (shape_q[0], None, gqa.num_key_value_heads, gqa.head_dim)
        cache_v_shape = (shape_q[0], None, gqa.num_key_value_heads, gqa.head_dim)
        cache_k_sym = backend.KerasTensor(cache_k_shape, dtype=q_input.dtype)
        cache_v_sym = backend.KerasTensor(cache_v_shape, dtype=v_input.dtype)
        symbolic_cache_arg = (cache_k_sym, cache_v_sym)

        result = gqa(q_input, v_input, cache=symbolic_cache_arg, return_attention_scores=return_scores)

        if return_scores:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            self.assertIsInstance(result[0], backend.KerasTensor)
            self.assertIsInstance(result[1], backend.KerasTensor)
            self.assertIsInstance(result[2], tuple)
            self.assertEqual(len(result[2]), 2)
            self.assertIsInstance(result[2][0], backend.KerasTensor)
            self.assertIsInstance(result[2][1], backend.KerasTensor)
        else:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], backend.KerasTensor)
            self.assertIsInstance(result[1], tuple)
            self.assertEqual(len(result[1]), 2)
            self.assertIsInstance(result[1][0], backend.KerasTensor)
            self.assertIsInstance(result[1][1], backend.KerasTensor)
