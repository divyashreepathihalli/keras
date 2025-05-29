# flake8: noqa
import numpy as np
import pytest
from absl.testing import parameterized
import tensorflow as tf # Added for tf.errors.InternalError

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src import models # Added for Sequential model
from keras.src import optimizers # Added for Adam optimizer
from keras.src import mixed_precision # Added for mixed precision


class UpSampling2dTest(testing.TestCase):
    @parameterized.product(
        data_format=["channels_first", "channels_last"],
        length_row=[2],
        length_col=[2, 3],
    )
    @pytest.mark.requires_trainable_backend
    def test_upsampling_2d(self, data_format, length_row, length_col):
        num_samples = 2
        stack_size = 2
        input_num_row = 11
        input_num_col = 12

        if data_format == "channels_first":
            inputs = np.random.rand(
                num_samples, stack_size, input_num_row, input_num_col
            )
        else:
            inputs = np.random.rand(
                num_samples, input_num_row, input_num_col, stack_size
            )

        # basic test
        self.run_layer_test(
            layers.UpSampling2D,
            init_kwargs={"size": (2, 2), "data_format": data_format},
            input_shape=inputs.shape,
        )

        layer = layers.UpSampling2D(
            size=(length_row, length_col),
            data_format=data_format,
        )
        layer.build(inputs.shape)
        np_output = layer(inputs=backend.Variable(inputs))
        if data_format == "channels_first":
            assert np_output.shape[2] == length_row * input_num_row
            assert np_output.shape[3] == length_col * input_num_col
        else:
            assert np_output.shape[1] == length_row * input_num_row
            assert np_output.shape[2] == length_col * input_num_col

        # compare with numpy
        if data_format == "channels_first":
            expected_out = np.repeat(inputs, length_row, axis=2)
            expected_out = np.repeat(expected_out, length_col, axis=3)
        else:
            expected_out = np.repeat(inputs, length_row, axis=1)
            expected_out = np.repeat(expected_out, length_col, axis=2)

        self.assertAllClose(np_output, expected_out)

    @parameterized.product(
        data_format=["channels_first", "channels_last"],
        length_row=[2],
        length_col=[2, 3],
    )
    @pytest.mark.requires_trainable_backend
    def test_upsampling_2d_bilinear(self, data_format, length_row, length_col):
        num_samples = 2
        stack_size = 2
        input_num_row = 11
        input_num_col = 12
        if data_format == "channels_first":
            inputs = np.random.rand(
                num_samples, stack_size, input_num_row, input_num_col
            )
        else:
            inputs = np.random.rand(
                num_samples, input_num_row, input_num_col, stack_size
            )

        self.run_layer_test(
            layers.UpSampling2D,
            init_kwargs={
                "size": (2, 2),
                "data_format": data_format,
                "interpolation": "bilinear",
            },
            input_shape=inputs.shape,
        )

        layer = layers.UpSampling2D(
            size=(length_row, length_col),
            data_format=data_format,
        )
        layer.build(inputs.shape)
        np_output = layer(inputs=backend.Variable(inputs))
        if data_format == "channels_first":
            self.assertEqual(np_output.shape[2], length_row * input_num_row)
            self.assertEqual(np_output.shape[3], length_col * input_num_col)
        else:
            self.assertEqual(np_output.shape[1], length_row * input_num_row)
            self.assertEqual(np_output.shape[2], length_col * input_num_col)

    def test_upsampling_2d_correctness(self):
        input_shape = (2, 2, 1, 3)
        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        # fmt: off
        expected_output = np.array(
            [[[[ 0.,  1.,  2.],
               [ 0.,  1.,  2.]],
              [[ 3.,  4.,  5.],
               [ 3.,  4.,  5.]]],
             [[[ 6.,  7.,  8.],
               [ 6.,  7.,  8.]],
              [[ 9., 10., 11.],
               [ 9., 10., 11.]]]]
        )
        # fmt: on
        if backend.config.image_data_format() == "channels_first":
            expected_output = expected_output.transpose((0, 3, 1, 2))
            x = x.transpose((0, 3, 1, 2))
        self.assertAllClose(
            layers.UpSampling2D(size=(1, 2))(x), expected_output
        )

    def test_upsampling_2d_various_interpolation_methods(self):
        input_shape = (2, 2, 1, 3)
        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        for interpolation in ["nearest", "bilinear", "bicubic"]:
            layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(x)

    @pytest.mark.skipif(
        backend.backend() == "torch", reason="Torch does not support lanczos."
    )
    def test_upsampling_2d_lanczos_interpolation_methods(self):
        input_shape = (2, 2, 1, 3)
        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        for interpolation in ["lanczos3", "lanczos5"]:
            layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(x)

    @pytest.mark.requires_trainable_backend
    def test_upsampling2d_mixed_precision_bilinear_jit_correctness(self):
        if backend.backend() != "tensorflow":
            pytest.skip(
                "This test is specific to TensorFlow backend due to XLA and "
                "ResizeBilinearGrad behavior."
            )

        original_policy = mixed_precision.global_policy()
        try:
            mixed_precision.set_global_policy("mixed_float16")

            model = models.Sequential()
            # Input shape needs to be large enough to not trigger all-zero gradients
            # which might mask the issue.
            model.add(layers.Input(shape=[16, 16, 32], dtype="float32"))
            model.add(layers.Conv2D(filters=4, kernel_size=3, padding="same"))
            model.add(
                layers.UpSampling2D(
                    size=2,
                    interpolation="bilinear",
                    name="up_sample_bilinear",
                )
            )
            # Using Conv2D after UpSampling2D as it's a common scenario
            # and its gradient computation will interact with UpSampling2D's gradient.
            model.add(layers.Conv2D(filters=4, kernel_size=3, padding="same", activation="relu"))
            model.add(layers.GlobalAveragePooling2D()) # To reduce output size for Dense
            model.add(layers.Dense(1, activation="sigmoid"))


            model.compile(
                optimizer=optimizers.Adam(),
                loss="binary_crossentropy", # Use binary_crossentropy for sigmoid output
                jit_compile=True,
            )

            # Dummy data
            batch_size = 2
            input_data = np.ones((batch_size, 16, 16, 32), dtype="float32")
            # Target data for sigmoid output should be between 0 and 1
            target_data = np.random.randint(0, 2, size=(batch_size, 1)).astype("float32")
            
            try:
                model.fit(input_data, target_data, epochs=1, steps_per_epoch=1)
            except tf.errors.InternalError as e:
                self.fail(f"model.fit() raised InternalError with JIT and mixed_float16: {e}")
            except Exception as e:
                self.fail(f"model.fit() raised an unexpected exception: {e}")

        finally:
            mixed_precision.set_global_policy(original_policy)
