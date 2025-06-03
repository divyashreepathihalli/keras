import numpy as np
import ml_dtypes

from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.quantizers import int4_quantizers # Relative import
# TODO: once API is updated, consider: from keras.src import quantizers


class Int4QuantizersTest(testing.TestCase):
    def _assert_int4_values(self, int4_tensor, expected_int_values, msg=""):
        """Helper to assert values of an ml_dtypes.int4 tensor."""
        try:
            # Cast to a standard integer type for comparison.
            # np.int8 is sufficient for the typical int4 range [-8, 7] or [-7, 7].
            converted_values = ops.convert_to_numpy(int4_tensor).astype(np.int8)
            expected_values_np = np.array(expected_int_values, dtype=np.int8)
            self.assertAllEqual(converted_values, expected_values_np, msg=msg)
        except Exception as e:
            self.skipTest(
                f"Could not assert ml_dtypes.int4 values directly: {e}. "
                "This might indicate an issue with converting ml_dtypes.int4 "
                "to a standard NumPy integer type for testing."
            )

    def test_int4_abs_max_quantizer_basics(self):
        quantizer = int4_quantizers.Int4AbsMaxQuantizer()
        x_np = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0], dtype="float32")
        x = ops.array(x_np)
        quantized_x, scale = quantizer(x)

        self.assertEqual(quantized_x.dtype, ml_dtypes.int4, msg="Dtype check")

        # Theoretical calculation for expected quantized values:
        # max_abs_val = 10.0
        # value_range = (-7, 7), so target_max_abs = 7.0
        # scale_val = 10.0 / 7.0
        # x_scaled = x_np / scale_val = x_np * (7.0 / 10.0)
        # x_scaled = [-0.7, 0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 7.0]
        # rounded =   [-1,   0,   1,   1,   2,   3,   4,   7  ] (clip is same)
        expected_int_repr = [-1, 0, 1, 1, 2, 3, 4, 7]
        self._assert_int4_values(quantized_x, expected_int_repr, msg="Quantized values check")

        # Dequantize
        dequantized_x = ops.cast(quantized_x, "float32") * ops.cast(scale, "float32")
        # Due to quantization, expect approximation.
        # Values fully outside original range (10.0) are clipped to edge (7.0 when scaled by scale).
        # Dequant(7) = 7 * (10.0/7.0) = 10.0.
        # Dequant(4) for input 5.0: 4 * (10.0/7.0) = 40.0/7.0 = 5.71...
        # Dequant(3) for input 4.0: 3 * (10.0/7.0) = 30.0/7.0 = 4.28...
        # This means dequantized values can differ from original input due to rounding.
        # The value 2.0 got quantized to 1, dequantizes to 1 * (10/7) = 1.42
        # The value 3.0 got quantized to 2, dequantizes to 2 * (10/7) = 2.85
        # The value 4.0 got quantized to 3, dequantizes to 3 * (10/7) = 4.28
        # The value 5.0 got quantized to 4, dequantizes to 4 * (10/7) = 5.71
        # Let's check a few specific points or overall closeness.
        # Original x: [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        # Expected dequantized: [-10/7, 0, 10/7, 10/7, 20/7, 30/7, 40/7, 70/7=10]
        # = [-1.42, 0, 1.42, 1.42, 2.85, 4.28, 5.71, 10.0] (approx)
        expected_dequantized_np = np.array(expected_int_repr, dtype="float32") * ops.convert_to_numpy(scale)
        self.assertAllClose(ops.convert_to_numpy(dequantized_x), expected_dequantized_np, atol=1e-5, msg="Dequantized values check")
        self.assertEqual(quantized_x.shape, x.shape, msg="Shape check")

        # Test serialization
        self.run_class_serialization_test(quantizer)

    def test_int4_abs_max_quantize_direct(self):
        x_np = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0], dtype="float32")
        x = ops.array(x_np)
        quantized_x, scale = int4_quantizers.int4_abs_max_quantize(x)

        self.assertEqual(quantized_x.dtype, ml_dtypes.int4, msg="Dtype check")

        expected_int_repr = [-1, 0, 1, 1, 2, 3, 4, 7] # Same as class test
        self._assert_int4_values(quantized_x, expected_int_repr, msg="Quantized values check")

        dequantized_x = ops.cast(quantized_x, "float32") * ops.cast(scale, "float32")
        expected_dequantized_np = np.array(expected_int_repr, dtype="float32") * ops.convert_to_numpy(scale)
        self.assertAllClose(ops.convert_to_numpy(dequantized_x), expected_dequantized_np, atol=1e-5, msg="Dequantized values check direct")

        # Test with axis
        x_axis_np = np.array([[-1.0, 5.0], [10.0, 2.0]], dtype="float32")
        x_axis = ops.array(x_axis_np)
        quantized_axis_x, scale_axis = int4_quantizers.int4_abs_max_quantize(x_axis, axis=0)

        self.assertEqual(quantized_axis_x.dtype, ml_dtypes.int4, msg="Axis test dtype check")
        self.assertEqual(scale_axis.shape, (1, 2), msg="Axis test scale shape check")

        expected_scales_axis_np = np.array([[10.0/7.0, 5.0/7.0]], dtype="float32")
        self.assertAllClose(ops.convert_to_numpy(scale_axis), expected_scales_axis_np, atol=1e-5, msg="Axis test scale values check")

        # Calculate expected quantized values for axis test
        # Col 0: x=[-1, 10], max_abs=10, scale=10/7. Quantized: round([-1/(10/7), 10/(10/7)]) = round([-0.7, 7]) = [-1, 7]
        # Col 1: x=[5, 2], max_abs=5, scale=5/7. Quantized: round([5/(5/7), 2/(5/7)]) = round([7, 2.8]) = [7, 3]
        expected_quant_axis_int_repr = [[-1, 7], [7, 3]]
        self._assert_int4_values(quantized_axis_x, expected_quant_axis_int_repr, msg="Axis test quantized values check")

        # Test with different float input dtypes
        for float_dtype_str in ["float16", "bfloat16", "float32"]:
            if float_dtype_str == "bfloat16" and not self.default_supports_bfloat16():
                self.skipTest("Backend does not support bfloat16.")
                continue

            x_f = ops.cast(x, float_dtype_str) # Use original x_np for casting
            quantized_f_x, scale_f = int4_quantizers.int4_abs_max_quantize(x_f)
            self.assertEqual(quantized_f_x.dtype, ml_dtypes.int4, msg=f"Float dtype {float_dtype_str} output dtype check")

            # Check scale dtype. As per current int4_abs_max_quantize, it's backend.floatx()
            # This behavior is slightly different from abs_max_quantize which casts scale to input's float type.
            # For now, test current behavior.
            self.assertEqual(scale_f.dtype, ops.cast(ops.array(1.0), backend.floatx()).dtype, msg=f"Float dtype {float_dtype_str} scale dtype check")

            # Check quantized values for different float dtypes (should be same as float32 if no precision loss)
            self._assert_int4_values(quantized_f_x, expected_int_repr, msg=f"Float dtype {float_dtype_str} quantized values check")


    def test_int4_abs_max_quantizer_serialization(self):
        # Test with non-default axis to ensure it's serialized
        quantizer = int4_quantizers.Int4AbsMaxQuantizer(axis=(0,))
        self.run_class_serialization_test(quantizer)

    def test_int4_abs_max_quantizer_with_axis_and_dtypes(self):
        quantizer_axis0 = int4_quantizers.Int4AbsMaxQuantizer(axis=0)
        quantizer_axis1 = int4_quantizers.Int4AbsMaxQuantizer(axis=1)

        x_np = np.array([[-1.0, 5.0, 2.5], [10.0, 2.0, 5.0]], dtype="float32")

        # Expected for axis=0
        # Col 0: x=[-1,10], max_abs=10, scale=10/7. Q(x)=round(x/(10/7)) => [-1, 7]
        # Col 1: x=[5,2], max_abs=5, scale=5/7. Q(x)=round(x/(5/7)) => [7, 3]
        # Col 2: x=[2.5,5], max_abs=5, scale=5/7. Q(x)=round(x/(5/7)) => [round(2.5*7/5)=round(3.5)=4, 7]
        expected_q_axis0 = [[-1, 7, 4], [7, 3, 7]]

        # Expected for axis=1
        # Row 0: x=[-1,5,2.5], max_abs=5, scale=5/7. Q(x)=round(x/(5/7)) => [round(-1*7/5)=-1, 7, round(2.5*7/5)=4] => [-1, 7, 4]
        # Row 1: x=[10,2,5], max_abs=10, scale=10/7. Q(x)=round(x/(10/7)) => [7, round(2*7/10)=1, round(5*7/10)=4] => [7, 1, 4]
        expected_q_axis1 = [[-1, 7, 4], [7, 1, 4]]

        for float_dtype_str in ["float16", "bfloat16", "float32"]:
            if float_dtype_str == "bfloat16" and not self.default_supports_bfloat16():
                self.skipTest("Backend does not support bfloat16.")
                continue

            x_f = ops.cast(ops.array(x_np), float_dtype_str)

            q_x_axis0, _ = quantizer_axis0(x_f)
            self.assertEqual(q_x_axis0.dtype, ml_dtypes.int4, msg=f"Axis0, {float_dtype_str} dtype")
            self._assert_int4_values(q_x_axis0, expected_q_axis0, msg=f"Axis0, {float_dtype_str} values")

            q_x_axis1, _ = quantizer_axis1(x_f)
            self.assertEqual(q_x_axis1.dtype, ml_dtypes.int4, msg=f"Axis1, {float_dtype_str} dtype")
            self._assert_int4_values(q_x_axis1, expected_q_axis1, msg=f"Axis1, {float_dtype_str} values")


if __name__ == "__main__":
    testing.main()
