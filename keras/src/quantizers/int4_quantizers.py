import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.quantizers.quantizers import Quantizer


@keras_export("keras.quantizers.int4_abs_max_quantize")
def int4_abs_max_quantize(
    x, axis=None, value_range=(-7, 7), epsilon=1e-7, dtype=ml_dtypes.int4
):
    """Quantizes a tensor to int4 using absolute maximum scaling.

    The function computes the maximum absolute value of the input tensor `x`
    along the specified `axis`. This value is used to determine a `scale` factor
    such that the data, when scaled, can be represented within the `value_range`
    (typically `[-7, 7]` for int4). The scaled values are then rounded to the
    nearest integer and clipped to `value_range`. Finally, the result is cast
    to the specified `dtype`.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to calculate the maximum absolute value.
            If `None`, the maximum is taken over the entire tensor.
        value_range: A tuple `(min_val, max_val)` defining the target range for
            quantized values. Defaults to `(-7, 7)`.
        epsilon: A small float to prevent division by zero if the maximum
            absolute value is very close to zero. Defaults to `1e-7`.
        dtype: The target data type for the quantized output. Defaults to
            `ml_dtypes.int4`.

    Returns:
        A tuple `(quantized_x, scale)`, where `quantized_x` is the quantized
        tensor of type `dtype`, and `scale` is the computed scale factor.
    """
    x_float = ops.cast(x, dtype=backend.floatx())
    max_abs_val = ops.max(ops.abs(x_float), axis=axis, keepdims=True)
    # Calculate scale such that dividing x_float by scale maps max_abs_val to value_range[1]
    scale = max_abs_val / value_range[1]
    scale = ops.maximum(scale, epsilon)

    quantized_x = ops.round(x_float / scale)
    quantized_x = ops.clip(quantized_x, value_range[0], value_range[1])
    quantized_x = ops.cast(quantized_x, dtype=dtype)
    return quantized_x, scale


@keras_export("keras.quantizers.Int4AbsMaxQuantizer")
class Int4AbsMaxQuantizer(Quantizer):
    def __init__(self, axis=None, value_range=(-7, 7), epsilon=1e-7):
        super().__init__(output_dtype=ml_dtypes.int4)
        self.axis = axis
        self.value_range = value_range
        self.epsilon = epsilon

    def __call__(self, x):
        # The int4_abs_max_quantize function now handles casting to output_dtype
        return int4_abs_max_quantize(
            x,
            axis=self.axis,
            value_range=self.value_range,
            epsilon=self.epsilon,
            dtype=self.output_dtype, # Pass the class's output_dtype
        )

    def get_config(self):
        return {
            "axis": self.axis,
            "value_range": self.value_range,
            "epsilon": self.epsilon,
            "output_dtype": self.output_dtype,
        }
