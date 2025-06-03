import inspect

from keras.src.api_export import keras_export
from keras.src.quantizers.int4_quantizers import Int4AbsMaxQuantizer
from keras.src.quantizers.int4_quantizers import int4_abs_max_quantize
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import Quantizer
from keras.src.quantizers.quantizers import abs_max_quantize
from keras.src.quantizers.quantizers import compute_float8_amax_history
from keras.src.quantizers.quantizers import compute_float8_scale
from keras.src.quantizers.quantizers import fake_quant_with_min_max_vars
from keras.src.quantizers.quantizers import quantize_and_dequantize
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

# Public symbols
__all__ = [
    "Quantizer",
    "AbsMaxQuantizer",
    "Int4AbsMaxQuantizer",
    "abs_max_quantize",
    "int4_abs_max_quantize",
    "fake_quant_with_min_max_vars",
    "quantize_and_dequantize",
    "compute_float8_scale",
    "compute_float8_amax_history",
    "serialize",
    "deserialize",
    "get",
]

ALL_OBJECTS = {Quantizer, AbsMaxQuantizer, Int4AbsMaxQuantizer}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
# Make sure snake-cased version of Int4AbsMaxQuantizer is available
# The above update might create "int4_abs_max_quantizer"
# If a specific alias like "int4absmq" is required:
# ALL_OBJECTS_DICT["int4absmq"] = Int4AbsMaxQuantizer


@keras_export("keras.quantizers.serialize")
def serialize(initializer):
    return serialization_lib.serialize_keras_object(initializer)


@keras_export("keras.quantizers.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras quantizer object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.quantizers.get")
def get(identifier, **kwargs):
    """Retrieve a Keras quantizer object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj(kwargs)
        return obj
    else:
        raise ValueError(
            f"Could not interpret quantizer identifier: {identifier}"
        )
