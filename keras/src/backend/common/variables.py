import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name


class Variable:
    """Represents a backend-agnostic variable in Keras."""

    def __init__(
        self,
        initializer,
        shape=None,
        dtype=None,
        trainable=True,
        autocast=True,
        aggregation="none",
        synchronization="auto",
        name=None,
        **kwargs,
    ):
        del kwargs
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                f"Argument `name` must be a string and cannot contain character `/`. Received: name={name}"
            )
        if aggregation not in (None, "none", "mean", "sum", "only_first_replica"):
            raise ValueError(f"Invalid value for argument `aggregation`. Received: aggregation={aggregation}")
        if aggregation is None:
            aggregation = "none"
        if synchronization not in (None, "none", "on_read", "on_write", "auto"):
            raise ValueError(f"Invalid value for argument `synchronization`. Received: synchronization={synchronization}")
        if synchronization is None:
            synchronization = "none"

        self._name = name
        parent_path = current_path()
        if parent_path:
            self._path = current_path() + "/" + name
        else:
            self._path = name

        self._trainable = bool(trainable)
        self._autocast = bool(autocast)
        self._aggregation = aggregation
        self._synchronization = synchronization
        self._overwrite_with_gradient = False
        self._regularizer = None
        self._constraint = None

        # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': Received initializer type {type(initializer)}, shape {shape}, dtype {dtype}")

        if isinstance(initializer, str):
            # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': Initializer is string '{initializer}'. Getting callable.")
            from keras.src import initializers as KerasInitializers
            initializer = KerasInitializers.get(initializer)

        # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': After string check, initializer type {type(initializer)}, callable: {callable(initializer)}")

        if callable(initializer):
            if shape is None:
                raise ValueError(
                    "When creating a Variable from a callable initializer, "
                    f"the `shape` argument must be specified. Received: initializer={initializer}, shape={shape}"
                )
            self._value = None
            self._initializer = initializer
            self._shape = self._validate_shape(shape) # _validate_shape ensures no None in variable shape
            self._dtype = standardize_dtype(dtype)
            # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': Initializer IS callable. self._initializer set. self.shape={self._shape}, self.dtype={self._dtype}")

            if in_stateless_scope():
                # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': In stateless scope, registering uninitialized_variable.")
                register_uninitialized_variable(self)
            else:
                # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': NOT in stateless scope, calling _initialize_with_initializer.")
                self._initialize_with_initializer(self._initializer)
        else:
            # Initializer is a concrete value
            # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': Initializer is CONCRETE value.")
            concrete_value = self._convert_to_tensor(initializer, dtype=dtype)

            _resolved_dtype = concrete_value.dtype if dtype is None else dtype # type: ignore
            self._dtype = standardize_dtype(_resolved_dtype)

            self._value = concrete_value
            self._initializer = None
            self._shape = self._validate_shape(self._value.shape) # _validate_shape ensures no None
            # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': self._initializer is None. self.shape={self._shape}, self.dtype={self._dtype}")

            if in_stateless_scope():
                raise ValueError("Cannot create Variable from concrete value in stateless scope.")

            self._initialize(self._value)

        self._ndim = len(self._shape)
        # print(f"[KERAS_VAR_DEBUG] KerasVariable.__init__ for '{self._path}': COMPLETED. Final self._initializer type {type(self._initializer)}")

    def _deferred_initialize(self):
        if self._value is not None:
            if config.is_nnx_enabled():
                if self._initializer is not None :
                    # print(f"[KERAS_VAR_DEBUG] KerasVariable._deferred_initialize for '{self.path}': Already initialized, clearing callable initializer.")
                    self._initializer = None
                return
            raise ValueError(f"Variable {self.path} is already initialized.")

        if in_stateless_scope():
            raise ValueError("You are attempting to initialize a variable while in a stateless scope.")
        if self._initializer is None:
            raise ValueError(f"Variable {self.path} has no initializer to defer.")

        # print(f"[KERAS_VAR_DEBUG] KerasVariable._deferred_initialize for '{self.path}': Calling _initialize_with_initializer.")
        self._initialize_with_initializer(self._initializer)
        self._initializer = None
        # print(f"[KERAS_VAR_DEBUG] KerasVariable._deferred_initialize for '{self.path}': COMPLETED. self._initializer is now {type(self._initializer)}.")

    def _validate_shape(self, shape):
        # This is for actual Variable shapes, must be fully defined.
        shape = standardize_shape(shape) # Standardize first (e.g. list to tuple)
        if any(e is None for e in shape): # Check for None after standardization
            raise ValueError(
                "Shapes used to initialize variables must be "
                f"fully-defined (no `None` dimensions). Received: shape={shape} for variable path='{self.path}'"
            )
        return shape

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if self._autocast and autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self)

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def synchronization(self):
        return self._synchronization

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            if self._initializer is not None:
                # print(f"[KERAS_VAR_DEBUG] KerasVariable.value for '{self.path}': _value is None, using _initializer for placeholder.")
                return self._maybe_autocast(
                    self._initializer(self._shape, dtype=self._dtype)
                )
            else:
                raise ValueError(f"Variable {self.path} has not been initialized and has no initializer callable.")
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self.dtype)
        if not shape_equal(value.shape, self.shape):
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                f"`variable.assign(value)` must match. variable.shape={self.shape}, "
                f"Received: value.shape={value.shape}. Target variable: {self}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)
        return value

    def assign_add(self, value):
        return self.assign(self.value + value)

    def assign_sub(self, value):
        return self.assign(self.value - value)

    @property
    def dtype(self):
        autocast_scope = get_autocast_scope()
        if (
            self._autocast
            and autocast_scope is not None
            and is_float_dtype(self._dtype)
        ):
            dtype = autocast_scope.dtype
        else:
            dtype = self._dtype
        return standardize_dtype(dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = bool(value)

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def overwrite_with_gradient(self):
        return self._overwrite_with_gradient

    @overwrite_with_gradient.setter
    def overwrite_with_gradient(self, value):
        if not isinstance(value, bool):
            raise TypeError("`overwrite_with_gradient` must be a boolean.")
        self._overwrite_with_gradient = value

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value):
        from keras.src.regularizers import Regularizer
        if value is not None and not isinstance(value, Regularizer):
            raise ValueError("Invalid regularizer")
        self._regularizer = value

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        from keras.src.constraints import Constraint
        if value is not None and not isinstance(value, Constraint):
            raise ValueError("Invalid constraint")
        self._constraint = value

    def __repr__(self):
        val_for_repr = None
        try:
            if hasattr(self, '_shape') and self._shape is not None and \
               hasattr(self, '_dtype') and self._dtype is not None:
                if self._value is not None:
                    val_for_repr = backend.core.convert_to_numpy(self._value)
        except:
            pass
        path_str = self._path if hasattr(self, '_path') else self._name if hasattr(self, '_name') else 'Unknown'
        shape_str = self._shape if hasattr(self, '_shape') and self._shape is not None else 'Unknown'
        dtype_str = self._dtype if hasattr(self, '_dtype') and self._dtype is not None else 'Unknown'
        value_str = f", value={val_for_repr}" if val_for_repr is not None else ""
        return (
            f"<Variable path={path_str}, shape={shape_str}, "
            f"dtype={dtype_str}{value_str}>"
        )

    def _initialize(self, value):
        raise NotImplementedError("Subclasses must implement _initialize.")

    def _initialize_with_initializer(self, initializer):
        # print(f"[KERAS_VAR_DEBUG] KerasVariable._initialize_with_initializer for '{self.path}' using initializer: {type(initializer)}")
        value = self._convert_to_tensor(
            initializer(self.shape, dtype=self.dtype)
        )
        self._initialize(value)

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError("Subclasses must implement _convert_to_tensor.")

    def _direct_assign(self, value):
        raise NotImplementedError("Subclasses must implement _direct_assign.")

    def __getitem__(self, idx): return self.value[idx]
    def __int__(self):
        if self.ndim > 0: raise TypeError("Only scalar arrays can be converted.")
        return int(self.value)
    def __float__(self):
        if self.ndim > 0: raise TypeError("Only scalar arrays can be converted.")
        return float(self.value)
    def __array__(self, dtype=None): return np.asarray(self.value.__array__(dtype))
    def __bool__(self): raise TypeError("A Keras Variable cannot be used as a boolean.")
    def __neg__(self): return backend.numpy.negative(self.value)
    def __pos__(self): return self.value
    def __abs__(self): return backend.numpy.absolute(self.value)
    def __invert__(self): return backend.numpy.invert(self.value) # type: ignore
    def __eq__(self, other): return backend.numpy.equal(self.value, other)
    def __ne__(self, other): return backend.numpy.not_equal(self.value, other)
    def __lt__(self, other): return backend.numpy.less(self.value, other)
    def __le__(self, other): return backend.numpy.less_equal(self.value, other)
    def __gt__(self, other): return backend.numpy.greater(self.value, other)
    def __ge__(self, other): return backend.numpy.greater_equal(self.value, other)
    def __add__(self, other): return backend.numpy.add(self.value, other)
    def __radd__(self, other): return backend.numpy.add(other, self.value)
    def __sub__(self, other): return backend.numpy.subtract(self.value, other)
    def __rsub__(self, other): return backend.numpy.subtract(other, self.value)
    def __mul__(self, other): return backend.numpy.multiply(self.value, other)
    def __rmul__(self, other): return backend.numpy.multiply(other, self.value)
    def __truediv__(self, other): return backend.numpy.true_divide(self.value, other)
    def __rtruediv__(self, other): return backend.numpy.true_divide(other, self.value)
    def __floordiv__(self, other): return backend.numpy.floor_divide(self.value, other)
    def __rfloordiv__(self, other): return backend.numpy.floor_divide(other, self.value)
    def __mod__(self, other): return backend.numpy.mod(self.value, other)
    def __rmod__(self, other): return backend.numpy.mod(other, self.value)
    def __pow__(self, other): return backend.numpy.power(self.value, other)
    def __rpow__(self, other): return backend.numpy.power(other, self.value)
    def __matmul__(self, other): return backend.numpy.matmul(self.value, other)
    def __rmatmul__(self, other): return backend.numpy.matmul(other, self.value)
    def __and__(self, other): return backend.numpy.logical_and(self.value, other)
    def __rand__(self, other): return backend.numpy.logical_and(other, self.value)
    def __or__(self, other): return backend.numpy.logical_or(self.value, other)
    def __ror__(self, other): return backend.numpy.logical_or(other, self.value)
    def __xor__(self, other): return backend.numpy.logical_xor(self.value, other)
    def __rxor__(self, other): return backend.numpy.logical_xor(other, self.value)
    def __round__(self, ndigits=None): return backend.numpy.round(self.value, decimals=(ndigits or 0))

def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute(
        "uninitialized_variables", [], set_to_default=True
    )
    uninitialized_variables.append(variable)

def initialize_all_variables():
    collection = global_state.get_global_attribute("uninitialized_variables")
    if collection:
        for v in collection:
            v._deferred_initialize()
    global_state.set_global_attribute("uninitialized_variables", [])

@keras_export(
    ["keras.utils.standardize_dtype", "keras.backend.standardize_dtype"]
)
def standardize_dtype(dtype):
    if dtype is None:
        return config.floatx()
    dtype_str = dtypes.PYTHON_DTYPES_MAP.get(dtype, dtype)
    if hasattr(dtype_str, "name"):
        dtype_str = dtype_str.name
    elif hasattr(dtype_str, "__name__"):
        dtype_str = dtype_str.__name__
    elif hasattr(dtype_str, "__str__") and (
        "torch" in str(dtype_str) or "jax.numpy" in str(dtype_str) or "jaxlib" in str(dtype_str)
    ):
        dtype_str = str(dtype_str).split(".")[-1]
    dtype_str = str(dtype_str).lower()
    if dtype_str not in dtypes.ALLOWED_DTYPES:
        for allowed_dtype in dtypes.ALLOWED_DTYPES:
            if allowed_dtype in dtype_str:
                dtype_str = allowed_dtype
                break
        else:
            raise ValueError(f"Invalid dtype: {dtype} (standardized to {dtype_str})")
    return dtype_str

def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
             raise ValueError("Shape cannot be None for standardize_shape.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        if config.backend() == "tensorflow":
            if tf is not None and isinstance(shape, tf.TensorShape):
                shape = shape.as_list()
        shape = tuple(shape)

    for e in shape:
        if e is None: # Allowed for symbolic KerasTensor shapes, InputLayer will pass (None, dim)
            continue
        if config.backend() == "jax" and "_DimExpr" in str(type(e)):
            continue
        if not is_int_dtype(type(e)):
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. Found invalid entry '{e}' of type '{type(e)}'. "
            )
        if e < 0:
            raise ValueError("Negative dimensions are not allowed in shapes.")
    return shape

def shape_equal(a_shape, b_shape):
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        # For variable shapes, None should not appear due to _validate_shape.
        # If comparing symbolic shapes, None might mean "any size".
        # For variable assignment, shapes must be concrete and equal.
        if e1 is not None and e2 is not None and e1 != e2:
            return False
        if (e1 is None and e2 is not None) or (e1 is not None and e2 is None): # Mismatch if one is None and other isn't
            return False
    return True

@keras_export("keras.backend.is_float_dtype")
def is_float_dtype(dtype):
    s_dtype = standardize_dtype(dtype)
    return s_dtype.startswith("float") or s_dtype.startswith("bfloat")

@keras_export("keras.backend.is_int_dtype")
def is_int_dtype(dtype):
    if isinstance(dtype, type):
        if issubclass(dtype, (int, np.integer)): # type: ignore
            return True
    elif isinstance(dtype, (int, np.integer)):
         return True
    s_dtype = standardize_dtype(dtype)
    return s_dtype.startswith("int") or s_dtype.startswith("uint")

def get_autocast_scope():
    return global_state.get_global_attribute("autocast_scope")

class AutocastScope:
    def __init__(self, dtype):
        if dtype is not None:
            s_dtype = standardize_dtype(dtype)
            if not is_float_dtype(s_dtype):
                raise ValueError("`AutocastScope` can only be used with a floating-point target dtype.")
            self.dtype = s_dtype
        else:
            self.dtype = None
        self.original_scope = None
    def maybe_cast(self, value):
        from keras.src import backend as K
        if self.dtype is not None and hasattr(value, 'dtype') and is_float_dtype(value.dtype):
            return K.cast(value, dtype=self.dtype)
        return value
    def __enter__(self):
        self.original_scope = get_autocast_scope()
        global_state.set_global_attribute("autocast_scope", self)
    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("autocast_scope", self.original_scope)

KerasVariable = Variable
```
