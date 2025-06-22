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

# --- Standardize Dtype & Shape Utilities --- (Copied from previous correct state)

@keras_export("keras.backend.is_int_dtype")
def is_int_dtype(dtype):
    if isinstance(dtype, type):
        if issubclass(dtype, (int, np.integer)): # type: ignore
            return True
    elif isinstance(dtype, (int, np.integer)):
         return True
    # Fallback to string check for standardized dtypes
    # standardize_dtype might not be defined yet if this is called during its own definition,
    # so handle potential recursion or ensure order. For now, assume it's available or called later.
    try:
        s_dtype = standardize_dtype(dtype)
        return s_dtype.startswith("int") or s_dtype.startswith("uint")
    except ValueError: # If standardize_dtype fails
        return False
    except RecursionError: # If called during standardize_dtype's own definition process
        # Basic string check for integer types if standardize_dtype is not ready
        dtype_str = str(dtype).lower()
        return "int" in dtype_str or "uint" in dtype_str


@keras_export(
    ["keras.utils.standardize_dtype", "keras.backend.standardize_dtype"]
)
def standardize_dtype(dtype):
    if dtype is None:
        return config.floatx()

    dtype_name = dtypes.PYTHON_DTYPES_MAP.get(dtype, None)
    if dtype_name is not None:
        current_dtype_str = dtype_name
    elif hasattr(dtype, "name"):
        current_dtype_str = dtype.name
    elif hasattr(dtype, "__name__"):
        current_dtype_str = dtype.__name__
    elif hasattr(dtype, "__str__"):
        dtype_str_repr = str(dtype).lower()
        if "torch." in dtype_str_repr:
            current_dtype_str = dtype_str_repr.replace("torch.", "")
        elif "jax.numpy." in dtype_str_repr or "jaxlib.xla_extension." in dtype_str_repr :
            current_dtype_str = dtype_str_repr.split(".")[-1]
            if current_dtype_str == "bool_": current_dtype_str = "bool"
        else:
            current_dtype_str = dtype_str_repr # Fallback to string representation
    else: # Should not happen for valid dtype inputs
        current_dtype_str = str(dtype)

    current_dtype_str = str(current_dtype_str).lower()

    if current_dtype_str not in dtypes.ALLOWED_DTYPES:
        for allowed_dtype_val in dtypes.ALLOWED_DTYPES:
            if allowed_dtype_val in current_dtype_str:
                current_dtype_str = allowed_dtype_val
                break
        else:
            raise ValueError(f"Invalid dtype: {dtype} (standardized to {current_dtype_str})")
    return current_dtype_str


@keras_export("keras.backend.is_float_dtype")
def is_float_dtype(dtype):
    s_dtype = standardize_dtype(dtype)
    return s_dtype.startswith("float") or s_dtype.startswith("bfloat")


def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Shape argument to standardize_shape should be an iterable, not None itself.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape, not iterable.")

        if config.backend() == "tensorflow" and tf is not None and isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        shape = tuple(shape)

    processed_shape = []
    for e in shape:
        if e is None:
            processed_shape.append(None)
            continue
        if config.backend() == "jax" and hasattr(e, '__class__') and "_DimExpr" in str(e.__class__):
            processed_shape.append(e)
            continue

        try:
            e_int = int(e)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot convert shape '{shape}' to a valid shape. "
                f"Found non-integer entry '{e}' of type '{type(e)}'. "
            )
        if e_int < 0:
            raise ValueError(f"Negative dimensions are not allowed in shapes. Got: {shape}")
        processed_shape.append(e_int)
    return tuple(processed_shape)

def shape_equal(a_shape, b_shape):
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is None or e2 is None:
            if e1 is not e2:
                return False
            continue
        if e1 != e2:
            return False
    return True

# --- Variable Class ---
class Variable:
    """Represents a backend-agnostic variable in Keras.
    (Docstring omitted for brevity but is part of the actual file)
    """
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
        if aggregation is None: aggregation = "none"
        if synchronization not in (None, "none", "on_read", "on_write", "auto"):
            raise ValueError(f"Invalid value for argument `synchronization`. Received: synchronization={synchronization}")
        if synchronization is None: synchronization = "none"

        self._name = name
        parent_path = current_path()
        if parent_path: self._path = current_path() + "/" + name
        else: self._path = name

        self._trainable = bool(trainable)
        self._autocast = bool(autocast)
        self._aggregation = aggregation
        self._synchronization = synchronization
        self._overwrite_with_gradient = False
        self._regularizer = None
        self._constraint = None

        if isinstance(initializer, str):
            from keras.src import initializers as KerasInitializers
            initializer = KerasInitializers.get(initializer)

        self._dtype = standardize_dtype(dtype)

        if callable(initializer):
            if shape is None:
                raise ValueError(
                    "When creating a Variable from a callable initializer, "
                    f"the `shape` argument must be specified. Received: initializer={initializer}, shape={shape}"
                )
            self._value = None
            self._initializer = initializer
            self._shape = self._validate_shape(shape)
            if in_stateless_scope():
                register_uninitialized_variable(self)
            else:
                self._initialize_with_initializer(self._initializer)
        else: # Initializer is a concrete value
            concrete_value = self._convert_to_tensor(initializer, dtype=self._dtype)
            if dtype is None:
                self._dtype = standardize_dtype(concrete_value.dtype) # type: ignore

            self._value = concrete_value
            self._initializer = None
            self._shape = self._validate_shape(self._value.shape)
            if in_stateless_scope():
                raise ValueError("Cannot create Variable from concrete value in stateless scope.")
            self._initialize(self._value)

        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            if config.is_nnx_enabled():
                if self._initializer is not None :
                    self._initializer = None
                return
            raise ValueError(f"Variable {self.path} is already initialized.")
        if in_stateless_scope():
            raise ValueError("Cannot initialize variable in stateless scope.")
        if self._initializer is None:
            raise ValueError(f"Variable {self.path} has no initializer to defer.")
        self._initialize_with_initializer(self._initializer)
        self._initializer = None

    def _validate_shape(self, shape):
        shape = standardize_shape(shape)
        if any(e is None for e in shape):
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

    def numpy(self): return np.array(self)
    @property
    def aggregation(self): return self._aggregation
    @property
    def synchronization(self): return self._synchronization
    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None: return self._maybe_autocast(value)
        if self._value is None:
            if self._initializer is not None:
                return self._maybe_autocast(
                    self._initializer(self._shape, dtype=self._dtype)
                )
            raise ValueError(f"Variable {self.path} not initialized and no initializer callable.")
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self.dtype)
        if not shape_equal(value.shape, self.shape):
            raise ValueError(f"Shape mismatch for {self.path}: expected {self.shape}, got {value.shape}.")
        if in_stateless_scope():
            get_stateless_scope().add_update((self, value))
        else:
            self._direct_assign(value)
        return value
    def assign_add(self, value): return self.assign(self.value + value)
    def assign_sub(self, value): return self.assign(self.value - value)
    @property
    def dtype(self): # Note: self._dtype is already standardized in __init__ or by concrete value
        autocast_scope = get_autocast_scope()
        if self._autocast and autocast_scope is not None and is_float_dtype(self._dtype):
            return standardize_dtype(autocast_scope.dtype)
        return self._dtype # Already standardized
    @property
    def shape(self): return self._shape
    @property
    def ndim(self): return self._ndim
    @property
    def trainable(self): return self._trainable
    @trainable.setter
    def trainable(self, value): self._trainable = bool(value)
    @property
    def name(self): return self._name
    @property
    def path(self): return self._path
    @property
    def overwrite_with_gradient(self): return self._overwrite_with_gradient
    @overwrite_with_gradient.setter
    def overwrite_with_gradient(self, value):
        if not isinstance(value, bool): raise TypeError("`overwrite_with_gradient` must be a boolean.")
        self._overwrite_with_gradient = value
    @property
    def regularizer(self): return self._regularizer
    @regularizer.setter
    def regularizer(self, value):
        from keras.src.regularizers import Regularizer
        if value is not None and not isinstance(value, Regularizer): raise ValueError("Invalid regularizer")
        self._regularizer = value
    @property
    def constraint(self): return self._constraint
    @constraint.setter
    def constraint(self, value):
        from keras.src.constraints import Constraint
        if value is not None and not isinstance(value, Constraint): raise ValueError("Invalid constraint")
        self._constraint = value
    def __repr__(self):
        val_for_repr = None
        try:
            # Check if _value is set and usable before trying to convert
            if hasattr(self, '_value') and self._value is not None:
                 # Ensure backend.core.convert_to_numpy exists or use a safe alternative
                if hasattr(backend, 'core') and hasattr(backend.core, 'convert_to_numpy'):
                     val_for_repr = backend.core.convert_to_numpy(self._value)
                else:
                     val_for_repr = np.array(self._value)
        except: pass # Be very defensive for repr
        path_str = getattr(self, '_path', getattr(self, '_name', 'Unknown'))
        shape_str = getattr(self, '_shape', 'Unknown')
        dtype_str = getattr(self, '_dtype', 'Unknown') # Use self._dtype as self.dtype might involve autocast scope
        value_str = f", value={val_for_repr}" if val_for_repr is not None else ""
        return (f"<Variable path={path_str}, shape={shape_str}, dtype={dtype_str}{value_str}>")
    def _initialize(self, value): raise NotImplementedError("Subclasses must implement _initialize.")
    def _initialize_with_initializer(self, initializer):
        value = self._convert_to_tensor(initializer(self.shape, dtype=self.dtype))
        self._initialize(value)
    def _convert_to_tensor(self, value, dtype=None): raise NotImplementedError("Subclasses must implement _convert_to_tensor.")
    def _direct_assign(self, value): raise NotImplementedError("Subclasses must implement _direct_assign.")
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
    def __invert__(self): return backend.numpy.invert(self.value)
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
    uninitialized_variables = global_state.get_global_attribute("uninitialized_variables", [], set_to_default=True)
    uninitialized_variables.append(variable)

def initialize_all_variables():
    collection = global_state.get_global_attribute("uninitialized_variables")
    if collection:
        for v in collection:
            v._deferred_initialize()
    global_state.set_global_attribute("uninitialized_variables", [])

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
