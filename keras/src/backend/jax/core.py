import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from flax.nnx import tracers # This import is fine and used by nnx.Variable

# NO MORE IMPORTS FROM flax.experimental.nnx.* needed for mutable_array or config

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


class JaxVariableImpl(KerasVariable):
    def __init__(self, *args, layout=None, **kwargs):
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _initialize(self, value):
        self._shape = self._validate_shape(value.shape)
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            tensor_layout = distribution.get_variable_layout(self)
            from keras.src.distribution import TensorLayout # Keras internal import
            if isinstance(tensor_layout, TensorLayout):
                self._layout = tensor_layout.backend_layout
            else:
                self._layout = tensor_layout
        self._direct_assign(value) # Calls Variable._direct_assign

    def _direct_assign(self, value): # Keras JAX backend's base assignment
        if self._layout is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value # Sets Keras's internal JAX array

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    def __jax_array__(self):
        return self.value


class Variable(JaxVariableImpl, nnx.Variable):
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
        layout=None,
        nnx_mutable=None, # User's hint for NNX mutability
        **nnx_specific_metadata, # Metadata for nnx.Variable
    ):
        # 1. Pre-initialize parts of nnx.Variable that must exist before Keras's init
        #    (which might trigger attribute sets via MRO).
        object.__setattr__(self, '_trace_state', tracers.TraceState())
        
        _effective_nnx_metadata = nnx_specific_metadata.copy()
        var_t = type(self)
        # Populate default hooks from type(self) if not provided by user
        for hook_name in ['on_get_value', 'on_set_value', 'on_create_value', 'on_add_axis', 'on_remove_axis']:
            if hasattr(var_t, hook_name) and hook_name not in _effective_nnx_metadata:
                _effective_nnx_metadata[hook_name] = getattr(var_t, hook_name)
        object.__setattr__(self, '_var_metadata', _effective_nnx_metadata)

        # Store NNX-specific arguments for when we complete its initialization
        self._nnx_mutable_arg = nnx_mutable
        self._nnx_original_metadata_arg = nnx_specific_metadata.copy() 
        self._nnx_init_pending = True # Flag: full nnx.Variable.__init__ hasn't run yet

        # 2. Call Keras initialization chain (JaxVariableImpl -> KerasVariable)
        super().__init__( # This calls JaxVariableImpl.__init__
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            aggregation=aggregation,
            synchronization=synchronization,
            name=name,
            layout=layout,
        )
        # After this, Keras's self._value is set if not deferred.
        # Keras's self._initializer is set if deferred.

        # 3. If Keras value is ready, complete NNX initialization
        if self._initializer is None: # Keras's way to check if not deferred
            self._complete_nnx_init()

    def _complete_nnx_init(self):
        """
        Completes the nnx.Variable part of initialization by calling its __init__.
        This is done once Keras's self._value is definitely available.
        """
        if not self._nnx_init_pending:
            return
        if self._value is None: # Should only happen if Keras init failed unexpectedly
            raise ValueError(
                "Cannot complete NNX initialization: Keras self._value is None, "
                "but Keras initialization was not deferred."
            )

        current_nnx_mutable = self._nnx_mutable_arg
        if current_nnx_mutable is None: # Default NNX mutability to Keras trainability
            current_nnx_mutable = self.trainable

        # Call nnx.Variable.__init__ to set up its internal state, including self.raw_value.
        # nnx.Variable.__init__ will use its *own internal* logic for mutable_array
        # handling based on the `mutable` flag and its internal config.
        nnx.Variable.__init__(
            self,
            value=self._value, # Pass Keras's JAX array as the initial value
            mutable=current_nnx_mutable,
            **self._nnx_original_metadata_arg # Pass original metadata
        )
        # Now, nnx.Variable has created self.raw_value and run its 'on_create_value' hooks.
        self._nnx_init_pending = False

    def _deferred_initialize(self):
        """Called by Keras for deferred variable initialization."""
        super()._deferred_initialize() # Runs JaxVariableImpl._initialize -> sets self._value
        self._complete_nnx_init()      # Now self._value is ready, complete NNX part

    def _direct_assign(self, value_to_assign):
        """
        Keras's low-level assignment method.
        Called by Keras's self.assign() and during initialization.
        """
        # 1. Let Keras backend (JaxVariableImpl) set self._value
        super()._direct_assign(value_to_assign) # Sets self._value

        # 2. Sync self.raw_value (NNX) from the new self._value (Keras)
        if not self._nnx_init_pending:
            # Invoke nnx.Variable's original value property setter.
            # This ensures NNX's internal logic (hooks, mutability config) is used.
            if hasattr(nnx.Variable, 'value') and \
               isinstance(nnx.Variable.value, property) and \
               nnx.Variable.value.fset is not None:
                nnx.Variable.value.fset(self, self._value)
            else:
                # This is a fallback or error if nnx.Variable's structure changed.
                # It's crucial that nnx.Variable.value.fset is accessible and works.
                raise RuntimeError(
                    "Cannot access nnx.Variable.value.fset for synchronization. "
                    "NNX Variable API might have changed."
                )
    
    # KerasVariable's `value` property is inherited via JaxVariableImpl.
    # Its setter calls `self.assign()`, which calls `self._direct_assign()`.
    # This is the desired loop for external `var.value = x` assignments.
    # The `_direct_assign` sync to `raw_value` uses `nnx.Variable.value.fset` to avoid recursion.

    def copy_from(self, other: nnx.Variable):
        if not isinstance(other, nnx.Variable): # Basic NNX check
            raise TypeError(f"Expected nnx.Variable, got {type(other).__name__}")

        # 1. Let nnx.Variable copy its parts (raw_value, _var_metadata)
        nnx.Variable.copy_from(self, other) # This updates our self.raw_value

        # 2. Sync Keras's self._value from the new self.raw_value.
        #    self.assign() will handle Keras-side checks and call _direct_assign,
        #    which will then re-sync raw_value using nnx.Variable.value.fset.
        #    To get the actual JAX array from raw_value (if it's a mutable_array):
        current_raw_value = self.raw_value # Copied by nnx.Variable.copy_from
        
        # We need to check if current_raw_value is an NNX mutable_array to unwrap it.
        # Since we can't import is_mutable_array, we rely on nnx.Variable.value.fget
        # to give us the "processed" value, which should be the JAX array.
        # However, nnx.Variable.value.fget might also have hooks.
        # A more direct way if raw_value holds the JAX array or a known wrapper:
        # If nnx.Variable made raw_value a mutable_array, it has __jax_array__ or [...].
        # For simplicity, assume raw_value is either the JAX array or can be converted.
        # This part is tricky without is_mutable_array.
        # Let's assume nnx.Variable.copy_from sets raw_value to something assignable.
        # The nnx.Variable.value getter will give the JAX array.
        value_from_nnx_raw = nnx.Variable.value.fget(self) if \
            hasattr(nnx.Variable, 'value') and isinstance(nnx.Variable.value, property) and nnx.Variable.value.fget else self.raw_value

        self.assign(value_from_nnx_raw) # Syncs Keras self._value, then re-syncs raw_value via _direct_assign

        # 3. Sync Keras-specific attributes if `other` is also our combined type
        if isinstance(other, Variable):
            self.trainable = other.trainable # Keras property
            self._autocast = other._autocast
            self._aggregation = other._aggregation
            self._synchronization = other._synchronization
            if hasattr(other, "_layout"):
                self._layout = other._layout

    def update_from_state(self, variable_state: nnx.graph.VariableState): # type: ignore
        # 1. Let nnx.Variable update from its state (sets self.raw_value, self._var_metadata)
        nnx.Variable.update_from_state(self, variable_state)

        # 2. Sync Keras's self._value from the new self.raw_value
        value_from_nnx_raw = nnx.Variable.value.fget(self) if \
            hasattr(nnx.Variable, 'value') and isinstance(nnx.Variable.value, property) and nnx.Variable.value.fget else self.raw_value
            
        self.assign(value_from_nnx_raw) # Syncs Keras self._value, then re-syncs raw_value

        # 3. Sync Keras attributes if they were part of variable_state.metadata
        metadata = variable_state._var_metadata # type: ignore
        if "trainable" in metadata:
            self.trainable = metadata["trainable"]
        if "autocast" in metadata:
            self._autocast = metadata["autocast"]
        # ... other Keras attributes ...

    def __getstate__(self):
        if self._nnx_init_pending and self._initializer is None and self._value is not None:
            self._complete_nnx_init() # Ensure raw_value is set if possible
            
        keras_attrs_to_save = [
            "_name", "_path", "_trainable", "_dtype", "_shape", 
            "_autocast", "_aggregation", "_synchronization",
            "_regularizer", "_constraint", "_layout", "_value", 
            "_initializer", "_nnx_mutable_arg", 
            "_nnx_original_metadata_arg", "_nnx_init_pending"
        ]
        keras_state = {attr: getattr(self, attr) for attr in keras_attrs_to_save if hasattr(self, attr)}
        
        # Get NNX state using its own __getstate__ if possible and robust
        # nnx.Variable.__getstate__ returns {'raw_value': ..., '_trace_state': ..., '_var_metadata': ...}
        # Ensure raw_value is accessed safely.
        nnx_state = nnx.Variable.__getstate__(self) if hasattr(nnx.Variable, '__getstate__') else {
            'raw_value': object.__getattribute__(self, 'raw_value') if hasattr(self, 'raw_value') else None,
            '_trace_state': object.__getattribute__(self, '_trace_state'),
            '_var_metadata': object.__getattribute__(self, '_var_metadata'),
        }
        return {"keras_state": keras_state, "nnx_state": nnx_state}

    def __setstate__(self, state):
        keras_state = state["keras_state"]
        nnx_state = state["nnx_state"]

        # Restore Keras attributes first
        for k, v in keras_state.items():
            object.__setattr__(self, k, v)

        # Restore NNX attributes using its __setstate__
        # This will set self.raw_value, self._trace_state, self._var_metadata
        if hasattr(nnx.Variable, '__setstate__'):
            nnx.Variable.__setstate__(self, nnx_state)
        else: # Manual restore if __setstate__ isn't found (unlikely for nnx.Variable)
            object.__setattr__(self, '_trace_state', nnx_state['_trace_state'])
            object.__setattr__(self, '_var_metadata', nnx_state['_var_metadata'])
            object.__setattr__(self, 'raw_value', nnx_state.get('raw_value'))

        # Post-restore synchronization and checks:
        if self._initializer is not None and self._value is None:
            # Was Keras-deferred pre-pickle. Keras will handle re-initialization if used.
            # If self.raw_value exists from nnx_state, _complete_nnx_init might be
            # callable if self._value can be derived from raw_value first.
            # For now, assume Keras deferred init takes precedence if self._initializer is present.
            pass 
        
        if self._value is not None: # Keras _value restored from keras_state is primary
            if self._nnx_init_pending:
                # Keras part is initialized, NNX part was not fully (e.g. pickled mid-init)
                self._complete_nnx_init() # This will use self._value to call nnx.Variable.__init__
            else:
                # Both Keras and NNX parts were initialized. Sync raw_value from Keras _value.
                if hasattr(nnx.Variable, 'value') and \
                   isinstance(nnx.Variable.value, property) and \
                   nnx.Variable.value.fset is not None:
                    nnx.Variable.value.fset(self, self._value)
                else:
                    raise RuntimeError("Cannot access nnx.Variable.value.fset for __setstate__ synchronization.")
        elif not self._nnx_init_pending and hasattr(self, 'raw_value') and self.raw_value is not None:
            # Keras _value is None, but NNX raw_value exists (e.g. from older pickle)
            # Keras part should adopt NNX's value.
            # Get the JAX array from raw_value (nnx.Variable.value.fget handles hooks/unwrapping)
            _value_from_nnx = nnx.Variable.value.fget(self) if \
                hasattr(nnx.Variable, 'value') and isinstance(nnx.Variable.value, property) and nnx.Variable.value.fget else self.raw_value
            
            object.__setattr__(self, '_value', _value_from_nnx)
            # Update Keras shape/dtype from this adopted value
            if hasattr(self, '_value') and self._value is not None:
                object.__setattr__(self, '_shape', self._validate_shape(self._value.shape))
                object.__setattr__(self, '_dtype', standardize_dtype(self._value.dtype))


# --- convert_to_tensor and other utility functions ---
def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)

    if isinstance(x, Variable): 
        if dtype is not None and x.dtype != dtype:
            return x.value.astype(dtype)
        return x.value

    if isinstance(x, (jnp.ndarray, jax.Array)) and ( # jax.Array is the more modern JAX type
        dtype is None or x.dtype == dtype
    ):
        return x

    if isinstance(x, jax_sparse.JAXSparse):
        if sparse is not None and not sparse:
            x = x.todense()
        elif dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        else:
            return x

    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        return jnp.asarray(x).astype(dtype)
    return jnp.asarray(x, dtype=dtype)

def is_tensor(x):
    # Note: Intentionally does not include `Variable` instances.
    # jax.Array covers DeviceArray and other JAX array types.
    if isinstance(x, (jnp.ndarray, jax.Array, jax_sparse.JAXSparse)):
        return True
    return False

# ... (rest of the backend file: convert_to_numpy, shape, cast, etc.)
# ... (rest of the file)

def convert_to_numpy(x):
    if isinstance(x, jax_sparse.JAXSparse):
        x = x.todense()
    if is_tensor(x) and x.dtype == "bfloat16":
        return np.array(x, dtype=ml_dtypes.bfloat16)
    return np.array(x)

def shape(x):
    # This will work as long as we disallow
    # dynamic shapes in JAX.
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


# Shape / dtype / sparseness inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():
        built_in_types = (type(None), int, float, str, bool, complex, bytes)

        # First, separate symbolic args from other args
        static_args_idx = []
        static_args = []
        maybe_symbolic_args = []
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for idx, arg in enumerate(args):
            if isinstance(arg, built_in_types):
                static_args_idx.append(idx)
                static_args.append(arg)
            else:
                maybe_symbolic_args.append(arg)
        maybe_symbolic_args = tuple(maybe_symbolic_args)
        for k, v in kwargs.items():
            if isinstance(v, built_in_types):
                static_kwargs[k] = v
            else:
                maybe_symbolic_kwargs[k] = v

        # Second, find out if there are dynamic shapes
        has_none = False
        for x in tree.flatten((maybe_symbolic_args, maybe_symbolic_kwargs)):
            if isinstance(x, KerasTensor) and any(d is None for d in x.shape):
                has_none = True

        def convert_keras_tensor_to_jax(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                jax_tensor = jax.ShapeDtypeStruct(shape, dtype=x.dtype)
                return jax_tensor
            if isinstance(x, dict):
                return {
                    k: convert_keras_tensor_to_jax(v, fill_value=fill_value)
                    for k, v in x.items()
                }
            if isinstance(x, list):
                return [
                    convert_keras_tensor_to_jax(xi, fill_value=fill_value)
                    for xi in x
                ]
            return x

        def wrapped_fn(*args, **kwargs):
            # Turn inputs that are sparse to BCOO tensors
            def to_bcoo_if_sparse(x, maybe_symbolic_x):
                if (
                    isinstance(maybe_symbolic_x, KerasTensor)
                    and maybe_symbolic_x.sparse
                ):
                    return jax_sparse.BCOO.fromdense(x, nse=1)
                return x

            args, kwargs = tree.map_structure(
                to_bcoo_if_sparse,
                (args, kwargs),
                (maybe_symbolic_args, maybe_symbolic_kwargs),
            )

            rec_args = []
            idx_static = 0
            idx_sym = 0
            i = 0
            while idx_static < len(static_args) or idx_sym < len(args):
                if i in static_args_idx:
                    rec_args.append(static_args[idx_static])
                    idx_static += 1
                else:
                    rec_args.append(args[idx_sym])
                    idx_sym += 1

                i += 1
            with StatelessScope():
                return fn(*rec_args, **kwargs, **static_kwargs)

        if has_none:
            ms_args_1, ms_kwargs_1 = tree.map_structure(
                lambda x: convert_keras_tensor_to_jax(x, fill_value=83),
                (maybe_symbolic_args, maybe_symbolic_kwargs),
            )
            _, jax_out_1 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                *ms_args_1, **ms_kwargs_1
            )

            ms_args_2, ms_kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_jax(x, fill_value=89),
                (maybe_symbolic_args, maybe_symbolic_kwargs),
            )
            _, jax_out_2 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                *ms_args_2, **ms_kwargs_2
            )

            def merge_shapes(shape1, shape2):
                return tuple(
                    [d1 if d1 == d2 else None for d1, d2 in zip(shape1, shape2)]
                )

            def convert_jax_specs_to_keras_tensor(x1, x2):
                if isinstance(x1, jax.ShapeDtypeStruct):
                    if not isinstance(x2, jax.ShapeDtypeStruct):
                        raise ValueError("Indeterministic output ordering.")
                    return KerasTensor(
                        merge_shapes(x1.shape, x2.shape), dtype=x1.dtype
                    )
                elif isinstance(x1, jax_sparse.BCOO):
                    if not isinstance(x2, jax_sparse.BCOO):
                        raise ValueError("Indeterministic output ordering.")
                    return KerasTensor(
                        merge_shapes(x1.shape, x2.shape),
                        dtype=x1.dtype,
                        sparse=True,
                    )
                else:
                    return x1

            return tree.map_structure(
                convert_jax_specs_to_keras_tensor, jax_out_1, jax_out_2
            )

        maybe_symbolic_args, maybe_symbolic_kwargs = tree.map_structure(
            convert_keras_tensor_to_jax,
            (maybe_symbolic_args, maybe_symbolic_kwargs),
        )
        _, jax_out = jax.make_jaxpr(wrapped_fn, return_shape=True)(
            *maybe_symbolic_args, **maybe_symbolic_kwargs
        )

        def convert_jax_spec_to_keras_tensor(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                return KerasTensor(x.shape, x.dtype)
            elif isinstance(x, jax_sparse.BCOO):
                return KerasTensor(x.shape, x.dtype, sparse=True)
            return x

        return tree.map_structure(convert_jax_spec_to_keras_tensor, jax_out)


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fun=true_fn, false_fun=false_fn)


def vectorized_map(function, elements):
    return jax.vmap(function)(elements)


def map(f, xs):
    return jax.lax.map(f, xs)


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    return jax.lax.scan(
        f, init=init, xs=xs, length=length, reverse=reverse, unroll=unroll
    )


def associative_scan(f, elems, reverse=False, axis=0):
    return jax.lax.associative_scan(f, elems, reverse, axis)


def scatter(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(values)


def scatter_update(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = jnp.array(indices)
    indices = jnp.transpose(indices)
    inputs = inputs.at[tuple(indices)].set(updates)
    return inputs


def slice(inputs, start_indices, shape):
    return jax.lax.dynamic_slice(inputs, start_indices, shape)


def slice_update(inputs, start_indices, updates):
    return jax.lax.dynamic_update_slice(inputs, updates, start_indices)


def switch(index, branches, *operands):
    return jax.lax.switch(index, branches, *operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    if maximum_iterations is not None:
        current_iter = 0
        loop_vars = loop_vars + (current_iter,)

        # Unpack list/tuple args. The last argument is `current_iter`.
        def _cond(args):
            return cond(*args[:-1]) & (args[-1] < maximum_iterations)

        def _body(args):
            outputs = body(*args[:-1])
            outputs = tuple(outputs) if is_tuple else (outputs,)
            return outputs + (args[-1] + 1,)

    else:

        def _cond(args):
            return cond(*args)

        def _body(args):
            outputs = body(*args)
            return tuple(outputs) if is_tuple else (outputs,)

    outputs = jax.lax.while_loop(_cond, _body, loop_vars)
    if maximum_iterations is not None:
        outputs = outputs[:-1]
    return outputs if is_tuple else outputs[0]


def fori_loop(lower, upper, body_fun, init_val):
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    return jax.lax.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]


def random_seed_dtype():
    # jax random seed uses uint32.
    return "uint32"


def custom_gradient(fun):
    return jax.custom_gradient(fun=fun)


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that defines a custom gradient, which
        recomputes f on the backwards pass of a gradient call.
    """
    return jax.checkpoint(f)


class name_scope(base_name_scope):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._jax_name_scope = jax.named_scope(name)

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[], set_to_default=True
        )
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        self._jax_name_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        if self._pop_on_exit:
            self._jax_name_scope.__exit__(*args, **kwargs)


def device_scope(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", etc.
        device_name = device_name.lower()
        jax_device = distribution_lib._to_backend_device(device_name)
    elif not isinstance(device_name, jax.Device):
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or a `jax.Device` instance. "
            f"Received: device_name='{device_name}'"
        )
    else:
        jax_device = device_name
    return jax.default_device(jax_device)
