import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

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

import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from flax.nnx import tracers  # <--- Added import
from flax.nnx import utils as nnx_utils  # <--- Added import

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


# Renamed original JaxVariable to avoid confusion if it's kept separate
# If this is the only JAX variable, Variable can directly inherit KerasVariable
class JaxVariableImpl(KerasVariable):  # Or "_BaseJaxKerasVariable"
    def __init__(self, *args, layout=None, **kwargs):
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _initialize(self, value):
        self._shape = self._validate_shape(value.shape)
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            tensor_layout = distribution.get_variable_layout(self)
            from keras.src.distribution import TensorLayout

            if isinstance(tensor_layout, TensorLayout):
                self._layout = tensor_layout.backend_layout
            else:
                self._layout = tensor_layout
        # Calls _direct_assign of the most derived class (our new Variable)
        self._direct_assign(value)

    def _direct_assign(self, value):
        # This will be called by super()._direct_assign() from the new Variable
        if self._layout is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value # Keras internal JAX array

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    def __jax_array__(self):
        return self.value


# This is now the main Variable class for the JAX backend
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
        layout=None,  # Keras JAX backend specific
        # NNX specific args
        nnx_mutable=None,
        **nnx_specific_metadata, # Changed name for clarity
    ):
        # --- Start of Critical NNX Pre-initialization ---
        # These must be set before Keras's __init__ runs, as Keras's attribute
        # assignments will trigger nnx.Variable.__setattr__.
        # We use object.__setattr__ to bypass nnx.Variable's own __setattr__
        # for these initial setup attributes.
        object.__setattr__(self, '_trace_state', tracers.TraceState())
        
        # nnx.Variable.__init__ copies the passed metadata. We do the same.
        # It also can populate default hooks here from type(self) if not in metadata.
        # For simplicity, we'll just use what's passed first.
        # If you need the default hook registration, it would go here.
        _effective_nnx_metadata = nnx_specific_metadata.copy()
        var_t = type(self) # Variable type

        # Populate default hooks from type(self) if not provided in nnx_specific_metadata
        # This mimics nnx.Variable.__init__'s hook registration behavior.
        if hasattr(var_t, 'on_get_value') and 'on_get_value' not in _effective_nnx_metadata:
            _effective_nnx_metadata['on_get_value'] = var_t.on_get_value
        if hasattr(var_t, 'on_set_value') and 'on_set_value' not in _effective_nnx_metadata:
            _effective_nnx_metadata['on_set_value'] = var_t.on_set_value
        if hasattr(var_t, 'on_create_value') and 'on_create_value' not in _effective_nnx_metadata:
            _effective_nnx_metadata['on_create_value'] = var_t.on_create_value
        if hasattr(var_t, 'on_add_axis') and 'on_add_axis' not in _effective_nnx_metadata:
            _effective_nnx_metadata['on_add_axis'] = var_t.on_add_axis
        if hasattr(var_t, 'on_remove_axis') and 'on_remove_axis' not in _effective_nnx_metadata:
            _effective_nnx_metadata['on_remove_axis'] = var_t.on_remove_axis
            
        object.__setattr__(self, '_var_metadata', _effective_nnx_metadata)
        # --- End of Critical NNX Pre-initialization ---

        # Store NNX args for the completion of NNX initialization later
        self._nnx_mutable_arg = nnx_mutable
        # self._nnx_metadata_arg is not strictly needed if _var_metadata is set,
        # but keeping it if _complete_nnx_init needs original user intent.
        self._nnx_original_metadata_arg = nnx_specific_metadata.copy() 
        self._nnx_init_pending = True # Flag to indicate full NNX setup is pending

        # Call Keras initialization chain (JaxVariableImpl -> KerasVariable)
        # Pass only Keras-relevant arguments.
        super().__init__( 
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            aggregation=aggregation,
            synchronization=synchronization,
            name=name,
            layout=layout,
            # KerasVariable's __init__ has a `**kwargs` that it `del`s.
            # So, nnx_specific args should not be passed here.
        )

        # If Keras initialization was not deferred (i.e., self._value is set),
        # complete the NNX-specific part of the initialization.
        if self._initializer is None: # Keras's way to check if not deferred
            self._complete_nnx_init()

    def _complete_nnx_init(self):
        """Completes the nnx.Variable part of initialization once Keras's _value is ready."""
        if not self._nnx_init_pending:
            return

        if self._value is None: # Should not happen if called correctly
            raise ValueError(
                "Cannot complete NNX initialization: Keras self._value is None, "
                "but Keras initializer is also None (should not be deferred)."
            )

        current_nnx_mutable = self._nnx_mutable_arg
        if current_nnx_mutable is None:
            current_nnx_mutable = self.trainable # Default link to Keras trainable

        _value_for_nnx = self._value # Keras's JAX array

        # Process value for NNX (e.g., wrap in mutable_array if needed)
        # This logic is similar to what nnx.Variable.__init__ does.
        if current_nnx_mutable:
            if not nnx_utils.is_mutable_array(_value_for_nnx):
                _value_for_nnx = nnx_utils.mutable_array(jnp.asarray(_value_for_nnx))
        else:
            # Ensure it's a JAX array, not a mutable_array if not mutable
            if nnx_utils.is_mutable_array(_value_for_nnx):
                _value_for_nnx = _value_for_nnx.__array__() # Unwrap
            _value_for_nnx = jnp.asarray(_value_for_nnx)
        
        object.__setattr__(self, 'raw_value', _value_for_nnx)

        # Run create_value hooks (self.create_value is from nnx.Variable)
        # This uses the 'on_create_value' hook potentially in self._var_metadata
        # The self.raw_value might be replaced by the hook.
        object.__setattr__(self, 'raw_value', self.create_value(self.raw_value))
        
        self._nnx_init_pending = False

    def _deferred_initialize(self):
        # Called by Keras for deferred initialization
        super()._deferred_initialize() # Runs JaxVariableImpl._initialize -> sets self._value
        self._complete_nnx_init() # Now complete NNX part

    def _direct_assign(self, value_to_assign):
        # Called by Keras logic (e.g., _initialize or self.assign)
        # First, let the Keras side (JaxVariableImpl) handle its assignment logic
        super()._direct_assign(value_to_assign) # This sets self._value via JaxVariableImpl

        # After Keras's self._value is updated, sync nnx.Variable's raw_value
        # Only if the full NNX initialization has completed.
        if not self._nnx_init_pending:
            # self.raw_value has already been appropriately typed (mutable or not)
            # by _complete_nnx_init.
            if nnx_utils.is_mutable_array(self.raw_value):
                self.raw_value[...] = self._value # Assign into the mutable array
            else:
                # If raw_value is not a mutable_array (e.g., nnx_mutable was False),
                # then self.raw_value should directly be the JAX array.
                object.__setattr__(self, 'raw_value', self._value)

    @property
    def value(self):
        # MRO ensures KerasVariable.value (via JaxVariableImpl) is called.
        # This handles stateless scope, autocasting, and returns self._value.
        return super().value

    @value.setter
    def value(self, new_value):
        # Use Keras's assign mechanism to ensure all Keras logic (shape/dtype checks,
        # distribution, calling _direct_assign) is respected.
        self.assign(new_value) # self.assign will eventually call our _direct_assign

    def copy_from(self, other: nnx.Variable):
        if not isinstance(other, nnx.Variable):
            raise TypeError(f"Expected nnx.Variable, got {type(other).__name__}")

        # Let nnx.Variable handle its part (raw_value, _var_metadata)
        nnx.Variable.copy_from(self, other) # Call nnx.Variable's method directly

        keras_value_to_assign = self.raw_value
        if nnx_utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__()
        
        self.assign(keras_value_to_assign) # Sync Keras side

        if isinstance(other, Variable): # If other is also our combined type
            self.trainable = other.trainable
            self._autocast = other._autocast
            self._aggregation = other._aggregation
            self._synchronization = other._synchronization # Added
            if hasattr(other, "_layout"):
                self._layout = other._layout
            # Note: _var_metadata was already copied by nnx.Variable.copy_from.
            # If Keras attributes were stored there, they'd be copied too.

    def update_from_state(self, variable_state: nnx.graph.VariableState): # type: ignore
        nnx.Variable.update_from_state(self, variable_state) # NNX part

        keras_value_to_assign = self.raw_value
        if nnx_utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__()

        self.assign(keras_value_to_assign) # Sync Keras side

        # Sync Keras attributes if they were part of variable_state.metadata
        # This depends on how variable_state was created.
        # Make these checks safer with .get() or hasattr
        metadata = variable_state._var_metadata # type: ignore
        if "trainable" in metadata:
            self.trainable = metadata["trainable"]
        if "autocast" in metadata: # Assuming autocast is a boolean Keras attr
            self._autocast = metadata["autocast"]
        # Add other Keras attributes as needed

    def __getstate__(self):
        # Ensure NNX part is fully initialized if it was pending, so raw_value is correct
        if self._nnx_init_pending and self._initializer is None and self._value is not None:
            self._complete_nnx_init()
            
        keras_attrs_to_save = [
            "_name", "_path", "_trainable", "_dtype", "_shape", 
            "_autocast", "_aggregation", "_synchronization",
            "_regularizer", "_constraint", "_layout", "_value", 
            "_initializer", "_nnx_mutable_arg", 
            "_nnx_original_metadata_arg", # Save original nnx metadata passed by user
            "_nnx_init_pending"
        ]
        keras_state = {attr: getattr(self, attr) for attr in keras_attrs_to_save if hasattr(self, attr)}
        
        # Get NNX state, this will include raw_value, _trace_state, _var_metadata
        # Use object.__getattribute__ to be safe if nnx.Variable overrides __getattr__ heavily
        nnx_state = {
            'raw_value': object.__getattribute__(self, 'raw_value') if hasattr(self, 'raw_value') else None,
            '_trace_state': object.__getattribute__(self, '_trace_state'),
            '_var_metadata': object.__getattribute__(self, '_var_metadata'),
        }
        # Or, if nnx.Variable.__getstate__ is well-behaved:
        # nnx_state = nnx.Variable.__getstate__(self)

        return {"keras_state": keras_state, "nnx_state": nnx_state}

    def __setstate__(self, state):
        keras_state = state["keras_state"]
        nnx_state = state["nnx_state"]

        # Restore Keras attributes first
        for k, v in keras_state.items():
            object.__setattr__(self, k, v)

        # Restore NNX attributes
        # This sets self.raw_value, self._trace_state, self._var_metadata from nnx_state
        object.__setattr__(self, '_trace_state', nnx_state['_trace_state'])
        object.__setattr__(self, '_var_metadata', nnx_state['_var_metadata'])
        # raw_value might be None if pickled before init, or if Keras value is the source of truth
        object.__setattr__(self, 'raw_value', nnx_state.get('raw_value'))


        # Post-restore synchronization and checks:
        if self._initializer is not None and self._value is None:
            # Was deferred pre-pickle. Keras will handle re-initialization if used.
            # If raw_value from NNX state exists, it might be an inconsistency
            # or an indication that deferred init should use it. For now, Keras handles.
            pass
        
        if self._value is not None: # Keras _value is the source of truth after unpickling Keras state
            if self._nnx_init_pending:
                # This implies Keras was initialized (self._value exists), but NNX part wasn't fully done.
                # Common if pickled right after Keras init but before _complete_nnx_init naturally ran.
                self._complete_nnx_init() # This will use self._value to set raw_value
            else:
                # Both Keras and NNX parts were initialized. Ensure raw_value matches Keras _value.
                # This logic is similar to _direct_assign's sync.
                current_nnx_mutable = self._nnx_mutable_arg
                if current_nnx_mutable is None: current_nnx_mutable = self.trainable

                if current_nnx_mutable and nnx_utils.is_mutable_array(self.raw_value):
                    self.raw_value[...] = self._value
                else: # If not mutable_array or nnx_mutable is False
                    # self._value is already a JAX array.
                    object.__setattr__(self, 'raw_value', self._value)
        elif not self._nnx_init_pending and self.raw_value is not None:
            # Keras _value is None (e.g. from old pickle or never set), but NNX raw_value exists
            # Keras part should adopt NNX's value.
            _value_from_nnx = self.raw_value
            if nnx_utils.is_mutable_array(_value_from_nnx):
                 _value_from_nnx = _value_from_nnx.__array__()
            object.__setattr__(self, '_value', _value_from_nnx)
            # Also update Keras shape/dtype from this adopted value
            if hasattr(self, '_value') and self._value is not None:
                object.__setattr__(self, '_shape', self._validate_shape(self._value.shape))
                object.__setattr__(self, '_dtype', standardize_dtype(self._value.dtype))

    # __nnx_repr__ and other NNX specific Pytree/display methods can be inherited
    # or overridden if needed. __jax_array__ is inherited from JaxVariableImpl.

# --- Rest of your jax/core.py file ---
# Make sure `convert_to_tensor` and other functions that might check
# `isinstance(x, Variable)` now correctly refer to this new `Variable` class.

def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype) # Keras's standardize_dtype

    # This now correctly checks against the new integrated Variable class
    if isinstance(x, Variable): 
        if dtype is not None and x.dtype != dtype:
            # x.value correctly calls KerasVariable.value via MRO
            return x.value.astype(dtype)
        return x.value # Return Keras's self._value (JAX array)

    if isinstance(x, (jnp.ndarray, jax.Array)) and (
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

# ... (convert_to_numpy, is_tensor, shape, cast, etc. should be mostly fine)
# Make sure `is_tensor` does NOT include `Variable`
def is_tensor(x):
    if isinstance(x, (jnp.ndarray, jax.DeviceArray, jax.Array, jax_sparse.JAXSparse)): # More specific JAX types
        return True
    return False

# ... (rest of the file)


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
