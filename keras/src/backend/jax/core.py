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
from functools import partial # For Pytree unflatten if needed

from keras.src.backend.common import KerasVariable as CommonKerasVariable # Alias to avoid confusion
from keras.src.backend.common import standardize_dtype as keras_standardize_dtype # Alias
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


def in_stateless_scope():
    return global_state.get_global_attribute("stateless_scope") is not None


def get_stateless_scope():
    return global_state.get_global_attribute("stateless_scope")


def shape_equal(a_shape, b_shape):
    """Return whether a_shape == b_shape (allows None entries)."""
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is not None and e2 is not None and e1 != e2:
            return False
    return True


# This is the existing Variable class from the JAX backend
class KerasJaxVariableImpl(CommonKerasVariable): # Renamed for clarity in this combined file
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
        self._direct_assign(value) # This will call the _direct_assign of the most derived class

    def _direct_assign(self, value):
        # This base implementation will be called by JaxNnxVariable's _direct_assign
        if self._layout is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value # Keras internal JAX array

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    def __jax_array__(self):
        return self.value # Crucial for JAX/NNX integration


# NEW INTEGRATED VARIABLE CLASS
class Variable(KerasJaxVariableImpl, nnx.Variable):
    def __init__(
        self,
        initializer,
        shape=None,
        dtype=None,
        trainable=True,
        # Keras specific args from KerasVariable
        autocast=True,
        aggregation="none",
        synchronization="auto",
        name=None,
        # Keras JAX backend specific
        layout=None,
        # NNX specific args
        nnx_mutable=None, # NNX's own mutable flag
        **nnx_metadata # For nnx.Variable's **metadata
    ):
        # We need to call KerasJaxVariableImpl.__init__ first
        # KerasJaxVariableImpl's __init__ takes `layout` specifically
        # and forwards other Keras common args to CommonKerasVariable.__init__
        super(KerasJaxVariableImpl, self).__init__( # Explicitly call KerasJaxVariableImpl's __init__
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            aggregation=aggregation,
            synchronization=synchronization,
            name=name,
            layout=layout,
            # CommonKerasVariable's __init__ has a **kwargs catch-all,
            # but nnx_metadata is for nnx.Variable.
            # Ensure no nnx_metadata keys clash with Keras ones or handle separation.
            # For now, assuming nnx_metadata are distinct.
        )

        # Store NNX args for potential deferred initialization
        self._nnx_mutable_arg = nnx_mutable
        self._nnx_metadata_arg = nnx_metadata.copy()
        self._nnx_init_pending = True # Flag to indicate nnx.Variable part is not yet initialized

        # If Keras initialization was not deferred, self._value is now set.
        # So we can proceed to initialize the nnx.Variable part.
        if self._initializer is None: # Keras way to check if not deferred
            self._finish_nnx_init()

    def _finish_nnx_init(self):
        """Initializes the nnx.Variable part of this instance."""
        if not self._nnx_init_pending:
            return # Already done

        if self._value is None:
            # This can happen if _deferred_initialize was called but _value somehow didn't get set
            # Or if this is called too early. Keras's _initializer should not be None here.
            raise ValueError(
                "Cannot initialize NNX part: Keras self._value is None, "
                "but Keras initializer is also None (should not be deferred)."
            )

        # Determine nnx_mutable for nnx.Variable.__init__
        # If user didn't specify nnx_mutable, default to Keras's trainable status.
        current_nnx_mutable = self._nnx_mutable_arg
        if current_nnx_mutable is None:
            current_nnx_mutable = self.trainable # A sensible default link

        # Now call nnx.Variable.__init__
        # object.__setattr__ is used by nnx.Variable for _trace_state, raw_value, _var_metadata
        # We are effectively doing what nnx.Variable.__init__ does.
        # super(nnx.Variable, self) here would be problematic with MRO if KerasJaxVariableImpl had an __init__ with same sig
        # So, directly call nnx.Variable's init.
        nnx.Variable.__init__(
            self,
            value=self._value, # Use Keras's initialized JAX array
            mutable=current_nnx_mutable,
            **self._nnx_metadata_arg
        )
        self._nnx_init_pending = False
        # Clean up stored args if you want, but they are small
        # del self._nnx_mutable_arg
        # del self._nnx_metadata_arg

    def _deferred_initialize(self):
        # This is called by Keras when it's time to actually create the variable's value
        super()._deferred_initialize() # Calls KerasJaxVariableImpl._deferred_initialize
                                       # which sets self._value via self._initialize -> _direct_assign
        # Now self._value is guaranteed to be set by Keras.
        # So, we can initialize the nnx.Variable part.
        self._finish_nnx_init()

    def _direct_assign(self, value_to_assign):
        # Called by KerasJaxVariableImpl._initialize and KerasVariable.assign
        # First, let KerasJaxVariableImpl handle its assignment (sets self._value, handles layout)
        super()._direct_assign(value_to_assign) # This sets self._value

        # After self._value is updated by Keras, sync nnx.Variable.raw_value
        # Only if NNX part is already initialized.
        if not self._nnx_init_pending:
            # nnx.Variable.value setter handles on_set_value hooks and mutable_array updates
            # So, using it is cleaner than directly setting self.raw_value.
            # However, nnx.Variable.value setter calls self.raw_value[...] = value
            # or object.__setattr__(self, 'raw_value', value).
            # We need to ensure the value passed is the one Keras just set in self._value.

            # Determine how nnx.Variable would have stored it (mutable or not)
            nnx_stores_mutable = False
            if self._nnx_mutable_arg is None: # Check how nnx_mutable was resolved
                nnx_stores_mutable = self.trainable
            else:
                nnx_stores_mutable = self._nnx_mutable_arg

            if nnx_stores_mutable and nnx.utils.is_mutable_array(self.raw_value):
                 # If raw_value is a mutable_array, update its content
                self.raw_value[...] = self._value
            else:
                # Otherwise, self.raw_value should be the plain JAX array.
                # Or, if nnx_mutable was True but raw_value isn't mutable_array (e.g. init issue),
                # we might need to re-wrap. For safety, stick to direct object.__setattr__.
                object.__setattr__(self, 'raw_value', self._value)


    # For NNX methods that change state, we need to ensure Keras state is also updated.
    # nnx.Variable.value setter is a good target if other NNX methods use it.
    # Let's override the `value` property's setter from nnx.Variable to ensure sync.

    @property
    def value(self):
        # MRO will pick KerasVariable.value:
        # JaxNnxVar -> KerasJaxVarImpl -> CommonKerasVar -> nnx.Variable
        # CommonKerasVar.value handles stateless scope, autocast, uses self._value
        return super().value

    @value.setter
    def value(self, new_value):
        # This setter will be called if someone does `my_var.value = ...`
        # We want Keras's `assign` logic to run.
        self.assign(new_value) # assign will call _direct_assign, which syncs raw_value

    # Overriding NNX methods that modify `raw_value` or `_var_metadata` directly
    # to ensure Keras's `_value` and other Keras states are in sync.

    def copy_from(self, other: nnx.Variable): # type: ignore
        if not isinstance(other, nnx.Variable): # Basic check from nnx
             raise TypeError(f"Expected nnx.Variable, got {type(other).__name__}")
        if not isinstance(other, Variable): # More specific check for our type
            # If `other` is a plain nnx.Variable, it won't have Keras attributes
            # This could lead to an inconsistent state for the Keras part.
            # Depending on strictness, you might raise an error or try a best-effort copy.
            # For now, let's allow it but be aware Keras part might not be fully synced.
            pass

        # Let nnx.Variable handle its part (updates self.raw_value and self._var_metadata)
        # Need to call nnx.Variable.copy_from specifically.
        nnx.Variable.copy_from(self, other)

        # Now, self.raw_value is updated. Sync Keras's self._value.
        # Extract the JAX array if raw_value is a nnx.mutable_array
        keras_value_to_assign = self.raw_value
        if nnx.utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__() # Get underlying JAX array

        self.assign(keras_value_to_assign) # Use Keras assign for full Keras logic

        # Sync Keras-specific attributes if `other` is also a JaxNnxVariable
        if isinstance(other, Variable):
            self.trainable = other.trainable # Uses KerasVariable's property setter
            self._autocast = other._autocast # Direct Keras attribute
            self._aggregation = other._aggregation # Direct Keras attribute
            # Name and path are usually structural and unique, avoid copying unless intended.
            # self._layout might also need syncing if relevant for `other`.
            if hasattr(other, '_layout'):
                 self._layout = other._layout
        # Note: self._var_metadata (NNX) was updated by nnx.Variable.copy_from.
        # If Keras attributes were stored in there, they'd be copied.

    def update_from_state(self, variable_state: nnx.graph.VariableState): # type: ignore
        # Let nnx.Variable handle its part (updates self.raw_value and self._var_metadata)
        nnx.Variable.update_from_state(self, variable_state)

        # Sync Keras's self._value
        keras_value_to_assign = self.raw_value
        if nnx.utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__()

        self.assign(keras_value_to_assign)

        # Potentially sync Keras attributes if they were part of variable_state.metadata
        # This depends on how variable_state was created.
        # Example: if 'trainable' was in variable_state._var_metadata:
        if 'trainable' in variable_state._var_metadata: # type: ignore
            self.trainable = variable_state._var_metadata['trainable'] # type: ignore
        if 'autocast' in variable_state._var_metadata: # type: ignore
            self._autocast = variable_state._var_metadata['autocast'] # type: ignore
        # ... and so on for other Keras attributes you might have packed.

    # __getstate__ and __setstate__ for pickling:
    # nnx.Variable defines them. They save/load raw_value, _trace_state, _var_metadata.
    # Keras variables are often part of a layer/model which has its own saving.
    # If direct pickling of these variables is needed, ensure Keras state is also included.
    def __getstate__(self):
        keras_state = {
            # Keras common attributes (from CommonKerasVariable)
            "_name": self._name,
            "_path": self._path,
            "_trainable": self._trainable,
            "_dtype": self._dtype,
            "_shape": self._shape, # Shape is important
            "_autocast": self._autocast,
            "_aggregation": self._aggregation,
            "_synchronization": self._synchronization,
            "_regularizer": self._regularizer, # If you support them
            "_constraint": self._constraint,   # If you support them
            # Keras JAX backend specific
            "_layout": self._layout,
            # Value itself (will be part of nnx_state's raw_value too)
            "_value": self._value, # Keras's value (JAX array)
            "_initializer": self._initializer, # In case it's not initialized
            # NNX specific args that were stored at init
            "_nnx_mutable_arg": self._nnx_mutable_arg,
            "_nnx_metadata_arg": self._nnx_metadata_arg,
            "_nnx_init_pending": self._nnx_init_pending,
        }
        nnx_state = nnx.Variable.__getstate__(self)
        return {"keras_state": keras_state, "nnx_state": nnx_state}

    def __setstate__(self, state):
        keras_state = state["keras_state"]
        nnx_state = state["nnx_state"]

        # Restore Keras attributes
        for k, v in keras_state.items():
            object.__setattr__(self, k, v)

        # Restore NNX attributes using its __setstate__
        nnx.Variable.__setstate__(self, nnx_state)

        # Ensure consistency after loading, especially if _value and raw_value diverged
        # If Keras was deferred, self._value might be None in keras_state,
        # but nnx_state.raw_value would have the actual array.
        if self._initializer is not None and self._value is None: # Was deferred pre-pickle
            if not self._nnx_init_pending and hasattr(self, 'raw_value') and self.raw_value is not None:
                 # If NNX part was initialized and has a value, Keras part should adopt it.
                 # This path assumes pickle happened *after* deferred init.
                 # If pickle happened *before* deferred init, then raw_value might also be placeholder.
                 # This part is complex because of deferred init.
                 # Safest: if self._initializer exists, Keras expects to run it.
                 # But if we are unpickling a *trained* variable, self._value from pickle is king.
                 # Let's assume pickled _value is the source of truth if present.
                 pass # self._value is already set from keras_state.

        # If self._value exists (from Keras state), ensure nnx.raw_value matches
        if self._value is not None:
            if self._nnx_init_pending: # If NNX wasn't init'd (e.g. Keras deferred and pickled before init)
                self._finish_nnx_init() # This will use self._value
            else: # NNX was init'd, ensure raw_value syncs with loaded _value
                # This is similar to _direct_assign's sync logic.
                current_nnx_mutable = self._nnx_mutable_arg
                if current_nnx_mutable is None: current_nnx_mutable = self.trainable

                if current_nnx_mutable and nnx.utils.is_mutable_array(self.raw_value):
                    self.raw_value[...] = self._value
                else:
                    object.__setattr__(self, 'raw_value', self._value)
        elif not self._nnx_init_pending and hasattr(self, 'raw_value') and self.raw_value is not None:
            # Keras _value is None (maybe from an old pickle or error), but NNX raw_value exists
            # This implies Keras part should take NNX's value
            object.__setattr__(self, '_value', self.raw_value) # TODO: unwrap mutable array?

    # NNX Pytree unflattening (nnx.Variable uses a partial, we need to ensure type)
    # This might be automatically handled by nnx.Variable's __init_subclass__
    # if not, it would be:
    # @classmethod
    # def _custom_unflatten(cls, keys, values):
    #     tree_state = dict(zip(keys, values))
    #     return cls.from_metadata(
    #         value=tree_state.pop('raw_value'),
    #         attributes=tree_state['_var_metadata'], # NNX stores all else in _var_metadata
    #         # Keras attributes would need to be part of metadata or handled separately
    #     )

# Make sure the `Variable` class defined above is the one exported or used by the backend.
# Other functions like convert_to_tensor, etc., remain the same.

def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    # ... (existing implementation)
    # Make sure it uses the correct Variable class if it checks isinstance(x, Variable)
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
    if dtype is not None:
        dtype = keras_standardize_dtype(dtype) # Use aliased import

    # Check against the unified Variable class
    if isinstance(x, Variable): # This now refers to JaxNnxVariable
        if dtype is not None and x.dtype != dtype:
            # Accessing x.value here will go through KerasVariable.value logic
            # which is good (autocast, stateless scope if active)
            return x.value.astype(dtype)
        return x.value

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

    if not is_tensor(x) and keras_standardize_dtype(dtype) == "bfloat16":
        return jnp.asarray(x).astype(dtype)
    return jnp.asarray(x, dtype=dtype)


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, (jnp.ndarray, jax.Array)) and (
        dtype is None or x.dtype == dtype
    ):
        # Skip the conversion early if the instance is already a JAX array.
        # This is important in the multi-process context since jax.array(x) for
        # an existing distributed jax array will raise error.
        return x

    if isinstance(x, Variable):
        if dtype is not None and x.dtype != dtype:
            return x.value.astype(dtype)
        return x.value

    if isinstance(x, jax_sparse.JAXSparse):
        if sparse is not None and not sparse:
            x = x.todense()
        elif dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        else:
            return x

    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return jnp.asarray(x).astype(dtype)
    return jnp.asarray(x, dtype=dtype)


def convert_to_numpy(x):
    if isinstance(x, jax_sparse.JAXSparse):
        x = x.todense()
    if is_tensor(x) and x.dtype == "bfloat16":
        return np.array(x, dtype=ml_dtypes.bfloat16)
    return np.array(x)


def is_tensor(x):
    if isinstance(x, (jnp.ndarray, jax_sparse.JAXSparse)):
        return True
    return False


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
