import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from keras.src import tree
from keras.src.backend import config
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


class JaxVariable(KerasVariable):
    def __init__(self, *args, layout=None, **kwargs):
        # Intercept layout parameter so that it is available
        # during initialization.
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _initialize(self, value):
        # Note that variable.shape is needed by distribution_lib
        self._shape = self._validate_shape(value.shape)
        # We can't import the keras/distribution/distribution_lib
        # due to circular dependency.
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            tensor_layout = distribution.get_variable_layout(self)
            from keras.src.distribution import TensorLayout

            if isinstance(tensor_layout, TensorLayout):
                self._layout = tensor_layout.backend_layout
            else:
                self._layout = tensor_layout
        self._direct_assign(value)

    def _direct_assign(self, value):
        if self._layout is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


_JAX_VARIABLE_TYPE = JaxVariable
if config.is_nnx_enabled():
    from flax import nnx

    class NnxVariable(KerasVariable, nnx.Module):
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
        ):
            # 1. Call KerasVariable.__init__ to set up Keras-side attributes
            # This will resolve shape, dtype, name, path, and set self._initializer to a callable.
            # It also sets self.trainable.
            super().__init__(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                synchronization=synchronization,
                name=name
            )

            # Cache Keras-resolved properties before nnx.Module.__init__ might affect them
            # (though unlikely for underscore-prefixed attributes if nnx.Module is well-behaved)
            _keras_initializer = self._initializer
            _keras_shape = self.shape
            _keras_dtype = self.dtype
            _keras_trainable = self.trainable
            _keras_path = self._path
            _keras_name = self._name
            _keras_autocast = self._autocast
            _keras_aggregation = self._aggregation
            _keras_synchronization = self._synchronization

            # 2. Initialize nnx.Module part. This sets up self._object__state.
            nnx.Module.__init__(self)

            # Re-apply Keras bookkeeping attributes directly to __dict__ or via object.__setattr__
            # to ensure they are on the instance if nnx.Module.__init__ was restrictive.
            # These are not NNX state, just Keras metadata.
            object.__setattr__(self, "_initializer", _keras_initializer)
            object.__setattr__(self, "_shape", _keras_shape)
            object.__setattr__(self, "_dtype", _keras_dtype)
            object.__setattr__(self, "_trainable", _keras_trainable) # Keras's view of trainable
            object.__setattr__(self, "_path", _keras_path)
            object.__setattr__(self, "_name", _keras_name)
            object.__setattr__(self, "_autocast", _keras_autocast)
            object.__setattr__(self, "_aggregation", _keras_aggregation)
            object.__setattr__(self, "_synchronization", _keras_synchronization)
            object.__setattr__(self, "_layout", layout) # Custom JAX layout for this variable

            # 3. Create the internal nnx.Param or nnx.Variable using cached Keras properties
            if _keras_initializer is None:
                # This can happen if KerasVariable.__init__ received an already concrete value as initializer.
                # In that case, KerasVariable._initialize would have been called.
                # The value would be in KerasVariable._value if we hadn't overridden _initialize.
                # This scenario needs robust handling: get the concrete value KerasVariable would have stored.
                # For now, assuming initializer given to NnxVariable is string or Keras callable.
                raise ValueError(
                    f"NnxVariable '{_keras_name}' at path '{_keras_path}' resolved to a None initializer. "
                    "This typically means a concrete value was passed as initializer to KerasVariable, "
                    "and that value should be used here. This path needs refinement."
                )

            initial_value_array = _keras_initializer(_keras_shape, dtype=_keras_dtype)
            initial_value_array = self._convert_to_tensor(initial_value_array, dtype=_keras_dtype)

            print(f"[NNX_DEBUG] NnxVariable.__init__ for '{_keras_path}' ({id(self)}): Keras trainable flag is {_keras_trainable}")

            if _keras_trainable:
                self.actual_param = nnx.Param(initial_value_array)
                print(f"[NNX_DEBUG] NnxVariable '{_keras_path}': Created self.actual_param (id={id(self.actual_param)}) with value id {id(initial_value_array)}, shape {initial_value_array.shape}")
            else:
                self.actual_state = nnx.Variable(initial_value_array, collection="variables", mutable=False)
                print(f"[NNX_DEBUG] NnxVariable '{_keras_path}': Created self.actual_state (id={id(self.actual_state)}) with value id {id(initial_value_array)}, shape {initial_value_array.shape}")

        # --- KerasVariable API Implementation ---

        # Override _initialize from KerasVariable to control value storage
        def _initialize(self, value):
            # This method is called by KerasVariable's __init__ if the initializer
            # was a concrete value, or by deferred_initialize.
            # KerasVariable expects this to set self._value and self._shape.
            # We need self.shape to be set for the nnx.Param/Variable creation above.
            # The KerasVariable's self._value is now effectively ignored.
            current_shape = self._validate_shape(value.shape) # from KerasVariable
            object.__setattr__(self, "_shape", current_shape) # bypass our setattr
            object.__setattr__(self, "_ndim", len(current_shape))

            # The actual JAX array is now managed by self.actual_param or self.actual_state,
            # which should have been initialized in __init__ using this 'value' (via initializer).
            print(f"[NNX_DEBUG] NnxVariable '{self.path}': _initialize called. Shape set to {self._shape}. KerasVariable's self._value not used for storage.")

        # Override _direct_assign from KerasVariable
        def _direct_assign(self, value):
            # This is called by KerasVariable's assign if not fully overridden.
            # Delegate to our main assign method.
            # print(f"[NNX_DEBUG] NnxVariable '{self.path}': _direct_assign called. Delegating to self.assign.")
            self.assign(value) # Will use the overridden NnxVariable.assign

        # Override value property from KerasVariable
        @property
        def value(self):
            # print(f"[NNX_DEBUG] NnxVariable.value GET for '{self.path}' id({id(self)})")
            if self.trainable:
                if hasattr(self, 'actual_param'):
                    return self.actual_param.value
                else:
                    # This might happen if accessed before __init__ fully completes internal setup.
                    # Or if KerasVariable's _initializer logic runs before actual_param is created.
                    # KerasVariable._initializer is called by KerasVariable.__init__
                    # We need to ensure actual_param/actual_state is created before .value is needed by KerasVariable init if so.
                    # The current __init__ order should be fine: KerasVar init (which sets self._initializer), then nnx.Module init, then create actual_param/state.
                    raise AttributeError(f"NnxVariable '{self.path}' is trainable but 'actual_param' not yet initialized.")
            else:
                if hasattr(self, 'actual_state'):
                    return self.actual_state.value
                else:
                    raise AttributeError(f"NnxVariable '{self.path}' is not trainable but 'actual_state' not yet initialized.")

        # Override assign method from KerasVariable
        def assign(self, value_to_assign):
            # print(f"[NNX_DEBUG] NnxVariable.assign for '{self.path}'")
            converted_value = self._convert_to_tensor(value_to_assign, dtype=self.dtype)
            if not common_variables.shape_equal(converted_value.shape, self.shape):
                 raise ValueError(f"Shape mismatch for {self.path}: expected {self.shape}, got {converted_value.shape}")

            if self.trainable:
                self.actual_param.value = converted_value
            else:
                self.actual_state.value = converted_value
            return converted_value

        # _convert_to_tensor is used by KerasVariable, ensure it uses JAX backend's version
        def _convert_to_tensor(self, v, dtype=None):
            from keras.src.backend.jax.core import convert_to_tensor as jax_convert_to_tensor # Local import
            return jax_convert_to_tensor(v, dtype=dtype)

        # Ensure __jax_array__ for JAX operations (e.g. jnp.add(var, 1))
        def __jax_array__(self):
            return self.value

        # We might need to override other KerasVariable methods if they directly access self._value.
        # For example, numpy(). KerasVariable.numpy() calls np.array(self), which calls self.__array__.
        # KerasVariable.__array__ calls self.value.__array__(). This should be fine if self.value works.

        # No longer needed as NnxVariable is not an nnx.Variable itself.
        # def __getstate__(self): ...
        # def __setstate__(self, state): ...

        # The print statements from the previous version for init/value/assign can be re-added if needed.
        # For brevity, I'm omitting them from this direct replacement block.

        # Todo: NNX has agreed to fix it on their end. I will remove it once
        # that is done
        def __hash__(self):
            return id(self)

    _JAX_VARIABLE_TYPE = NnxVariable


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

    if isinstance(x, _JAX_VARIABLE_TYPE):
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
    if isinstance(variable, _JAX_VARIABLE_TYPE):
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
