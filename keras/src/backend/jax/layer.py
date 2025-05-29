from flax import nnx

# Import Param, trying stable then experimental, and log outcome
NNXParam = None # Initialize to None
_PARAM_IMPORT_PATH = None
try:
    from flax.nnx import Param as NNXParamImported
    NNXParam = NNXParamImported
    _PARAM_IMPORT_PATH = "flax.nnx.Param"
except ImportError:
    try:
        from flax.experimental.nnx import Param as ExperimentalNNXParamImported
        NNXParam = ExperimentalNNXParamImported
        _PARAM_IMPORT_PATH = "flax.experimental.nnx.Param"
    except ImportError:
        # NNXParam remains None
        pass # Error will be printed below

if _PARAM_IMPORT_PATH:
    print(f"INFO: JaxLayer will use Param from: {_PARAM_IMPORT_PATH}")
else:
    print("ERROR: JaxLayer could not import Param from flax.nnx or flax.experimental.nnx. NNX Param wrapping will be skipped.")
    # Define a placeholder that would error if called, to make it clear if used when None
    def NNXParam(*args, **kwargs): # This redefines NNXParam if it was None
        raise ImportError("NNXParam could not be resolved during import and was called.")

from keras.src.backend.jax.core import Variable as KerasJaxVariable
import jax


class JaxLayer(nnx.Module):
    def __init_subclass__(cls, **kwargs): 
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        # This __new__ method is from the state at the end of Subtask #4 (Turn 3 of prior interaction).
        # It calls super().__new__ (which includes Operation.__new__ and nnx.ObjectBase.__new__)
        # and stores __init_args and __init_kwargs.
        # It does NOT explicitly set _object__state here anymore.
        instance = super(JaxLayer, cls).__new__(cls, *args, **kwargs)

        instance.__init_args = args 
        instance.__init_kwargs = kwargs
        return instance

    # NO __init__ method in JaxLayer.

    def __setattr__(self, name: str, value: any):
        print(f"DEBUG: JaxLayer.__setattr__: Attempting to set '{name}' with value of type {type(value)}")

        is_keras_jax_variable = isinstance(value, KerasJaxVariable)
        
        if NNXParam is not None and is_keras_jax_variable:
            actual_value = value.value # Extract jax.Array from KerasJaxVariable
            print(f"DEBUG: JaxLayer.__setattr__: '{name}' identified as KerasJaxVariable. Actual value type: {type(actual_value)}")

            if isinstance(actual_value, jax.Array): # Ensure it's a JAX array
                print(f"DEBUG: JaxLayer.__setattr__: Wrapping '{name}' with NNXParam.")
                # NNXParam was aliased or is the placeholder
                param_value = NNXParam(actual_value) 
                super().__setattr__(name, param_value)
                # Check type after setting
                # getattr can trigger __getattribute__ or __getattr__, be careful in debug prints if those are also overridden.
                # For safety, access __dict__ if possible, or just rely on the flow.
                # new_type = type(getattr(self, name, None)) 
                # print(f"DEBUG: JaxLayer.__setattr__: '{name}' successfully set as NNXParam. Attribute type is now: {new_type}")
                # Printing new_type here can be misleading if getattr itself has side-effects or if 'name' is a property.
                # Let's defer final type check to after super().__setattr__ using a direct __dict__ access if safe.
                if hasattr(self, "__dict__") and name in self.__dict__:
                     new_type_direct = type(self.__dict__[name])
                     print(f"DEBUG: JaxLayer.__setattr__: '{name}' successfully set as NNXParam. Attribute type via __dict__ is now: {new_type_direct}")
                else:
                     print(f"DEBUG: JaxLayer.__setattr__: '{name}' successfully set as NNXParam. Cannot verify type via __dict__ immediately.")

            else:
                print(f"DEBUG: JaxLayer.__setattr__: '{name}' KerasJaxVariable's value is not a jax.Array (type: {type(actual_value)}). Setting as regular attribute (original KerasJaxVariable).")
                super().__setattr__(name, value) # Pass original KerasJaxVariable wrapper
        else:
            if NNXParam is None:
                print(f"DEBUG: JaxLayer.__setattr__: NNXParam not available. Setting '{name}' as regular attribute.")
            elif not is_keras_jax_variable:
                # This path will also be taken for internal attributes like _lock, _tracker, _object__state etc.
                print(f"DEBUG: JaxLayer.__setattr__: '{name}' (type: {type(value)}) is not a KerasJaxVariable. Setting as regular attribute.")
            
            super().__setattr__(name, value)
            # For non-weights, a type check here is less critical for NNX Param wrapping.
            # new_type = type(getattr(self, name, None))
            # print(f"DEBUG: JaxLayer.__setattr__: '{name}' set as regular attribute. Attribute type is now: {new_type}")

        # Add a final print for all cases to see the end state of the attribute for critical names
        if name in ["kernel", "bias"]: # Check common weight names
             final_attr_value = getattr(self, name, None) # Use getattr to ensure property-like access works if any
             final_attr_type = type(final_attr_value)
             print(f"DEBUG: JaxLayer.__setattr__: Final type of '{name}' is {final_attr_type}")
             if isinstance(final_attr_value, NNXParam):
                 print(f"DEBUG: JaxLayer.__setattr__: '{name}' is indeed an NNXParam. Value type: {type(final_attr_value.value)}")
             elif isinstance(final_attr_value, KerasJaxVariable):
                 print(f"DEBUG: JaxLayer.__setattr__: '{name}' is a KerasJaxVariable. Value type: {type(final_attr_value.value)}")
             elif isinstance(final_attr_value, jax.Array):
                 print(f"DEBUG: JaxLayer.__setattr__: '{name}' is a jax.Array.")
             else:
                 print(f"DEBUG: JaxLayer.__setattr__: '{name}' is neither NNXParam, KerasJaxVariable, nor jax.Array.")
