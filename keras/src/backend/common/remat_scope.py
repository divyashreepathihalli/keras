from keras.src.backend.common import global_state


class RematScope:
    """A context manager for enabling rematerialization in Keras.

    Rematerialization (gradient checkpointing) trades memory for computation by
    recomputing intermediate activations during the backward pass. This is
    particularly useful for training large models or large batch sizes within
    limited memory constraints.

    Args:
        mode: Rematerialization mode to apply.
            Options:
            - "full": Apply rematerialization globally to all supported
              operations.
            - "activations": Apply rematerialization only to activation layers.
            - "larger_than": Apply rematerialization to layers with output sizes
              larger than `output_size_threshold`.
            - "list_of_layers": Apply rematerialization to a specific list of
              layer names.
            - None: Disable rematerialization.
        output_size_threshold: Output size threshold for the
            `"larger_than"` mode. Layers producing outputs larger than this
            threshold will be rematerialized. Default is `1024`.
        layer_names: List of layer names for the
            `"list_of_layers"` mode. Default is an empty list.

    Examples:
        Using "list_of_layers" mode:

        ```python
        from keras.src.backend.common.remat_scope import RematScope

        with RematScope(mode="list_of_layers", layer_names=["dense_1",
        "conv2d_1"]):
            layer1 = keras.layers.Dense(128, name="dense_1")
            layer2 = keras.layers.Conv2D(64, (3, 3), name="conv2d_1")
            layer3 = keras.layers.Dense(64, name="dense_2")

            # Only layer1 and layer2 will apply rematerialization
            output1 = layer1(input_tensor)
            output2 = layer2(output1)
            output3 = layer3(output2)
        ```

        Using "larger_than" mode with a specific output size threshold:

        ```python
        from keras.src.backend.common.remat_scope import RematScope

        with RematScope(mode="larger_than", output_size_threshold=2048):
            layer = keras.layers.Conv2D(64, (3, 3))
            output = layer(input_tensor)  # Conv2D outputs larger than 2048
        ```

        Nested scopes for fine-grained control:

        ```python
        from keras.src.backend.common.remat_scope import RematScope

        with RematScope(mode="full"):
            layer1 = keras.layers.Dense(128, activation='relu')
            with RematScope(mode="larger_than", output_size_threshold=512):
                layer2 = keras.layers.Conv2D(32, (3, 3))
                output = layer2(layer1(input_tensor))
        ```
    """

    def __init__(
        self, mode="full", output_size_threshold=1024, layer_names=None
    ):
        if mode not in {
            "full",
            "activations",
            "larger_than",
            "list_of_layers",
            None,
        }:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are: "
                "'full', 'activations', 'larger_than', 'list_of_layers', or "
                " None."
            )
        self.mode = mode
        self.output_size_threshold = output_size_threshold
        self.layer_names = layer_names or []
        self._pop_on_exit = False

    def __enter__(self):
        remat_scope_stack = global_state.get_global_attribute(
            "remat_scope_stack", default=[], set_to_default=True
        )
        remat_scope_stack.append(self)
        self._pop_on_exit = True
        return self

    def __exit__(self, *args, **kwargs):
        if self._pop_on_exit:
            remat_scope_stack = global_state.get_global_attribute(
                "remat_scope_stack"
            )
            remat_scope_stack.pop()


def get_current_remat_mode():
    """Get the current rematerialization mode and associated settings.

    Returns:
        dict: A dictionary containing the rematerialization mode and other
            settings.
            Example:
                {
                    "mode": "list_of_layers",
                    "output_size_threshold": 1024,
                    "layer_names": ["dense_1", "conv2d_1"]
                }
    """
    remat_scope_stack = global_state.get_global_attribute("remat_scope_stack")
    if remat_scope_stack is None or not remat_scope_stack:
        return None
    active_scope = remat_scope_stack[-1]
    return {
        "mode": active_scope.mode,
        "output_size_threshold": active_scope.output_size_threshold,
        "layer_names": active_scope.layer_names,
    }