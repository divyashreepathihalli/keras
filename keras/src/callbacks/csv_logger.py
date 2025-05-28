import collections
import csv
import numpy as np
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import file_utils
import os


@keras_export("keras.callbacks.CSVLogger")
class CSVLogger(Callback):
    """Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.

    Example:

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    """

    def __init__(self, filename, separator=",", append=False):
        super().__init__()
        self.sep = separator
        self.filename = file_utils.path_to_string(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self._will_do_validation = False # Initialize instance variable

    def on_train_begin(self, logs=None):
        # Check if validation will be performed, set by the model's fit method
        # self.params is set by the base CallbackList.set_params method
        # params["do_validation"] = bool(val_data or validation_split)
        self._will_do_validation = self.params.get("do_validation", False)

        if self.append:
            if file_utils.exists(self.filename):
                with file_utils.File(self.filename, "r") as f:
                    # Check if the file is empty or if the first line is empty
                    first_line = f.readline()
                    self.append_header = not bool(first_line.strip())
            mode = "a"
        else:
            mode = "w"

        # Ensure the directory exists if filename includes a path
        if os.path.dirname(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self.csv_file = file_utils.File(self.filename, mode)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                isinstance(k, collections.abc.Iterable)
                and not is_zero_dim_ndarray
            ):
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if self.keys is None:
            # Initialize self.keys from the logs of the first epoch
            current_epoch_keys = sorted(list(logs.keys())) # Ensure it's a list and sorted
            self.keys = list(current_epoch_keys) # Use a copy

            if self._will_do_validation:
                # If validation is expected, ensure val_ versions of training keys are present.
                # This handles validation_freq > 1 where val_keys might not be in the first epoch.

                # Identify training-like keys from the first epoch's logs
                # (those that don't already start with "val_")
                training_keys_from_first_epoch = [
                    k for k in current_epoch_keys if not k.startswith("val_")
                ]

                for train_key in training_keys_from_first_epoch:
                    val_key_to_add = "val_" + train_key
                    if val_key_to_add not in self.keys:
                        self.keys.append(val_key_to_add)

                # Ensure final keys are sorted for consistent header
                self.keys.sort()
            # If not self._will_do_validation, self.keys will only contain
            # what was in current_epoch_keys (i.e., no val_* keys unless
            # they unexpectedly came from logs, which shouldn't happen).

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        if hasattr(self, 'csv_file') and self.csv_file: # Check if csv_file was opened
            self.csv_file.close()
        self.writer = None
