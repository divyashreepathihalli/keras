import collections
import csv

import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import file_utils


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
        self.keys = None  # CSV headers, determined on the first epoch_end
        self.append_header = True # Default, may be changed in on_train_begin
        self.csv_file = None

    def on_train_begin(self, logs=None):
        # Determine if header should be written
        if self.append:
            if file_utils.exists(self.filename):
                try:
                    with file_utils.File(self.filename, "r") as f:
                        first_line = f.readline()
                        # If file exists and has a non-empty first line,
                        # assume header is already there.
                        if first_line and first_line.strip():
                            self.append_header = False
                        else:
                            # File exists but is empty or first line is blank
                            self.append_header = True
                except IOError:
                    # Fallback: if can't read, assume we might need a header
                    self.append_header = True
            else:
                # File doesn't exist, so we will need to write a header
                self.append_header = True
            mode = "a"
        else: # Overwriting
            self.append_header = True
            mode = "w"

        # Ensure csv_file is None or closed before reassigning
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        self.csv_file = file_utils.File(self.filename, mode)

        # Reset writer and keys for this training session
        # self.keys will be determined by the first epoch's logs
        self.writer = None
        self.keys = None # Crucial: reset keys for potential reuse of callback instance

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

        if self.keys is None:  # Determine CSV headers on the first epoch
            # Initialize self.keys with (sorted) keys from the first epoch's logs
            # These are the metrics Keras *actually* reported for this first epoch.
            current_epoch_log_keys = list(logs.keys())
            
            # Check if any validation keys were *already present* in these first-epoch logs
            val_keys_found_in_first_epoch_logs = False
            for key in current_epoch_log_keys:
                if key.startswith("val_"):
                    val_keys_found_in_first_epoch_logs = True
                    break
            
            # This logic aims to ensure val_ versions of metrics are present in headers
            # if base metrics exist, even if val metrics don't appear in the very first epoch
            # (e.g., due to validation_freq > 1). This matches test expectations.
            if not val_keys_found_in_first_epoch_logs:
                # If no val_ keys were in the first epoch logs,
                # create val_ versions for all *non-val* keys that were found.
                non_val_keys_from_first_epoch = [
                    k for k in current_epoch_log_keys if not k.startswith("val_")
                ]
                additional_val_keys = ["val_" + k for k in non_val_keys_from_first_epoch]
                current_epoch_log_keys.extend(additional_val_keys)

            # Final set of keys for the header: unique and sorted
            self.keys = sorted(list(set(current_epoch_log_keys)))


        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + (self.keys or [])

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()
                self.append_header = False # Header written for this session

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        self.writer = None
        # self.keys is reset in on_train_begin for future runs