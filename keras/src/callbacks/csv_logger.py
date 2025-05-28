import collections
import csv
import os  # For os.path.dirname and os.makedirs
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
        # self.keys will store the actual metric names for the CSV, excluding 'epoch'
        self.keys = []
        self.header_initialized = False # Flag to check if header/keys are set
        self.append_header = True # Whether to write the header row
        self.csv_file = None # Initialize to None

    def on_train_begin(self, logs=None):
        if self.append:
            if file_utils.exists(self.filename):
                with file_utils.File(self.filename, "r") as f:
                    first_line = f.readline()
                    # Only append header if the file is new or the first line is empty
                    self.append_header = not bool(first_line.strip())
            mode = "a"
        else:
            mode = "w"
        
        # Ensure the directory exists if filename includes a path
        file_dir = os.path.dirname(self.filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
            
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
                # Format iterables like "['item1', 'item2']" for CSV
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if not self.header_initialized:
            # `self.params` is set by the Keras training loop.
            # `self.params['metrics']` contains all metric names that Keras
            # expects to log, including 'val_...' versions if validation
            # is configured in model.fit(). This is the source of truth.
            if self.params and "metrics" in self.params:
                # These are all loggable metric names (e.g., 'loss', 'accuracy', 'val_loss')
                self.keys = list(self.params["metrics"])
                # 'epoch' is handled separately, remove if present in 'metrics'
                if "epoch" in self.keys:
                    self.keys.remove("epoch")
                self.keys = sorted(self.keys) # Ensure consistent column order
            else:
                # Fallback: If 'metrics' isn't in params (shouldn't happen in normal flow),
                # use keys from the current epoch's logs. This might miss val_* keys
                # if validation_freq > 1 and this is an early epoch where validation hasn't run.
                self.keys = sorted([k for k in logs.keys() if k != "epoch"])
            
            self.header_initialized = True

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            # The fieldnames for the CSV writer include 'epoch' plus sorted metric keys
            fieldnames = ["epoch"] + self.keys
            
            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        # Prepare row dictionary
        row_dict = collections.OrderedDict()
        row_dict["epoch"] = epoch
        for key in self.keys: # Iterate through metric keys determined at initialization
            row_dict[key] = handle_value(logs.get(key, np.nan)) # Use np.nan for missing values
        
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        if self.csv_file: # Check if csv_file was opened
            self.csv_file.close()
        self.writer = None
        # Reset for potential reuse of the callback instance in a new fit call
        self.header_initialized = False
        self.keys = []