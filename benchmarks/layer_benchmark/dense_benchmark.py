"""Benchmark dense layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.dense_benchmark \
    --benchmark_name=benchmark_dense_relu \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_dense(
    num_samples,
    batch_size,
    jit_compile=True,
    units=64,
    input_dim=256,
):
    layer_name = "Dense"
    init_args = {
        "units": units,
        "activation": None,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[input_dim],  # Typical 2D input (batch_size, input_dim)
        jit_compile=jit_compile,
    )

    print(
        f"\nBenchmarking Dense: units={units}, input_dim={input_dim}, "
        f"activation=None, jit_compile={jit_compile}"
    )
    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )
    # benchmark.benchmark_train(  # Training benchmarks can be verbose
    #     num_samples=num_samples,
    #     batch_size=batch_size,
    # )


def benchmark_dense_with_activation(
    num_samples,
    batch_size,
    jit_compile=True,
    activation="relu",
    units=64,
    input_dim=256,
):
    layer_name = "Dense"
    init_args = {
        "units": units,
        "activation": activation,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[input_dim],
        jit_compile=jit_compile,
    )
    print(
        f"\nBenchmarking Dense: units={units}, input_dim={input_dim}, "
        f"activation={activation}, jit_compile={jit_compile}"
    )
    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )
    # benchmark.benchmark_train(
    #     num_samples=num_samples,
    #     batch_size=batch_size,
    # )


# Define specific benchmark cases
def benchmark_dense_relu_units64_input256(num_samples, batch_size, jit_compile):
    benchmark_dense_with_activation(
        num_samples, batch_size, jit_compile, "relu", 64, 256
    )

def benchmark_dense_sigmoid_units64_input256(num_samples, batch_size, jit_compile):
    benchmark_dense_with_activation(
        num_samples, batch_size, jit_compile, "sigmoid", 64, 256
    )

def benchmark_dense_tanh_units64_input256(num_samples, batch_size, jit_compile):
    benchmark_dense_with_activation(
        num_samples, batch_size, jit_compile, "tanh", 64, 256
    )

def benchmark_dense_unfused_units64_input256(num_samples, batch_size, jit_compile):
    benchmark_dense(
        num_samples, batch_size, jit_compile, 64, 256
    )

def benchmark_dense_relu_units1024_input512(num_samples, batch_size, jit_compile):
    benchmark_dense_with_activation(
        num_samples, batch_size, jit_compile, "relu", 1024, 512
    )

def benchmark_dense_unfused_units1024_input512(num_samples, batch_size, jit_compile):
    benchmark_dense(
        num_samples, batch_size, jit_compile, 1024, 512
    )


BENCHMARK_NAMES = {
    "benchmark_dense_relu_units64_input256": benchmark_dense_relu_units64_input256,
    "benchmark_dense_sigmoid_units64_input256": benchmark_dense_sigmoid_units64_input256,
    "benchmark_dense_tanh_units64_input256": benchmark_dense_tanh_units64_input256,
    "benchmark_dense_unfused_units64_input256": benchmark_dense_unfused_units64_input256,
    "benchmark_dense_relu_units1024_input512": benchmark_dense_relu_units1024_input512,
    "benchmark_dense_unfused_units1024_input512": benchmark_dense_unfused_units1024_input512,
}


def main(_):
    benchmark_name_flag = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name_flag is None:
        for name, benchmark_fn in BENCHMARK_NAMES.items():
            print(f"Running benchmark: {name}")
            benchmark_fn(num_samples, batch_size, jit_compile)
        return

    if benchmark_name_flag not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name_flag}, `benchmark_name` "
            f"must be one of {list(BENCHMARK_NAMES.keys())}"
        )
    benchmark_fn = BENCHMARK_NAMES[benchmark_name_flag]
    print(f"Running benchmark: {benchmark_name_flag}")
    benchmark_fn(num_samples, batch_size, jit_compile)


if __name__ == "__main__":
    app.run(main)
