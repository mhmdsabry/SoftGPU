# SoftGPU: A Simple Educational GPU Simulator

**SoftGPU** is a simple educational GPU simulator inspired by *"Programming Massively Parallel Processors"*. It simulates threads, warps, and memory to demonstrate SIMD and kernel launching in a simplified way.

## Key Features

- **Warp & Thread Simulation**: Simulates the execution of warps, each containing a set of threads, and demonstrates how warps are scheduled across SMs.
- **Memory Simulation**: Simulates global and shared memory, with the ability to allocate, release, and transfer data between them.
- **Profiling & Visualization**: Provides a profiling tool to trace kernel executions, track memory access patterns, and visualize memory usage over time.

## src Modules

### `compute.py`

Simulates the computation aspects of a GPU, including warps and streaming multiprocessors:

- **Warp**: Represents a group of threads executed in parallel. It handles thread scheduling and execution.
- **GPUWarpScheduler**: Manages the scheduling of warps and distributes them across streaming multiprocessors.
- **StreamingMultiprocessor**: Simulates an SM that can execute multiple warps concurrently.

### `memory.py`

Models the memory system of a GPU, including global and shared memory:

- **GPUMemory**: Simulates global and shared memory allocation, read/write operations, and memory transfers.
- Tracks memory accesses to allow profiling of memory operations.

### `profiler.py`

A profiler that logs and visualizes kernel execution and memory operations:

- **GPUProfiler**: Tracks kernel execution times, memory accesses, and visualizes memory usage over time. It provides insights into how a GPU kernel behaves during execution.

## Example Usage

Here’s an example showing how to initialize the simulator, launch a kernel, and visualize memory usage:

```python
import numpy as np
from gpu import GPU

def example_kernel(warp_id, thread_id, global_memory, input_offset, output_offset):
    global_thread_id = thread_id + warp_id * 32 # Assuming threads per wrap is constant
    # Read input from global memory
    input_data = global_memory[input_offset + global_thread_id]
    # Perform computation
    result = input_data + warp_id
    # Write result to global memory
    global_memory[output_offset + global_thread_id] = result
    print(f"Thread {thread_id} writing to global memory[{output_offset + global_thread_id}] = {result}")


# GPU Execution
gpu = GPU(num_sms=2, threads_per_warp=32, warps_per_sm=2, global_memory_size=1024, shared_memory_size=256)

input_data = np.array([i for i in range(64)], dtype=np.float32)
output_data = np.zeros_like(input_data)

# Copy input data to global memory at offset 0
input_offest = 0
gpu.memory.copy_to_global(input_data, offset=input_offest)

# Copy output data to a different location, after input data
output_offest = len(input_data)
gpu.memory.copy_to_global(output_data, offset=output_offest)

# Launch kernel: specify grid dim, block dim
results = gpu.launch_kernel(example_kernel, (2,), (32,), gpu.memory.global_memory, input_offest, output_offest)

# Copy the updated output data back from global memory to the host memory
output_data = gpu.memory.copy_from_global(offset=len(input_data), size=len(output_data))

# Print the results
print(output_data)


# Print profiling information
gpu.profiler.print_summary()
gpu.profiler.visualize_memory_usage()

