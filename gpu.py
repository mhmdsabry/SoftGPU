from compute import StreamingMultiprocessor, GPUWarpScheduler, Warp
from memory import GPUMemory
from profiler import GPUProfiler
import numpy as np
import time


class GPU:
    def __init__(self, num_sms, threads_per_warp, warps_per_sm, global_memory_size, shared_memory_size):
        self.profiler = GPUProfiler()
        self.num_sms = num_sms
        self.threads_per_warp = threads_per_warp
        self.warps_per_sm = warps_per_sm
        self.memory = GPUMemory(global_memory_size, shared_memory_size, profiler=self.profiler)
        self.sms = [StreamingMultiprocessor(sm_id) for sm_id in range(num_sms)]
        

    def allocate_shared_memory(self, size):
        if size > self.memory.shared_size:
            raise MemoryError("Requested shared memory size exceeds GPU capacity.")
        self.memory.allocate_shared_memory(size)

    def release_shared_memory(self):
        self.memory.release_shared_memory()

    def launch_kernel(self, kernel_function, grid_dim, block_dim, *kernel_args):
        total_blocks = np.prod(grid_dim)
        threads_per_block = np.prod(block_dim)
        total_warps = total_blocks * threads_per_block // self.threads_per_warp

        # Log timeline start
        start_time = time.time()

        # Schedule warps
        scheduler = GPUWarpScheduler(total_warps, self.threads_per_warp, kernel_function, *kernel_args)

        self.profiler.start_profiling()

        # Distribute warps to SMs
        for sm_id, sm in enumerate(self.sms):  
            for _ in range(self.warps_per_sm):  
                if scheduler.warps:
                    warp = scheduler.warps.pop(0)
                    sm.add_warp(warp)


        # Execute all SMs and record results
        results = []
        for sm_id, sm in enumerate(self.sms):
            sm_results = sm.execute_warps()
            results.extend(sm_results)

        # Timeline end and Log kernel execution
        end_time = time.time()
        duration = end_time - start_time
        self.profiler.log_kernel_execution(kernel_function.__name__, grid_dim, block_dim, duration)

        return results

    
if __name__ == "__main__":
    # Kernel Example
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

