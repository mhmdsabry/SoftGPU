import time
import numpy as np
import matplotlib.pyplot as plt
import os

class GPUProfiler:
    def __init__(self):
        self.kernel_traces = []
        self.memory_traces = []
        self.start_time = None

        # profile_logs directory
        self.log_dir = "profile_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def start_profiling(self):
        self.start_time = time.time()

    def log_kernel_execution(self, kernel_name, grid_dim, block_dim, duration):
        self.kernel_traces.append({
            "kernel_name": kernel_name,
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "duration": duration,
        })

    def log_memory_operation(self, op_type, address, size, time_stamp=None):
        if time_stamp is None:
            time_stamp = time.time() - self.start_time
        self.memory_traces.append({
            "op_type": op_type,
            "address": address,
            "size": size,
            "time_stamp": time_stamp,
        })

    def visualize_memory_usage(self):
        read_times = [entry["time_stamp"] for entry in self.memory_traces if entry["op_type"] == "read"]
        write_times = [entry["time_stamp"] for entry in self.memory_traces if entry["op_type"] == "write"]
        read_sizes = [entry["size"] for entry in self.memory_traces if entry["op_type"] == "read"]
        write_sizes = [entry["size"] for entry in self.memory_traces if entry["op_type"] == "write"]

        plt.figure(figsize=(10, 5))
        plt.scatter(read_times, read_sizes, color="blue", label="Reads", alpha=0.6)
        plt.scatter(write_times, write_sizes, color="red", label="Writes", alpha=0.6)
        plt.title("Memory Operations Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Size (bytes)")
        plt.legend()
        plt.grid()

        # Save the plot to the profile_logs folder
        save_path = os.path.join(self.log_dir, "memory_usage.png")
        plt.savefig(save_path)
        print(f"Memory usage visualization saved to {save_path}")
        plt.close()

    def print_summary(self):
        total_duration = time.time() - self.start_time
        print(f"Total profiling duration: {total_duration:.4f} seconds")
        print("Kernel Execution Summary:")
        for trace in self.kernel_traces:
            print(f"- Kernel: {trace['kernel_name']}, Grid: {trace['grid_dim']}, Block: {trace['block_dim']}, "
                  f"Duration: {trace['duration']:.4f} seconds")
        print(f"Total memory operations: {len(self.memory_traces)}")
