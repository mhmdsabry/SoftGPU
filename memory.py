import time
import numpy as np

class GPUMemory:
    def __init__(self, global_size, shared_size, profiler=None):
        self.global_memory = np.zeros(global_size, dtype=np.float32)
        self.shared_memory = None
        self.shared_size = shared_size
        self.profiler = profiler  # profiler for logging (optional)

    def allocate_shared_memory(self, size):
        if size > self.shared_size:
            raise MemoryError("Shared memory size exceeded!")
        self.shared_memory = np.zeros(size, dtype=np.float32)
        if self.profiler:
            self.profiler.log_memory_operation(
                op_type="allocate",
                address="shared_memory",
                size=size * self.shared_memory.itemsize,
                time_stamp=time.time()
            )

    def release_shared_memory(self):
        if self.shared_memory is not None and self.profiler:
            size = len(self.shared_memory) * self.shared_memory.itemsize
            self.profiler.log_memory_operation(
                op_type="release",
                address="shared_memory",
                size=size,
                time_stamp=time.time()
            )
        self.shared_memory = None

    def _simulate_memory_access(self, memory_type, size):
        if memory_type == "global":
            # Simulate slower global memory access (delay)
            time.sleep(size * 0.00001)
        elif memory_type == "shared":
            # Simulate faster shared memory access (minimal than global memory)
            time.sleep(size * 0.000001)
    
    def copy_to_global(self, data, offset=0):
        if offset + len(data) > len(self.global_memory):
            raise MemoryError("Global memory access out of bounds!")
        
        # Simulate global memory write time
        self._simulate_memory_access("global", len(data))
        
        self.global_memory[offset:offset+len(data)] = data
        if self.profiler:
            self.profiler.log_memory_operation(
                op_type="write",
                address=offset,
                size=len(data) * self.global_memory.itemsize,
                time_stamp=time.time()
            )

    def copy_from_global(self, offset, size):
        if offset + size > len(self.global_memory):
            raise MemoryError("Global memory access out of bounds!")
        
        self._simulate_memory_access("global", size)
        
        result = self.global_memory[offset:offset+size]
        if self.profiler:
            self.profiler.log_memory_operation(
                op_type="read",
                address=offset,
                size=size * self.global_memory.itemsize,
                time_stamp=time.time()
            )
        return result

    def copy_to_shared(self, data, offset=0):
        if self.shared_memory is None:
            raise MemoryError("Shared memory is not allocated!")
        
        self._simulate_memory_access("shared", len(data))
        
        self.shared_memory[offset:offset+len(data)] = data
        if self.profiler:
            self.profiler.log_memory_operation(
                op_type="write",
                address=offset,
                size=len(data) * self.shared_memory.itemsize,
                time_stamp=time.time()
            )

    def copy_from_shared(self, offset, size):
        if self.shared_memory is None:
            raise MemoryError("Shared memory is not allocated!")
        
        
        self._simulate_memory_access("shared", size)
        
        result = self.shared_memory[offset:offset+size]
        if self.profiler:
            self.profiler.log_memory_operation(
                op_type="read",
                address=offset,
                size=size * self.shared_memory.itemsize,
                time_stamp=time.time()
            )
        return result