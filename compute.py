import threading
from concurrent.futures import ThreadPoolExecutor


class Warp:
    def __init__(self, warp_id, threads_per_warp, kernel_function, *kernel_args):
        self.warp_id = warp_id
        self.threads_per_warp = threads_per_warp
        self.kernel_function = kernel_function
        self.kernel_args = kernel_args
        self.results = [None] * threads_per_warp
        self.active_mask = [True] * threads_per_warp  # Track active threads (for divergence)

    def execute(self):
        threads = []
        for thread_id in range(self.threads_per_warp):
            thread = threading.Thread(
                target=self._execute_thread,
                args=(thread_id,)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()  # Ensure all threads complete

    def _execute_thread(self, thread_id):
        if self.active_mask[thread_id]:  # Only execute if the thread is active
            self.results[thread_id] = self.kernel_function(
                self.warp_id, thread_id, *self.kernel_args
            )


class GPUWarpScheduler:
    def __init__(self, num_warps, num_threads_per_warp, kernel_function, *kernel_args):
        self.num_warps = num_warps
        self.num_threads_per_warp = num_threads_per_warp
        self.kernel_function = kernel_function
        self.kernel_args = kernel_args
        self.warps = [
            Warp(warp_id, num_threads_per_warp, kernel_function, *kernel_args)
            for warp_id in range(num_warps)
        ]

    def execute(self):
        with ThreadPoolExecutor(max_workers=self.num_warps) as executor:
            futures = [executor.submit(warp.execute) for warp in self.warps]
            for future in futures:
                future.result()


class StreamingMultiprocessor:
    def __init__(self, sm_id):
        self.sm_id = sm_id
        self.warp_queue = []

    def add_warp(self, warp):
        self.warp_queue.append(warp)

    def execute_warps(self):
        results = []
        for warp in self.warp_queue:
            warp.execute()
            results.extend(warp.results)
        return results


def sync_threads_barrier(shared_barrier, thread_id):
    print(f"Thread {thread_id} waiting at barrier.")
    shared_barrier.wait()
    print(f"Thread {thread_id} passed the barrier.")



