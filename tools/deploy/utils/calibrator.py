import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # this automatically init the cuda

import tensorrt as trt


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, calib_loader):
        from tqdm import tqdm
        super(Calibrator, self).__init__()
        self.dataloader = calib_loader
        self.cache_file = cache_file
        self.iter = iter(self.dataloader)
        self.device_input = None
        self.bar = tqdm(iterable=self.dataloader)

    def get_batch_size(self):
        return self.dataloader.batch_size

    def get_batch(self, names):
        try:
            for i in range(6):
                batch = next(self.iter)
                self.bar.update()
            batch = next(self.iter)
            points = np.ascontiguousarray(batch["points"].reshape(self.get_batch_size(), -1, 5)[..., 1:])
            if not self.device_input:
                self.device_input = cuda.mem_alloc(points.nbytes)
            cuda.memcpy_htod(self.device_input, points)
            self.bar.update()
            return [int(self.device_input)]
        except StopIteration:
            self.bar.close()
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        import os
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
