import time
import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
from rd3d.api import quick_demo
import ctypes
from utils import load_plugins
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--engine")
print([pc.name for pc in trt.get_plugin_registry().plugin_creator_list])
model, dataloader, cfgs, args = quick_demo(parser)
model.cuda()
model.eval()

frame_id = [0, 1, 2, 3, 4, 5, 6, 7]
bs = len(frame_id)
batch_dict = dataloader.dataset.collate_batch([dataloader.dataset[fid] for fid in frame_id])
dataloader.dataset.load_data_to_gpu(batch_dict)
points = batch_dict['points'].view(bs, -1, 5)[..., 1:].contiguous().cpu().numpy()
print(points.shape)
num_points = points.shape[1]


def trt_inf():
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    # create engine
    with open(args.engine, "rb") as f, trt.Runtime(
            logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    h_inputs = {'points': points}
    d_inputs = {}
    h_outputs = {}
    d_outputs = {}
    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        context.set_optimization_profile_async(0, stream.handle)
        context.set_binding_shape(engine.get_binding_index("points"), (bs, num_points, 4))
        assert context.all_binding_shapes_specified

        for binding in engine:
            if engine.binding_is_input(binding):
                d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
            else:
                output_shape = context.get_binding_shape(engine.get_binding_index(binding))
                size = trt.volume(output_shape)
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)
                # print(f"{binding}: shape({output_shape}), dtype({dtype}), dbytes({h_outputs[binding].nbytes})")

        def infer():
            for key in h_inputs:
                cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
            context.execute_async_v2(
                bindings=[int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs],
                stream_handle=stream.handle)
            for key in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
            stream.synchronize()

        for _ in range(1000):
            t1 = time.time()
            infer()
            t2 = time.time()
            print(1000 * (t2 - t1))

        return h_outputs['boxes'].reshape(bs, 256, -1), \
            h_outputs['scores'].reshape(bs, 256, -1)


# trt_inf()
r1, r2 = trt_inf()
# print(t1.shape, t1.dtype, '\n', t1)
# print(t2.shape, t2.dtype, '\n', t2)
# print(r1.shape, r1.dtype, '\n', r1)
# print(r2.shape, r2.dtype, '\n', r2)

# import matplotlib.pyplot as plt
#
# plt.subplot(1, 2, 1)
# norm = np.linalg.norm(t1, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# norm = np.linalg.norm(r1, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# plt.subplot(1, 2, 2)
# norm = np.linalg.norm(t2, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# norm = np.linalg.norm(r2, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# plt.show()
#

from rd3d.utils.viz_utils import viz_scenes

xyz = points[-1]
box = r1[-1].reshape((-1, 8))[..., :7]
viz_scenes(
    (xyz, box),
)
