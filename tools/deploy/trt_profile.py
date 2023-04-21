"""
run:
nsys profile -o profile --capture-range cudaProfilerApi python tools/deploy/trt_profile.py --engine tools/models/trt/iassd_hvcsx2_4x8_80e_kitti_3cls\(export\).engine --batch 1
or
python tools/deploy/trt_profile.py --engine tools/models/trt/iassd_hvcsx2_4x8_80e_kitti_3cls\(export\).engine --batch 1 --build_in
"""
from rd3d.api import demo


def evaluate(engine_file, dataloader, use_build_in, num_points):
    import torch
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt
    import pycuda.autoinit
    from utils import load_plugins
    from utils.profiler import MyProfiler

    logger = trt.Logger(trt.Logger.ERROR)
    # create engine
    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    bs = dataloader.batch_size
    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        context.set_binding_shape(engine.get_binding_index("points"), (bs, num_points, 4))

        assert context.all_binding_shapes_specified
        batch_dict = next(iter(dataloader))
        dataloader.dataset.load_data_to_gpu(batch_dict)

        h_inputs = {'points': np.zeros((bs, num_points, 4), dtype=float)}
        d_inputs = {}
        h_outputs = {}
        d_outputs = {}
        t_outputs = {}
        for binding in engine:
            if engine.binding_is_input(binding):
                d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
            else:
                size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)

        h_inputs = {'points': batch_dict['points'].view(bs, -1, 5)[..., 1:].contiguous().cpu().numpy()}

        def infer():
            for key in h_inputs:
                cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
            context.execute_async_v2(
                bindings=[int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs],
                stream_handle=stream.handle)
            for key in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
            stream.synchronize()

        infer()  # warmup
        if use_build_in:
            context.profiler = MyProfiler(["HAVSampling", "BallQuery", "ForeignNode", "NMSBEV"])
        cuda.start_profiler()
        for i in range(1000 if use_build_in else 2):
            infer()
        if use_build_in:
            context.report_to_profiler()
            context.profiler.print()
        cuda.stop_profiler()
        print("done")


def main():
    import pickle
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--build_in', action='store_true', default=False)
    parser.add_argument('--points', type=int, default=16384)
    _, dataloader, args = demo(parser)

    use_build_in = args.build_in
    engine_file = args.engine

    evaluate(engine_file, dataloader, use_build_in, args.points)


if __name__ == "__main__":
    main()
