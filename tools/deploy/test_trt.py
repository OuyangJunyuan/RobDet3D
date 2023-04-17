from rd3d.api import demo


def evaluate(engine_file, dataloader):
    import torch
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt
    import pycuda.autoinit
    from tqdm import tqdm
    from utils import load_plugins

    logger = trt.Logger(trt.Logger.ERROR)
    # create engine
    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    pred_labels = []
    bs = dataloader.batch_size
    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        context.set_binding_shape(engine.get_binding_index("points"), (bs, 16384, 4))
        assert context.all_binding_shapes_specified

        h_inputs = {'points': np.zeros((bs, 16384, 4), dtype=float)}
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

        for batch_dict in tqdm(iterable=dataloader):
            dataloader.dataset.load_data_to_gpu(batch_dict)
            if bs != batch_dict['batch_size']:
                bs = batch_dict['batch_size']
                context.set_binding_shape(engine.get_binding_index("points"), (bs, 16384, 4))
                assert context.all_binding_shapes_specified

                h_inputs = {'points': np.zeros((bs, 16384, 4), dtype=float)}
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
            for key in h_inputs:
                cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
            context.execute_async_v2(
                bindings=[int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs],
                stream_handle=stream.handle)
            for key in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
            stream.synchronize()

            output_nums = torch.from_numpy(h_outputs['nums']).cuda()
            output_boxes = torch.from_numpy(h_outputs['boxes'].reshape((bs, 256, -1))).cuda()
            output_scores = torch.from_numpy(h_outputs['scores'].reshape((bs, 256))).cuda()

            final_output_dicts = [{'pred_boxes': output_boxes[i, :output_nums[i], :7],
                                   'pred_labels': output_boxes[i, :output_nums[i], -1].int(),
                                   'pred_scores': output_scores[i, :output_nums[i]]}
                                  for i in range(bs)]

            pred_labels.extend(dataloader.dataset.generate_prediction_dicts(batch_dict, final_output_dicts,
                                                                            dataloader.dataset.class_names))
    return pred_labels


def main():
    import pickle
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--cache', type=Path, default=Path("tools/experiments2/data/eval.pkl"))
    _, dataloader, args = demo(parser)

    engine_file = args.engine
    eval_file = args.cache

    if eval_file.exists():
        pred_labels = pickle.load(open(eval_file, 'br'))
    else:
        pred_labels = evaluate(engine_file, dataloader)
        pickle.dump(pred_labels, open(eval_file, 'wb'))

    result_str, result_dict = dataloader.dataset.evaluation(pred_labels, dataloader.dataset.class_names)
    print(result_str)


if __name__ == "__main__":
    main()
