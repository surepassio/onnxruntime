import argparse
import onnxruntime as ort
import os
import re
import sys

from ort_test_dir_utils import run_test_dir


def run(model, test_only):

    if not test_only:
        onnx_target_path = re.sub('.onnx$', '.optimized.onnx', model)
        ort_target_path = re.sub('.onnx$', '.ort', model)
        final_model = ort_target_path

        # First, convert to optimized ONNX. This is used as input for op reduction in a minimal build.
        # TEMP: Only do this if it doesn't exist. We're not changing the ONNX optimization so once we've created the
        # file it shouldn't change with new builds.
        if not os.path.exists(onnx_target_path):
            so = ort.SessionOptions()
            so.optimized_model_filepath = onnx_target_path
            so.add_session_config_entry('session.save_model_format', 'ONNX')
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # Skip NCHWc optimizations

            print(f"Optimizing {model} and saving in ONNX format to {onnx_target_path}")
            _ = ort.InferenceSession(model, sess_options=so)

            # Validate optimized ONNX model
            print(f"Running tests for {onnx_target_path}")
            run_test_dir(onnx_target_path)

        # Second, convert optimized ONNX model to ORT format
        so = ort.SessionOptions()
        so.optimized_model_filepath = ort_target_path
        so.add_session_config_entry('session.save_model_format', 'ORT')

        print(f"Converting optimized ONNX model {onnx_target_path} to ORT format model {ort_target_path}")
        _ = ort.InferenceSession(onnx_target_path, sess_options=so)

        orig_size = os.path.getsize(onnx_target_path)
        new_size = os.path.getsize(ort_target_path)
        print(f"Serialized {onnx_target_path} to {ort_target_path}. Sizes: orig={orig_size} "
              f"new={new_size} Diff={new_size-orig_size} Ratio={new_size / orig_size:.4f}%")

    else:
        final_model = model

    # Validate
    print(f"Running tests for {final_model}")
    if not os.path.exists(final_model):
        print(f"{final_model} not found.")
        sys.exit(-1)

    run_test_dir(final_model)


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Convert an onnx model -> optimized onnx model -> ORT format model. '
                                     'Model should be in onnx_test_runner format directory for validation. ')
    parser.add_argument('-t', '--test_only', required=False, action='store_true',
                        help='Test and validate ORT model. No model conversion. Use to validate minimal build.')
    parser.add_argument('-m', '--model', required=False,
                        help='Provide single model to test. Should be onnx model in onnx_test_runner format directory.')
    return parser.parse_args()


def main():
    args = parse_args()

    test_models = [
        r'D:\src\github\ORT test models\20190729\opset10\yolov3\yolov3.onnx',
        r'D:\src\github\ORT test models\20190729\opset10\tf_mobilenet_v2_1.4_224\model.onnx',
        r'D:\src\github\ORT test models\20190729\opset10\mlperf_ssd_mobilenet_300\ssd_mobilenet_v1_coco_2018_01_28.onnx',  # noqa
    ]

    if args.model:
        test_models = [args.model]

    for model in test_models:
        if args.test_only and not args.model:
            model = re.sub('.onnx$', '.ort', model)

        run(model, args.test_only)


if __name__ == '__main__':
    main()
