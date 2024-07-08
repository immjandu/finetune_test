import sys, os, glob
import warnings
warnings.simplefilter(action='ignore')
import torch, torchsummaryX #torchsummary
torch.set_printoptions(sci_mode=False)
import argparse
import numpy as np
np.set_printoptions(suppress=True)
import cv2
import PIL

import onnxruntime

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from test import get_model, predict_on_image, show_results

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def arguments():
    parser = argparse.ArgumentParser(description="Export to onnx model / Test with onnx model")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'export'], help='test onnx model or export to onnx model')
    parser.add_argument('--pretrained', type=str, default='weights/yolov7_insect_detection_yolov7_202406050631.pt', help="path to save and load model")
    parser.add_argument('--onnx_path', type=str, default='weights/yolov7_insect_detection_yolov7_202406050631.onnx', help='onnx path')
    parser.add_argument('--data_path', type=str, nargs='+')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args, _ = parser.parse_known_args()
    return args


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_onnx(model, dummy_input, save_onnx_path):
    torch.onnx.export(
        model,
        dummy_input,
        save_onnx_path,
        verbose=False,
        export_params=True,
        opset_version=12,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        dynamic_axes=None
    )


if __name__ == '__main__':
    args = arguments()
    args.model_name = 'yolov7'
    args.onnx_path = f'weights/{args.model_name}.onnx' if args.onnx_path == '' else args.onnx_path
    input_size = (640, 640)
    stride = 32
    
    if args.mode == 'export':
        # model
        model = get_model(args.pretrained)
        stride = int(model.stride.max())  # model stride
        # torchsummary.summary(model, input_size=tuple([3] + list(input_size)), device='cpu')
        torchsummaryX.summary(model, torch.rand((1, 3, 640, 640)))
        # data
        dummy_input = torch.rand(1, 3, 640, 640)
        print(f'Start exporting to onnx model...')
        export_onnx(model=model, dummy_input=dummy_input, save_onnx_path=args.onnx_path)
        print(f'Done! check the outout : {args.onnx_path}')
    else:
        # model
        if not os.path.exists(args.onnx_path):
            print(f'onnx path({args.onnx_path}) Not Exist')
            sys.exit(1)
        model = onnxruntime.InferenceSession(args.onnx_path)

        # data
        img_file_path_list = []
        if (len(args.data_path) == 1):
            if os.path.isdir(args.data_path[-1]):
                for img_file_path in glob.glob(os.path.join(args.data_path[-1], '**'), recursive=True):
                    if img_file_path.endswith('.jpg') or img_file_path.endswith('.jpeg') or img_file_path.endswith('.png'):
                        img_file_path_list.append(img_file_path)
            else:
                img_file_path_list = img_file_path_list + args.data_path
        else:
            img_file_path_list = img_file_path_list + args.data_path

        for img_file_path in img_file_path_list:
            raw_img = cv2.imread(img_file_path)
            raw_img_size = raw_img.shape
            # print(f'>>> (raw) {img_file_path} : {type(raw_img)} / {raw_img.dtype} / {raw_img_size}')
            img = letterbox(raw_img, new_shape=input_size, stride=stride)[0]
            # print(f'>>> (letterbox) {img_file_path} : {type(img)} / {img.dtype} / {img.shape}')
            pred = predict_on_image(img, model, is_onnx=True)
            # print(f'>>> (pred output) {img_file_path} : pred[0] - {pred[0].shape} / pred[1] - {[i.shape for i in pred[1]]}')
            pred = torch.Tensor(pred[0])
            # print(f'>>> (tensor pred) {img_file_path} : {type(pred)} / {pred.dtype} / {pred.shape}')
            
            res = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            # print(f'>>> (after nms) {img_file_path} : {type(res)} / {len(res)} / {type(res[0])} / {res[0].shape}')
            # print(res[0])
            
            # Process detections
            result_saved_path = f'{img_file_path.rsplit(".", 1)[0]}_detected.{img_file_path.rsplit(".", 1)[-1]}'
            for i, det in enumerate(res):  # detections per image
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    det[:, :4] = scale_coords(img.shape, det[:, :4], raw_img_size).round()

                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #     print(c, n, s)

                    for j in range(det.size()[0]):
                        xyxy = det[j, :4].view(-1).tolist()
                        conf = det[j, 4].detach().numpy()
                        class_num = det[j, 5].detach().numpy()

                        raw_img = show_results(raw_img, xyxy, conf, class_num)
                        # if save_img or view_img:  # Add bbox to image
                        #     label = f'{names[int(cls)]} {conf:.2f}'
                        #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            cv2.imwrite(result_saved_path, raw_img)