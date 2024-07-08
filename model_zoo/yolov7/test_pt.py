import argparse
import os, glob
import cv2
import numpy as np
import torch
# import torchsummary
import torchsummaryX

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

device = torch.device('cpu')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data_path', type=str, nargs='+')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()

    return args


def get_model(pretrained):
    ckpt = torch.load(pretrained, map_location=device)
    model = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model = model.float().fuse().eval()
    # model = model.float().eval()

    return model


def fixed_image_standardization(input_tensor):
    processed_tensor = (input_tensor - 0) / 255.0
    # processed_tensor = (input_tensor / 255.0 - 0.0) / 1.0 # same as above
    return processed_tensor


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict_on_image(input, model, verbose_flag=True, is_onnx=False):
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input).permute((2, 0, 1))
    input = input.unsqueeze(0)
    if verbose_flag:
        print(f'>>> (arrange to tensor) : {type(input)} / {input.dtype} / {input.shape}')

    input = input.to(device)
    input = fixed_image_standardization(input)
    if verbose_flag:
        print(f'>>> (after std of predict) : {type(input)} / {input.dtype} / {input.shape}')
    
    if is_onnx:
        input = to_numpy(input)
        input = {model.get_inputs()[0].name: input}
        output = model.run(None, input)
    else:
        output = model(input)

    return output


def show_results(img, xyxy, conf, class_num=None, landmarks=None):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    if landmarks is not None:
        for i in range(5):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = f'{int(class_num)}({str(conf)[:5]})'
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    args = get_arguments()
    args.input_size = (640, 640)

    # model
    model = get_model(args.pretrained)
    # torchsummary.summary(model, input_size=tuple([3] + list(args.input_size)), device='cpu')
    torchsummaryX.summary(model, torch.rand((1, 3, 640, 640)))
    stride = int(model.stride.max())  # model stride
    # tmp = model(torch.rand((1, 3, 640, 640)))

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
        print(f'>>> (raw) {img_file_path} : {type(raw_img)} / {raw_img.dtype} / {raw_img_size}')

        img = letterbox(raw_img, new_shape=args.input_size, stride=stride)[0]
        print(f'>>> (letterbox) {img_file_path} : {type(img)} / {img.dtype} / {img.shape}')
        pred = predict_on_image(img, model)
        print(f'>>> (pred output) {img_file_path} : pred[0] - {pred[0].shape} / pred[1] - {[i.shape for i in pred[1]]}')
        pred = torch.Tensor(pred[0]) # or use utils.postprocess.concat_detections(pred[1], model.stride)
        print(f'>>> (tensor pred) {img_file_path} : {type(pred)} / {pred.dtype} / {pred.shape}')

        res = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
        print(f'>>> (after nms) {img_file_path} : {type(res)} / {len(res)} / {type(res[0])} / {res[0].shape}')
        print(res[0])

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