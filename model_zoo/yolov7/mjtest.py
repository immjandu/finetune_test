base_path='/Users/mj/dxtest/model_zoo/yolov7'
import os; os.chdir(base_path)
import sys; sys.path.append(base_path)
from test_pt import *
from test_onnx import export_onnx
import onnxruntime
torch.set_printoptions(sci_mode=False)
input_size = (640, 640)

img_file_path = os.path.join(base_path, 'data/images/insect.jpg')
raw_img = cv2.imread(img_file_path)
raw_img_size = raw_img.shape
# img = cv2.resize(raw_img, input_size)

# pt model
pretrained = os.path.join(base_path, 'weights/yolov7_insect_detection_202406171141.pt')
model = get_model(pretrained)
stride = int(model.stride.max())  # model stride
img = letterbox(raw_img, new_shape=input_size, stride=stride)[0]
pred = predict_on_image(img, model)
print(len(pred))
print(type(pred[0]), type(pred[1]))
print(pred[0].shape, pred[1][0].shape, pred[1][1].shape, pred[1][2].shape)
# torch.Size([1, 25200, 6]) torch.Size([1, 3, 80, 80, 6]) torch.Size([1, 3, 40, 40, 6]) torch.Size([1, 3, 20, 20, 6])

def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

stride_list = model.stride
z = []  # inference output
nl = 3 # number of detection layers
nc = 1 # number of classes
no = nc + 5 # number of outputs per anchor
grid = [torch.zeros(1)] * nl
anchors = [
    [12,16, 19,36, 40,28],
    [36,75, 76,55, 72,146],
    [142,110, 192,243, 459,401]
]
anchors = torch.tensor(anchors).float().view(nl, -1, 2) # torch.Size([3, 3, 2])
anchor_grid = anchors.view(nl, 1, -1, 1, 1, 2) # torch.Size([3, 1, 3, 1, 1, 2])
x = pred[1]
results = []
for i in range(nl):
    bs, ch, ny, nx, _ = x[i].shape
    if grid[i].shape[2:4] != x[i].shape[2:4]:
        grid[i] = make_grid(nx, ny).to(device)
    y = x[i].sigmoid()
    if not torch.onnx.is_in_onnx_export():
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride_list[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
    else:
        xy, wh, conf = y.split((2, 2, nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
        xy = xy * (2. * stride_list[i]) + (stride_list[i] * (grid[i] - 0.5))  # new xy
        wh = wh ** 2 * (4 * anchor_grid[i].data)  # new wh
        y = torch.cat((xy, wh, conf), 4)
    results.append(y.view(bs, -1, no))
print([i.shape for i in results])
results = torch.cat(results, 1)
results.shape


# transform to onnx model
dummy_input = torch.rand(1, 3, 640, 640)
export_onnx(model=model, dummy_input=dummy_input, save_onnx_path=args.onnx_path)

# onnx model
pretrained = os.path.join(base_path, 'weights/yolov7_insect_detection_202406171141.onnx')
model = onnxruntime.InferenceSession(pretrained)
stride=32
img = letterbox(raw_img, new_shape=input_size, stride=stride)[0]
pred = predict_on_image(img, model, is_onnx=True)
print(len(pred))
print(type(pred[0]), type(pred[1]), type(pred[2]), type(pred[3]))
print(pred[0].shape, pred[1].shape, pred[2].shape, pred[3].shape)