import torch


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def concat_detections(
        x: list,
        model_stride_list:list,
        nl=3,  # number of detection layers
        nc=1, # number of classes
        anchors=None, device=None
):
    assert len(x) == 3

    device = torch.device('cpu') if device is None else device
    no = nc + 5  # number of outputs per anchor
    grid = [torch.zeros(1)] * nl
    anchors = [
        [12, 16, 19, 36, 40, 28],
        [36, 75, 76, 55, 72, 146],
        [142, 110, 192, 243, 459, 401]
    ] if anchors is None else anchors
    anchors = torch.tensor(anchors).float().view(nl, -1, 2)  # torch.Size([3, 3, 2])
    anchor_grid = anchors.view(nl, 1, -1, 1, 1, 2)  # torch.Size([3, 1, 3, 1, 1, 2])

    results = []
    for i in range(nl):
        bs, ch, ny, nx, _ = x[i].shape
        if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = make_grid(nx, ny).to(device)
        y = x[i].sigmoid()
        if not torch.onnx.is_in_onnx_export():
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * model_stride_list[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        else:
            xy, wh, conf = y.split((2, 2, nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            xy = xy * (2. * model_stride_list[i]) + (model_stride_list[i] * (grid[i] - 0.5))  # new xy
            wh = wh ** 2 * (4 * anchor_grid[i].data)  # new wh
            y = torch.cat((xy, wh, conf), 4)
        results.append(y.view(bs, -1, no))
    print(f'concat results : {[i.shape for i in results]} -> ', end='')
    results = torch.cat(results, 1)
    print(results.shape)
    return results


