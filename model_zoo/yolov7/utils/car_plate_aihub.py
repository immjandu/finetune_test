import sys, os, glob, shutil
import argparse
import json
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch
import cv2
import skimage

'''
def xyxy2xywh(old):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    new = old.clone().float() if isinstance(old, torch.Tensor) else np.copy(old).astype(float)
    new[:, 2] = old[:, 2] - old[:, 0] # weight
    new[:, 3]  = old[:, 3] - old[:, 1] # height
    new[:, 0] = old[:, 0] + (new[:, 2] / 2) # x center
    new[:, 1] = old[:, 1] + (new[:, 3] / 2) # y center
    return new
'''

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

'''
def xywh2xywhn(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to normalized [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone().float() if isinstance(x, torch.Tensor) else np.copy(x).astype(float)
    y[:, 0] = ((x[:, 0] + x[:, 2]/2) / w)  # xmin
    y[:, 1] = (x[:, 1] + x[:, 3]/2) / h  # ymin
    y[:, 2] = x[:, 2] / w  # width
    y[:, 3] = x[:, 3] / h  # height
    return y
'''
def xywh2xywhn(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to normalized [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone().float() if isinstance(x, torch.Tensor) else np.copy(x).astype(float)
    y[:, 0] = x[:, 0] / w  # xmin
    y[:, 1] = x[:, 1] / h  # ymin
    y[:, 2] = x[:, 2] / w  # width
    y[:, 3] = x[:, 3] / h  # height
    return y


def get_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', nargs='+', choices=['mapping','txtfiles','split'])
    parser.add_argument('--root_data_path', type=str, default='/home/ubuntu/nfs-mount/aihub_carplate')
    parser.add_argument('--img2car_plate_path', type=str, default='result_img2car_plate.pkl')
    parser.add_argument('--txt_file_dir', type=str, default='./label_txt')
    parser.add_argument('--split_root_dir', type=str, default=None)
    parser.add_argument('--train_split_ratio', type=float, default=0.8)
    parser.add_argument('--xywh2xywhn', action='store_true')
    parser.add_argument('--sample_cnt', type=int, default=-1)
    args = parser.parse_args()

    return args


def get_metainfo(json_path):
    img2car_plate = {}
    
    metainfo = json.load(open(json_path))
    if 'imagePath' in metainfo:
        value = {}
        for k in ['car', 'plate']:
            if k in metainfo:
                value[k] = sum(metainfo[k]['bbox'], [])
        if len(value) > 0:
            img2car_plate[metainfo['imagePath']] = [value]
        else:
            print(f'[WARN] No "car" and "plate" in metainfo {json_path}')
    else:
        print(f'[WARN] "imagePath" is not included in metainfo {json_path}')
    
    return img2car_plate


def check_data_dir(root_data_path):
    ## root_data_path
    ## - images
    ##      a.jpg
    ##      b.jpg
    ##      c.jpg
    ##      ...
    ##
    ## - labels
    ## ---- class1
    ## ------- class1_1
    ## ---------- class1_1_a.json
    ## ---------- class1_1_b.json
    ## ---------- class1_1_c.json
    ##            ...
    ## ------- class1_2
    ## ---------- class2_1_a.json
    ## ---------- class2_1_b.json
    ## ---------- class2_1_c.json
    ##            ...
    ##         ...
    obj_list = os.listdir(root_data_path)
    for dir_name in ['images', 'labels']:
        dir_path = os.path.join(root_data_path, dir_name)
        if not (dir_name in obj_list):
            raise Exception(f'[ERROR] "{dir_name}" directory NOT exist under {root_data_path}!')
        if not os.path.isdir(os.path.join(root_data_path, dir_name)):
            raise Exception(f'[ERROR] "{dir_name}" under {root_data_path} is NOT directory!')
        obj_ofdir_list = os.listdir(dir_path)
        if not (len(obj_ofdir_list) > 1):
            raise Exception(f'[ERROR] "{dir_path}" is Empty!')
    
        if dir_name == 'labels':
            if not any([os.path.isdir(os.path.join(dir_path, i)) for i in obj_ofdir_list]):
                raise Exception(f'[ERROR] "{dir_path}" has NO classes(labels)!')


if __name__ == '__main__':
    args = get_arguments()
    classes_dict = {'car':0, 'plate':1}

    check_data_dir(args.root_data_path)

    if 'mapping' in args.task:
        # map "raw image" with "json label&bbox info" which created by car and plate
        
        if os.path.exists(args.img2car_plate_path):
            print(f'[WARN] {args.img2car_plate_path} Already Exist!')
        else:
            result_img2car_plate = {}
            curr_cnt = 0
            for f in tqdm(glob.glob(os.path.join(args.root_data_path, 'labels', '**'), recursive=True)):
                if f.endswith('.json'):
                    img2car_plate = get_metainfo(f)
                    if len(img2car_plate) == 1:
                        img_file_name, values = list(img2car_plate.items())[-1]
                        img_file_path = os.path.join(args.root_data_path, 'images', img_file_name)
                        try:
                            img = skimage.io.imread(img_file_path)
                        except Exception as e:
                            print(f'[WARN] {img_file_path} Corrupted!!! : {e}')
                            continue
                        if os.path.exists(img_file_path) and (img is not None):
                            if args.xywh2xywhn:
                                img_h, img_w = list(img.shape)[:2]
                                new_values = {}
                                for class_key, bbox_xyxy in values[0].items():
                                    xywhn = xywh2xywhn(xyxy2xywh(np.asarray([bbox_xyxy])), w=img_w, h=img_h)
                                    if not (xywhn<=1).all():
                                        raise Exception(f'[ERROR] {img_file_path} : xywhn wrong {bbox_xyxy} -> {xywhn} ({img_w}x{img_h})')
                                    elif (xywhn<=0).any():
                                        raise Exception(f'[ERROR] {img_file_path} : xywhn wrong {bbox_xyxy} -> {xywhn} ({img_w}x{img_h})')
                                    else:
                                        new_values[class_key] = xywhn
                                values = [new_values]
                            
                            if img_file_path in result_img2car_plate:
                                # print(f'{img_file_path} is already exist!')
                                result_img2car_plate[img_file_path].extend(values)
                            else:
                                img2car_plate = {img_file_path: values}
                                result_img2car_plate.update(img2car_plate)
                            
                            if (args.sample_cnt > 0):
                                curr_cnt += 1
                                if (args.sample_cnt <= curr_cnt):
                                    break
                            
                        else:
                            print(f'[ERROR] {img_file_path} Not Exist or Corrupted!!!')
                            # sys.exit(1)
            pickle.dump(result_img2car_plate, open(args.img2car_plate_path,'wb'))
        
        print('DONE!!! map "raw image" with "json label&bbox info" which created by car and plate.')
    
    if 'txtfiles' in args.task:
        # write "label&bbox info" to txt file by "raw image"
        if os.path.exists(args.img2car_plate_path):
            result_img2car_plate = pickle.load(open(args.img2car_plate_path, 'rb'))
            os.makedirs(args.txt_file_dir, exist_ok=True)
            for img_file_path, value_list in tqdm(result_img2car_plate.items()):
                img_file_name = img_file_path.rsplit('/', 1)[-1]
                txt_file_path = os.path.join(args.txt_file_dir, f'{img_file_name.rsplit(".", 1)[0]}.txt')
                tmp_str_list = []
                for v_dict in value_list:
                    for c, idx in classes_dict.items():
                        if c in v_dict:
                            a_line = list(map(str, [idx] + list(v_dict[c].squeeze())))
                            tmp_str_list.append(' '.join(a_line))
                with open(txt_file_path, 'w') as f:
                    f.write('\n'.join(tmp_str_list))
        else:
            raise Exception(f'[ERROR] {args.img2car_plate_path} Not Exist!')
        print('DONE!!! write "label&bbox info" to txt file by "raw image".')
    
    if 'split' in args.task:
        # split data to train, valid and test directory
        result_img2car_plate = pickle.load(open(args.img2car_plate_path, 'rb'))
        if args.split_root_dir is not None:
            total_image_file_name = list(result_img2car_plate.keys())
            total_cnt = len(total_image_file_name)
            splitted = {}
            # train
            train_cnt = int(total_cnt * args.train_split_ratio)
            splitted['train'] = random.sample(total_image_file_name, k=train_cnt)
            rest_image_file_name = list(set(total_image_file_name) - set(splitted['train']))
            # valid
            valid_cnt = int(len(rest_image_file_name) / 2)
            splitted['valid'] = random.sample(rest_image_file_name, k=valid_cnt)
            # test
            splitted['test'] = list(set(rest_image_file_name) - set(splitted['valid']))
            
            print(f'Splitted(Total={total_cnt}) : train={len(splitted["train"])}, valid={len(splitted["valid"])}, test={len(splitted["test"])}')
            
            # copy files
            for tp in ['train', 'valid', 'test']:
                print(f'Start copy image and label files to {args.split_root_dir} [{tp.upper()}]')
                splitted_dir_path = os.path.join(args.split_root_dir, tp)
                os.makedirs(os.path.join(splitted_dir_path, 'images'), exist_ok=True)
                os.makedirs(os.path.join(splitted_dir_path, 'labels'), exist_ok=True)
                for src_img_file_path in tqdm(splitted[tp]):
                    image_file_name = src_img_file_path.rsplit('/', 1)[-1]
                    src_label_file_path = os.path.join(args.txt_file_dir, f'{image_file_name.rsplit(".", 1)[0]}.txt')
                    
                    if os.path.exists(src_img_file_path) and os.path.exists(src_label_file_path):
                        shutil.copyfile(
                            src_img_file_path,
                            os.path.join(splitted_dir_path, 'images', image_file_name)
                        )
                        shutil.copyfile(
                            src_label_file_path,
                            os.path.join(splitted_dir_path, 'labels', f'{image_file_name.rsplit(".", 1)[0]}.txt')
                        )
                    else:
                        print(f'[WARN] FILE EXIST : {src_img_file_path}={os.path.exists(src_img_file_path)} / {src_label_file_path}={os.path.exists(src_label_file_path)}')
        
        print('DONE!!! split data to train, valid and test directory.')