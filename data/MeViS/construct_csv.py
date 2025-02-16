###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import json
import numpy as np
import random
from tqdm import tqdm
import cv2
import csv
from pycocotools import mask as coco_mask
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, img_folder: Path, ann_file: Path, transforms):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        # create video meta data
        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder) + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            # vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            vid, exp, anno_id, frames, exp_id = \
                meta['video'], meta['exp'], meta['anno_id'], meta['frames'], meta['exp_id']
            
            # Color used for mask visualization
            color_dict = {x: [random.randint(0, 255) for _ in range(3)] for x in anno_id}

            # clean up the caption
            exp = " ".join(exp.lower().split())
            vid_len = len(frames)

            # read frames and masks
            boxes, masks, valid = [], [], []
            for j in tqdm(range(vid_len), desc="Processing frames"):
                frame_indx = j
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', vid, frame_name + '.jpg')
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                mask = np.zeros((h, w, 3), dtype=np.uint8)  # H * W * 3
                frame_boxes = []
                for x in anno_id:
                    frm_anno = self.mask_dict[x][j]
                    if frm_anno is not None:
                        decoded_mask = coco_mask.decode(frm_anno)  # H * W, 二值mask
                        mask[decoded_mask > 0] = color_dict[x]  # 设置颜色
                        rows, cols = np.where(decoded_mask > 0)
                        if len(rows) > 0 and len(cols) > 0:  # 确保 mask 不为空
                            x1, y1 = cols.min(), rows.min()  # 左上角
                            x2, y2 = cols.max(), rows.max()  # 右下角
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            frame_boxes.append([x1, y1, x2, y2, color_dict[x]])
                
                # append
                mask = torch.from_numpy(mask)
                if (mask > 0).any():
                    valid.append(1)
                else:
                    valid.append(0)
                masks.append(mask)
                boxes.append(frame_boxes)

            # transform
            masks = torch.stack(masks, dim=0)
            target = {
                'vid': vid,
                'boxes': boxes,  # List of each frame boxes
                'masks': masks,  # [T, H, W, 3]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
                'size': [h, w],
                'exp_id': exp_id,
            }

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return target

def save_masks_as_images(masks, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将 tensor 数据范围归一化到 [0, 255]
    masks = masks.clamp(0, 255).byte()

    # 遍历时间步 T，逐帧保存图片
    for t in tqdm(range(masks.shape[0]), desc="Saving Masks"):
        # 将当前帧转换为 PIL 图像
        img = Image.fromarray(masks[t].cpu().numpy())  # 转为 numpy 数组并传入 PIL

        # 保存图像
        img.save(os.path.join(output_dir, f"frame_{t:04d}.png"))

    print(f"Frames with masks saved in {output_dir}")

def save_boxes_as_images(boxes, save_dir, h, w):
    os.makedirs(save_dir, exist_ok=True)

    # boxes: [T, N, 4] -> T 帧, 每帧有 N 个 box，每个 box 是 4 个坐标
    for t in tqdm(range(len(boxes)), desc="Saving Boxes"):
        # 创建黑色背景
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 当前帧的 boxes
        frame_boxes = boxes[t]  # shape: [N, 4]
        
        # 遍历当前帧的所有 box
        for box in frame_boxes:
            # 提取坐标
            x1, y1, x2, y2, color = box
            # 将 RGB 转为 BGR
            color = tuple(reversed(color))
            # 绘制矩形框，颜色为红色 (0, 0, 255)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # 保存当前帧图片
        save_path = os.path.join(save_dir, f"frame_{t:04d}.png")
        cv2.imwrite(save_path, img)

    print(f"Frames with boxes saved in {save_dir}")

# if __name__ == '__main__':
#     root = Path('/datadrive2/MeViS_release')
#     image_set = 'train'
#     PATHS = {
#         "train": (root / "train", root / "train" / "meta_expressions.json"),
#     }
#     img_folder, ann_file = PATHS['train']

#     dataset = MeViSDataset(img_folder, ann_file, transforms=None)
    # mask_output_dir = "visualization/debug/masks"
    # box_output_dir = "visualization/debug/boxes"
    # total_img_dir = "/datadrive2/MeViS_release/train/JPEGImages"
    # csv_output_file = "data/MeViS/debug.csv"

#     existing_videoids = set()
#     if os.path.exists(csv_output_file):
#         with open(csv_output_file, mode='r') as csv_file:
#             reader = csv.reader(csv_file)
#             next(reader)  # 跳过表头
#             for row in reader:
#                 existing_videoids.add(row[6])  # 第6列是videoid

#     with open(csv_output_file, mode='a', newline='') as csv_file:
#         writer = csv.writer(csv_file)

#         # 如果文件为空，写入表头
#         if csv_file.tell() == 0:
#             writer.writerow([
#                 "text", "path", "num_frames", "width", "height",
#                 "trajectory_maps_path", "videoid", "mask_start_index",
#                 "mask_end_index", "box_path"
#             ])

#         # 初始化数据集
#         dataset = MeViSDataset(img_folder, ann_file, transforms=None)

#         # 遍历数据集
#         for meta in dataset:
#             # 获取元数据
#             videoid = meta["vid"]
#             exp_id = meta['exp_id']
#             h, w = meta["size"]
#             masks = meta["masks"]
#             caption = meta['caption']
#             boxes = meta["boxes"]

#             # 构建唯一的 videoid 标识
#             unique_videoid = f"{videoid}_{exp_id}"

#             # 检查是否已存在于 CSV 文件中
#             if unique_videoid in existing_videoids:
#                 print(f"Skipping {unique_videoid} as it already exists.")
#                 continue  # 跳过已存在的记录

#             # 创建保存路径
#             mask_dir = os.path.join(mask_output_dir, f"{videoid}/{exp_id}")
#             box_dir = os.path.join(box_output_dir, f"{videoid}/{exp_id}")
#             img_dir = os.path.join(total_img_dir, f"{videoid}")
#             os.makedirs(mask_dir, exist_ok=True)
#             os.makedirs(box_dir, exist_ok=True)

#             # 保存 masks 和 boxes
#             save_masks_as_images(masks, mask_dir)
#             save_boxes_as_images(boxes, box_dir, h, w)

#             # 计算元数据信息
#             num_frames = len(boxes)
#             assert num_frames == masks.shape[0]
#             mask_start_index = 0
#             mask_end_index = num_frames - 1
#             box_path = box_dir

#             # 写入 CSV 数据
#             writer.writerow([
#                 caption,           # text
#                 img_dir,           # path
#                 num_frames,        # num_frames
#                 w,                 # width
#                 h,                 # height
#                 mask_dir,          # trajectory_maps_path
#                 unique_videoid,    # videoid
#                 mask_start_index,  # mask_start_index
#                 mask_end_index,    # mask_end_index
#                 box_path           # box_path
#             ])

#             # 添加到已处理列表
#             existing_videoids.add(unique_videoid)
#             break

#     print(f"Metadata CSV saved at: {csv_output_file}")

# 全局变量与配置
root = Path('/datadrive2/MeViS_release')
image_set = 'train'
PATHS = {
    "train": (root / "train", root / "train" / "meta_expressions.json"),
}
img_folder, ann_file = PATHS['train']

mask_output_dir = "/datadrive2/MeViS_release/train/masks"
box_output_dir = "/datadrive2/MeViS_release/train/boxes"
total_img_dir = "/datadrive2/MeViS_release/train/JPEGImages"
csv_output_file = "data/MeViS/MeViS.csv"

# 多线程锁和已处理ID集合
lock = threading.Lock()
existing_videoids = set()

# 加载已存在的videoid
if os.path.exists(csv_output_file):
    with open(csv_output_file, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # 跳过表头
        for row in reader:
            existing_videoids.add(row[6])  # 第6列是videoid
            
def process_meta(meta):
    """处理单个meta数据"""
    # 获取元数据
    videoid = meta["vid"]
    exp_id = meta['exp_id']
    h, w = meta["size"]
    masks = meta["masks"]
    caption = meta['caption']
    boxes = meta["boxes"]

    # 构建唯一的 videoid 标识
    unique_videoid = f"{videoid}_{exp_id}"

    # 检查是否已存在于 CSV 文件中
    with lock:  # 确保线程安全地检查
        if unique_videoid in existing_videoids:
            print(f"Skipping {unique_videoid} as it already exists.")
            return None  # 跳过已存在的记录

    # 创建保存路径
    mask_dir = os.path.join(mask_output_dir, f"{videoid}/{exp_id}")
    box_dir = os.path.join(box_output_dir, f"{videoid}/{exp_id}")
    img_dir = os.path.join(total_img_dir, f"{videoid}")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    # 保存 masks 和 boxes
    save_masks_as_images(masks, mask_dir)
    save_boxes_as_images(boxes, box_dir, h, w)

    # 计算元数据信息
    num_frames = len(boxes)
    assert num_frames == masks.shape[0]
    mask_start_index = 0
    mask_end_index = num_frames - 1
    box_path = box_dir

    # 准备写入数据
    row = [
        caption,           # text
        img_dir,           # path
        num_frames,        # num_frames
        w,                 # width
        h,                 # height
        mask_dir,          # trajectory_maps_path
        unique_videoid,    # videoid
        mask_start_index,  # mask_start_index
        mask_end_index,    # mask_end_index
        box_path           # box_path
    ]

    # 写入CSV（加锁确保线程安全）
    with lock:
        with open(csv_output_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)
        existing_videoids.add(unique_videoid)  # 更新已存在列表

    print(f"Processed {unique_videoid}")
    return unique_videoid

if __name__ == '__main__':
    # 初始化数据集
    dataset = MeViSDataset(img_folder, ann_file, transforms=None)

    # 如果CSV文件为空，写入表头
    with lock:
        if not os.path.exists(csv_output_file) or os.stat(csv_output_file).st_size == 0:
            with open(csv_output_file, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "text", "path", "num_frames", "width", "height",
                    "trajectory_maps_path", "videoid", "mask_start_index",
                    "mask_end_index", "box_path"
                ])

    # 使用线程池并行处理
    max_threads = 90  # 根据机器配置调整线程数
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_meta, meta) for meta in dataset]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Completed {result}")
            except Exception as e:
                print(f"Error occurred: {e}")

    print(f"Metadata CSV saved at: {csv_output_file}")