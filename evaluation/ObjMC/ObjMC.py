import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from Grounded_SAM2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from Grounded_SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from Grounded_SAM2.segment import segment
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def video_segmentation_without_save(video_path, main_objects, video_predictor, image_predictor, processor, grounding_model, device):
    # Use the images_path and main_objects to do video segmentation
    frames = segment(video_path, main_objects, video_predictor, image_predictor, processor, grounding_model, device)
    return frames

def load_masks(masks_path):
    """
    Load ground truth masks from the specified directory.
    
    Args:
        masks_path (str): Path to the directory containing mask images.
    
    Returns:
        list: List of numpy arrays representing the masks.
    """
    masks = []
    for mask_file in os.listdir(masks_path):
        mask_path = os.path.join(masks_path, mask_file)
        mask = Image.open(mask_path).convert('RGB')  # Convert to RGB
        masks.append(np.array(mask))
    return masks

def resize_mask(mask, target_shape):
    """
    Resize the mask to the target shape.
    
    Args:
        mask (numpy array): The mask to be resized.
        target_shape (tuple): The target shape (height, width, channels).
    
    Returns:
        numpy array: The resized mask.
    """
    image = Image.fromarray(mask)
    resized_image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(resized_image)

def normalize_mask(mask):
    return mask / 255.0

def extract_bounding_boxes(mask):
    """
    Extract bounding boxes from a mask.
    
    Args:
        mask (numpy array): The mask to extract bounding boxes from.
    
    Returns:
        list: List of bounding boxes, each represented as [min_x, min_y, max_x, max_y].
    """
    bounding_boxes = []
    mask_img = mask.astype(np.uint8)
    df = pd.DataFrame(mask_img.reshape(-1, 3), columns=['R', 'G', 'B'])
    unique_colors_df = df.drop_duplicates()
    unique_colors = unique_colors_df.to_numpy()
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 排除黑色

    for color in unique_colors:
        mask = cv2.inRange(mask_img, np.array(color), np.array(color))
        coords = np.column_stack(np.where(mask))

        if coords.size > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            bounding_boxes.append([min_x, min_y, max_x, max_y])

    return bounding_boxes

def bounding_box_distance(box1, box2):
    """
    Calculate the distance between two bounding boxes based on their corners.
    
    Args:
        box1 (list): The first bounding box [min_x, min_y, max_x, max_y].
        box2 (list): The second bounding box [min_x, min_y, max_x, max_y].
    
    Returns:
        float: The distance between the two bounding boxes.
    """
    min_distance = np.linalg.norm(np.array([box1[0], box1[1]]) - np.array([box2[0], box2[1]]))
    max_distance = np.linalg.norm(np.array([box1[2], box1[3]]) - np.array([box2[2], box2[3]]))
    return min_distance + max_distance

def euclidean_distance(gt_masks, extracted_masks):
    """
    Calculate the Euclidean distance between ground truth masks and extracted masks.
    
    Args:
        gt_masks (list): List of numpy arrays representing the ground truth masks.
        extracted_masks (list): List of numpy arrays representing the extracted masks.
    
    Returns:
        float: The Euclidean distance score.
    """
    if len(gt_masks) != len(extracted_masks):
        raise ValueError("The number of ground truth masks and extracted masks must be the same.")
    
    total_distance = 0.0
    for gt_mask, ext_mask in zip(gt_masks, extracted_masks):
        assert gt_mask.shape == ext_mask.shape, "The shape of ground truth mask and extracted mask must be the same."
        gt_mask = normalize_mask(gt_mask)
        ext_mask = normalize_mask(ext_mask)
        total_distance += np.linalg.norm(gt_mask - ext_mask)
    
    return total_distance / len(gt_masks)

def initialize_models():
    """
    Initialize the models required for evaluation.
    
    Returns:
        tuple: video_predictor, image_predictor, processor, grounding_model, device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize sam2 video and image predictor
    sam2_checkpoint = "Grounded_SAM2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # initialize grounding-dino model
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("Grounded-SAM2 Models initialized")

    return video_predictor, image_predictor, processor, grounding_model, device

def eval(args, video_predictor, image_predictor, processor, grounding_model, device):
    video_path = args.video_path
    main_objects = args.main_objects
    extracted_masks = video_segmentation_without_save(video_path, main_objects, video_predictor, image_predictor, processor, grounding_model, device) # List of masks images

    # load ground truth masks
    masks_path = args.masks_path
    gt_masks = load_masks(masks_path)
    # Resize the masks to the size of the extracted_masks
    gt_masks = [resize_mask(mask, extracted_masks[0].shape) for mask in gt_masks]
    
    # calculate objmc scores
    objmc_scores = euclidean_distance(gt_masks, extracted_masks)
    print(f"Masks objmc scores: {objmc_scores}")

    # extract bounding boxes and calculate bounding box distances for each frame
    frame_bbox_distances = []
    for gt_mask, ext_mask in zip(gt_masks, extracted_masks):
        gt_bboxes = extract_bounding_boxes(gt_mask)
        ext_bboxes = extract_bounding_boxes(ext_mask)
        bbox_distances = []
        for gt_box, ext_box in zip(gt_bboxes, ext_bboxes):
            bbox_distances.append(bounding_box_distance(gt_box, ext_box))
        frame_bbox_distances.append(np.mean(bbox_distances))
    avg_bbox_distance = np.mean(frame_bbox_distances)
    print(f"Boxes objmc scores: {avg_bbox_distance}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video datasets with GPU parallelization.')

    # 添加命令行参数
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video to be calculated objmc scores')
    parser.add_argument('--masks_path', type=str, required=True, help='Path to the ground truth masks')
    parser.add_argument('--main_objects', type=str, required=True, help='main_objects of this video')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    video_predictor, image_predictor, processor, grounding_model, device = initialize_models()
    eval(args, video_predictor, image_predictor, processor, grounding_model, device)