from glob import glob
import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def normalize_name(name):
    return name.replace('_train_id', '')

def get_matching_image_mask_paths(image_dir, mask_dir, image_ext="*.jpg", mask_ext="*.png"):
    image_paths = glob(os.path.join(image_dir, "**", image_ext), recursive=True)
    mask_paths = glob(os.path.join(mask_dir, mask_ext))

    image_dict = {normalize_name(os.path.splitext(os.path.basename(p))[0]): p for p in image_paths}
    mask_dict = {normalize_name(os.path.splitext(os.path.basename(p))[0]): p for p in mask_paths}

    common_names = set(image_dict.keys()) & set(mask_dict.keys())

    matched_images = [image_dict[name] for name in sorted(common_names)]
    matched_masks = [mask_dict[name] for name in sorted(common_names)]

    print(f"Tổng số ảnh ban đầu: {len(image_paths)}")
    print(f"Tổng số mask ban đầu: {len(mask_paths)}")
    print(f"Số cặp khớp giữ lại: {len(matched_images)}")
    print(f"Số ảnh bị loại bỏ: {len(image_paths) - len(matched_images)}")
    print(f"Số mask bị loại bỏ: {len(mask_paths) - len(matched_masks)}")

    return matched_images, matched_masks


def extract_lane_masks(json_dir, output_mask_dir, image_size=(720, 1280)):
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Converting JSON to mask"):
        json_path = os.path.join(json_dir, json_file)

        with open(json_path, 'r') as f:
            data = json.load(f)

        frames = data.get('frames', [])
        for idx, frame in enumerate(frames):
            mask = np.zeros(image_size, dtype=np.uint8)

            for obj in frame.get('objects', []):
                category = obj.get('category', '')
                if category.startswith('lane/'):
                    polyline = obj.get('poly2d', [])
                    if len(polyline) >= 2:
                        pts = np.array([[int(p[0]), int(p[1])] for p in polyline], dtype=np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=3)

            output_filename = f"{os.path.splitext(json_file)[0]}.png"
            output_path = os.path.join(output_mask_dir, output_filename)
            Image.fromarray(mask * 255).save(output_path)