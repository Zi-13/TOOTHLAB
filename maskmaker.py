import os
import json
import cv2
import numpy as np

features_dir = 'templates/contours'  # BulidTheLab.py生成的contours目录
mask_dir = 'masks'                  # 生成的mask保存目录
os.makedirs(mask_dir, exist_ok=True)

# 假设所有图片尺寸一致，可用一张样例图片获取尺寸
sample_img_path = 'images/Tooth_6.png'  # 请替换为你的任意一张原图路径
if not os.path.exists(sample_img_path):
    raise FileNotFoundError('请将一张原图放在 images/ 目录，并命名为 sample.png 或修改 sample_img_path')
sample_img = cv2.imread(sample_img_path)
h, w = sample_img.shape[:2]

for json_file in os.listdir(features_dir):
    if not json_file.endswith('.json'):
        continue
    with open(os.path.join(features_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    features_list = data.get('features', [])
    mask = np.zeros((h, w), dtype=np.uint8)
    for i, feature in enumerate(features_list):
        # BulidTheLab.py的features里没有points，需从contours json读取points
        # 但如果features里有points字段，则直接用
        points = feature.get('points')
        if points is None:
            print(f"{json_file} 第{i}个目标无points字段，跳过")
            continue
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], i+1)
    mask_path = os.path.join(mask_dir, os.path.splitext(json_file)[0] + '_mask.png')
    cv2.imwrite(mask_path, mask)
    print(f"生成mask: {mask_path}")
