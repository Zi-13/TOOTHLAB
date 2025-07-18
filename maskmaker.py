import os
import json
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path

# 动态导入match.py中的ToothMatcher
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from match import ToothMatcher

def setup_logger(log_level):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def get_image_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return img.shape[:2]

def extract_points_with_match(image_path):
    matcher = ToothMatcher()
    # 自动色块分割：用默认参数，自动HSV阈值（可根据实际情况调整）
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 取整图均值作为自动色块（可优化为自适应聚类等）
    mean_color = np.mean(hsv.reshape(-1, 3), axis=0)
    picked_colors = [mean_color.astype(int)]
    mask = matcher._create_mask(hsv, picked_colors)
    valid_contours, _ = matcher._process_contours(mask)
    # 返回所有有效轮廓的points
    return [vc['points'] for vc in valid_contours if 'points' in vc]

def main():
    parser = argparse.ArgumentParser(description='自动批量生成mask工具')
    parser.add_argument('--input_dir', type=str, default='templates/contours', help='输入json目录')
    parser.add_argument('--output_dir', type=str, default='masks', help='输出mask目录')
    parser.add_argument('--log', type=str, default='INFO', help='日志级别')
    parser.add_argument('--force', action='store_true', help='强制重建已存在的mask')
    args = parser.parse_args()

    logger = setup_logger(args.log.upper())
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in input_dir.iterdir() if f.suffix == '.json']
    logger.info(f'共检测到{len(json_files)}个json文件，开始批量生成mask...')

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            features_list = data.get('features', [])
            image_path = data.get('image_path')
            if not image_path:
                logger.warning(f'{json_file} 缺少image_path字段，跳过')
                continue
            if not os.path.isabs(image_path):
                # 相对路径自动补全
                image_path = os.path.join('images', image_path) if not os.path.exists(image_path) else image_path
            if not os.path.exists(image_path):
                logger.warning(f'{json_file} 对应图片不存在: {image_path}，跳过')
                continue
            h, w = get_image_shape(image_path)
            mask = np.zeros((h, w), dtype=np.uint8)
            points_found = False
            for i, feature in enumerate(features_list):
                points = feature.get('points')
                if points is None:
                    logger.info(f'{json_file} 第{i}个目标无points字段，自动调用match.py提取...')
                    # 自动提取points
                    points_list = extract_points_with_match(image_path)
                    if not points_list:
                        logger.warning(f'{json_file} 第{i}个目标自动提取失败，跳过')
                        continue
                    for j, pts in enumerate(points_list):
                        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], j+1)
                    points_found = True
                    break  # 只要自动提取一次即可
                else:
                    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], i+1)
                    points_found = True
            if not points_found:
                logger.warning(f'{json_file} 未能获得任何points，跳过')
                continue
            mask_path = output_dir / (json_file.stem + '_mask.png')
            if mask_path.exists() and not args.force:
                logger.info(f'{mask_path} 已存在，跳过（如需重建请加--force）')
                continue
            cv2.imwrite(str(mask_path), mask)
            logger.info(f'生成mask: {mask_path}')
        except Exception as e:
            logger.error(f'{json_file} 处理失败: {e}')

if __name__ == '__main__':
    main()
