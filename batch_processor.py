import os
import json
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from BulidTheLab import ToothTemplateBuilder, ContourFeatureExtractor, pick_color_and_draw_edge
import sqlite3

class BatchTemplateProcessor:
    """批量模板处理器 - 支持目录级别的3D截图批量建库"""
    
    def __init__(self, templates_dir="templates", database_path="tooth_templates.db"):
        self.template_builder = ToothTemplateBuilder(database_path, templates_dir)
        self.feature_extractor = ContourFeatureExtractor()
        self.logger = logging.getLogger(__name__)
        self.templates_dir = Path(templates_dir)
        
    def process_directory(self, input_dir: str, auto_confirm: bool = False) -> Dict:
        """批量处理目录中的所有图像文件"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
            
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        self.logger.info(f"发现 {len(image_files)} 个图像文件")
        
        results = {
            'processed': 0,
            'failed': 0,
            'templates_created': []
        }
        
        for image_file in image_files:
            try:
                self.logger.info(f"处理图像: {image_file.name}")
                template_id = self._process_single_image(str(image_file), auto_confirm)
                if template_id:
                    results['processed'] += 1
                    results['templates_created'].append(template_id)
                else:
                    results['failed'] += 1
            except Exception as e:
                self.logger.error(f"处理 {image_file.name} 失败: {e}")
                results['failed'] += 1
                
        return results
    
    def _process_single_image(self, image_path: str, auto_confirm: bool) -> Optional[str]:
        """处理单个图像文件"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return None
                
            tooth_id = self.template_builder.get_next_tooth_id()
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(hsv.reshape(-1, 3), axis=0)
            
            mask = self._create_simple_mask(hsv, mean_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.warning(f"未找到轮廓: {image_path}")
                return None
                
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    points = contour.reshape(-1, 2)
                    features = self.feature_extractor.extract_all_features(contour, points)
                    valid_contours.append({
                        'contour': contour,
                        'features': features,
                        'area': area,
                        'points': contour.reshape(-1, 2).tolist()
                    })
            
            if valid_contours:
                self.template_builder.serialize_contours(valid_contours, tooth_id, image_path)
                return tooth_id
            else:
                self.logger.warning(f"未找到有效轮廓: {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"处理图像失败 {image_path}: {e}")
            return None
    
    def _create_simple_mask(self, hsv: np.ndarray, mean_color: np.ndarray) -> np.ndarray:
        """创建简单的颜色掩码"""
        tolerance = 30
        lower = np.array([max(0, mean_color[0] - tolerance), 50, 50])
        upper = np.array([min(179, mean_color[0] + tolerance), 255, 255])
        return cv2.inRange(hsv, lower, upper)

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='批量模板建库工具')
    parser.add_argument('--input_dir', required=True, help='输入图像目录')
    parser.add_argument('--auto_confirm', action='store_true', help='自动确认所有操作')
    parser.add_argument('--log_level', default='INFO', help='日志级别')
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    processor = BatchTemplateProcessor()
    results = processor.process_directory(args.input_dir, args.auto_confirm)
    
    print(f"批量处理完成:")
    print(f"  成功处理: {results['processed']} 个文件")
    print(f"  处理失败: {results['failed']} 个文件")
    print(f"  创建模板: {results['templates_created']}")

if __name__ == "__main__":
    main()
