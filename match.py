import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import json
import sqlite3
import os
import sys
from datetime import datetime
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
import traceback
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import argparse

# GUI相关导入
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

# matplotlib GUI集成导入
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先黑体、雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['font.size'] = 10

# 路径配置
CURRENT_DIR = Path(__file__).parent
IMAGES_DIR = CURRENT_DIR / 'images'
DEFAULT_IMAGE_NAME = 'test_tooth_3.jpg'  # 可以轻松修改默认图片
PHOTO_PATH = str(IMAGES_DIR / DEFAULT_IMAGE_NAME)

# ===== 尺度标定相关导入和类定义 =====
@dataclass
class ReferenceObject:
    """参考物规格定义"""
    size_mm: float = 10.0
    color_hsv_range: Dict = None
    
    def __post_init__(self):
        if self.color_hsv_range is None:
            self.color_hsv_range = {
                'lower': np.array([0, 144, 169]),
                'upper': np.array([15, 255, 255]),
                'lower2': np.array([165, 144, 169]),
                'upper2': np.array([180, 255, 255])
            }

@dataclass
class CalibrationResult:
    """标定结果数据类"""
    pixel_per_mm: float
    reference_pixel_size: float
    reference_position: Tuple[int, int, int, int]
    confidence: float
    error_message: str = ""

class ReferenceDetector:
    """参考物检测器"""
    
    def __init__(self, reference_obj: ReferenceObject):
        self.reference_obj = reference_obj
        
    def detect_reference_object(self, image: np.ndarray) -> CalibrationResult:
        """检测图像中的参考物并计算标定参数"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = self._create_color_mask(hsv)
            mask = self._clean_mask(mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未检测到参考物颜色")
            
            best_contour = self._find_best_reference_contour(contours)
            
            if best_contour is None:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未找到符合条件的参考物")
            
            return self._calculate_calibration(best_contour)
            
        except Exception as e:
            logger.error(f"参考物检测失败: {e}")
            return CalibrationResult(0, 0, (0, 0, 0, 0), 0, f"检测异常: {str(e)}")
    
    def _create_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建颜色掩码"""
        color_range = self.reference_obj.color_hsv_range
        mask1 = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
        return cv2.bitwise_or(mask1, mask2)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """清理掩码噪声"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _find_best_reference_contour(self, contours: List) -> Optional[np.ndarray]:
        """找到最佳的参考物轮廓"""
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            features = self._analyze_contour_shape(contour)
            score = self._evaluate_reference_candidate(features)
            
            if score > 0.5:
                candidates.append((contour, score, features))
        
        if not candidates:
            return None
        
        best_contour, best_score, best_features = max(candidates, key=lambda x: x[1])
        return best_contour
    
    def _analyze_contour_shape(self, contour: np.ndarray) -> Dict:
        """分析轮廓形状特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'circularity': circularity,
            'solidity': solidity,
            'bounding_rect': (x, y, w, h)
        }
    
    def _evaluate_reference_candidate(self, features: Dict) -> float:
        """评估参考物候选的质量"""
        score = 0.0
        
        aspect_ratio = features['aspect_ratio']
        if 0.8 <= aspect_ratio <= 1.25:
            score += 0.3
        
        rectangularity = features['rectangularity']
        if rectangularity > 0.7:
            score += 0.3
        
        solidity = features['solidity']
        if solidity > 0.8:
            score += 0.2
        
        area = features['area']
        if 100 <= area <= 10000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_calibration(self, contour: np.ndarray) -> CalibrationResult:
        """计算标定参数 - 修复正方形标定物的比例计算"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # 对于正方形标定物，使用较大的边作为像素尺寸
        # 这样可以避免检测误差导致的尺寸偏小
        pixel_size = max(w, h)  # 修复：使用最大边长而不是平均值
        
        # 计算像素/毫米比例
        # pixel_per_mm = 像素边长 / 真实边长(mm)
        pixel_per_mm = pixel_size / self.reference_obj.size_mm
        
        # 计算置信度（基于正方形程度）
        aspect_ratio = w / h if h > 0 else 0
        confidence = 1.0 - abs(1.0 - aspect_ratio)  # 越接近1（正方形）置信度越高
        confidence = max(0.0, min(1.0, confidence))
        
        # 添加调试信息
        logger.info(f"🔴 标定物检测结果:")
        logger.info(f"   检测位置: ({x}, {y}), 尺寸: {w}×{h} 像素")
        logger.info(f"   像素边长: {pixel_size} px")
        logger.info(f"   真实边长: {self.reference_obj.size_mm} mm")
        logger.info(f"   比例系数: {pixel_per_mm:.3f} px/mm")
        logger.info(f"   置信度: {confidence:.3f}")
        logger.info(f"   面积换算公式: 像素面积 ÷ {pixel_per_mm:.1f}² = 真实面积(mm²)")
        
        return CalibrationResult(
            pixel_per_mm=pixel_per_mm,
            reference_pixel_size=pixel_size,
            reference_position=(x, y, w, h),
            confidence=confidence
        )
# ===== 尺度标定相关类定义结束 =====



# 验证路径是否存在
if not IMAGES_DIR.exists():
    print(f"⚠️ 图像目录不存在: {IMAGES_DIR}")
    print("💡 请创建 images 目录并放入图片")

if not Path(PHOTO_PATH).exists():
    print(f"⚠️ 默认图片不存在: {PHOTO_PATH}")
    # 尝试找到第一个可用的圖片
    image_files = list(IMAGES_DIR.glob('*.png')) + list(IMAGES_DIR.glob('*.jpg'))
    if image_files:
        PHOTO_PATH = str(image_files[0])
        print(f"💡 使用第一个找到的图片: {PHOTO_PATH}")

# 配置常量
class Config:
    DEFAULT_HSV_TOLERANCE = {'h': 15, 's': 60, 'v': 60}
    FOURIER_ORDER = 80
    MIN_CONTOUR_POINTS = 20
    SIMILARITY_THRESHOLD = 0.99  # 改为1.0作为临界值
    SIZE_TOLERANCE = 0.3
    DATABASE_PATH = "tooth_templates.db"
    TEMPLATES_DIR = "templates"
    
    # 尺度标定相关配置
    REFERENCE_SIZE_MM = 10.0  # 默认参考物尺寸(毫米)
    SCALE_CALIBRATION_MODE = "auto"  # auto, manual, traditional
    SCALE_CONFIDENCE_THRESHOLD = 0.5  # 尺度标定置信度阈值
    ENABLE_SCALE_NORMALIZATION = True  # 是否启用尺度归一化
  
class FourierAnalyzer:
    """傅里叶级数分析器"""
    
    @staticmethod
    def fit_fourier_series(data: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
        """拟合傅里叶级数"""
        try:
            A = np.ones((len(t), 2 * order + 1))
            for k in range(1, order + 1):
                A[:, 2 * k - 1] = np.cos(k * t)
                A[:, 2 * k] = np.sin(k * t)
            coeffs, _, _, _ = lstsq(A, data, rcond=None)
            return coeffs
        except Exception as e:
            logger.error(f"傅里叶级数拟合失败: {e}")
            return np.zeros(2 * order + 1)

    @staticmethod
    def evaluate_fourier_series(coeffs: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
        """计算傅里叶级数值"""
        A = np.ones((len(t), 2 * order + 1))
        for k in range(1, order + 1):
            A[:, 2 * k - 1] = np.cos(k * t)
            A[:, 2 * k] = np.sin(k * t)
        return A @ coeffs

    def analyze_contour(self, points: np.ndarray, order: int = Config.FOURIER_ORDER, 
                       center_normalize: bool = True) -> dict:
        """分析轮廓的傅里叶特征"""
        try:
            x = points[:, 0].astype(float)
            y = points[:, 1].astype(float)
            
            # TODO 计算几何中心
            center_x = np.mean(x)
            center_y = np.mean(y)
            
            if center_normalize:
                # TODO 以几何中心为原点进行归一化
                x_normalized = x - center_x
                y_normalized = y - center_y
                
                # TODO 计算缩放因子（使用最大距离进行归一化）
                max_dist = np.max(np.sqrt(x_normalized**2 + y_normalized**2))
                if max_dist > 0:
                    x_normalized /= max_dist
                    y_normalized /= max_dist
            else:
                x_normalized = x
                y_normalized = y
                max_dist = 1.0
            
            N = len(points)
            t = np.linspace(0, 2 * np.pi, N)
            
            # TODO 对归一化后的坐标进行傅里叶拟合
            coeffs_x = self.fit_fourier_series(x_normalized, t, order)
            coeffs_y = self.fit_fourier_series(y_normalized, t, order)
            
            # TODO 生成更密集的参数点用于平滑显示
            t_dense = np.linspace(0, 2 * np.pi, N * 4)
            x_fit_normalized = self.evaluate_fourier_series(coeffs_x, t_dense, order)
            y_fit_normalized = self.evaluate_fourier_series(coeffs_y, t_dense, order)
            
            if center_normalize:
                # TODO 将拟合结果还原到原始坐标系
                x_fit = x_fit_normalized * max_dist + center_x
                y_fit = y_fit_normalized * max_dist + center_y
            else:
                x_fit = x_fit_normalized
                y_fit = y_fit_normalized
            
            return {
                'coeffs_x': coeffs_x,
                'coeffs_y': coeffs_y,
                'center_x': center_x,
                'center_y': center_y,
                'max_dist': max_dist,
                'order': order,
                'x_fit': x_fit,
                'y_fit': y_fit,
                'original_points': (x, y)
            }
            
        except Exception as e:
            logger.error(f"傅里叶分析失败: {e}")
            return {}

class ContourFeatureExtractor:
    """轮廓特征提取器"""
    
    def __init__(self):
        self.fourier_analyzer = FourierAnalyzer()
    
    def extract_geometric_contours(self, contour: np.ndarray, image_shape=None) -> dict:
        contours = {}
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if image_shape is not None:
            h, w = image_shape[:2]
            diag = (h**2 + w**2) ** 0.5
            area_norm = area / (diag ** 2)
            perimeter_norm = perimeter / diag
        else:
            area_norm = area
            perimeter_norm = perimeter
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = w_box / h_box if h_box != 0 else 0
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corner_count = len(approx)
        contours.update({
            'area': area,
            'perimeter': perimeter,
            'area_norm': area_norm,
            'perimeter_norm': perimeter_norm,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'corner_count': corner_count,
            'bounding_rect': (x, y, w_box, h_box)
        })
        return contours
    
    def extract_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        """提取Hu矩特征"""
        try:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # TODO 对数变换使其更稳定
            for i in range(len(hu_moments)):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
                else:
                    hu_moments[i] = 0
            
            return hu_moments
        except Exception as e:
            logger.error(f"Hu矩计算失败: {e}")
            return np.zeros(7)
    
    def extract_fourier_descriptors(self, points: np.ndarray) -> np.ndarray:
        """提取傅里叶描述符"""
        try:
            fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
            if fourier_data is not None:
                coeffs_x = fourier_data['coeffs_x']
                coeffs_y = fourier_data['coeffs_y']
                # TODO 组合前11个系数（0阶+10阶*2）
                fourier_contours = np.concatenate([coeffs_x[:11], coeffs_y[:11]])
                return fourier_contours
            else:
                return np.zeros(22)
        except Exception as e:
            logger.error(f"傅里叶描述符提取失败: {e}")
            return np.zeros(22)
    
    def extract_all_contours(self, contour: np.ndarray, points: np.ndarray, image_shape=None) -> dict:
        contours = {}
        geometric_contours = self.extract_geometric_contours(contour, image_shape=image_shape)
        contours.update(geometric_contours)
        contours['hu_moments'] = self.extract_hu_moments(contour)
        contours['fourier_descriptors'] = self.extract_fourier_descriptors(points)
        fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
        if fourier_data is not None:
            contours['fourier_x_fit'] = fourier_data['x_fit'].tolist()
            contours['fourier_y_fit'] = fourier_data['y_fit'].tolist()
        return contours

class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def calculate_size_similarity(contours1: dict, contours2: dict) -> float:
        """计算尺寸相似度（只用原始面积和周长）"""
        area1 = contours1.get('area', 0)
        area2 = contours2.get('area', 0)
        perimeter1 = contours1.get('perimeter', 0)
        perimeter2 = contours2.get('perimeter', 0)
        # 计算面积相似度
        if area1 == 0 and area2 == 0:
            area_sim = 1.0
        elif area1 == 0 or area2 == 0:
            area_sim = 0.0
        else:
            area_ratio = min(area1, area2) / max(area1, area2)
            area_sim = area_ratio
        # 计算周长相似度
        if perimeter1 == 0 and perimeter2 == 0:
            perimeter_sim = 1.0
        elif perimeter1 == 0 or perimeter2 == 0:
            perimeter_sim = 0.0
        else:
            perimeter_ratio = min(perimeter1, perimeter2) / max(perimeter1, perimeter2)
            perimeter_sim = perimeter_ratio
        return 0.3*area_sim + 0.7*perimeter_sim
    
    @staticmethod
    def calculate_geometric_similarity(contours1: dict, contours2: dict) -> float:
        """计算几何特征相似度"""
        geometric_contours = ['circularity', 'aspect_ratio', 'solidity']
        geometric_weights = [0.2, 0.1, 0.7]
        
        geometric_sim = []
        for feat in geometric_contours:
            v1, v2 = contours1[feat], contours2[feat]
            if v1 == 0 and v2 == 0:
                sim = 1.0
            elif v1 == 0 or v2 == 0:
                sim = 0.0
            else:
                diff = abs(v1 - v2) / max(v1, v2)
                sim = max(0, 1 - diff * 1.5)
            geometric_sim.append(sim)
        
        return sum(w * s for w, s in zip(geometric_weights, geometric_sim))
    
    @staticmethod
    def calculate_hu_similarity(contours1: dict, contours2: dict) -> float:
        """计算Hu矩相似度"""
        try:
            hu1 = contours1['hu_moments']
            hu2 = contours2['hu_moments']
            hu_sim = cosine_similarity([hu1], [hu2])[0][0]
            return max(0, hu_sim)
        except Exception as e:
            logger.error(f"Hu矩相似度计算失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_fourier_similarity(contours1: dict, contours2: dict) -> float:
        """计算傅里叶描述符相似度"""
        try:
            fourier1 = contours1['fourier_descriptors']
            fourier2 = contours2['fourier_descriptors']
            fourier_sim = cosine_similarity([fourier1], [fourier2])[0][0]
            return max(0, fourier_sim)
        except Exception as e:
            logger.error(f"傅里叶相似度计算失败: {e}")
            return 0.0
    
    def compare_contours(self, contours1: dict, contours2: dict, 
                        size_tolerance: float = Config.SIZE_TOLERANCE) -> dict:
        """比较两个轮廓的相似度"""
        similarities = {}
        
        # TODO 计算各项相似度
        size_similarity = self.calculate_size_similarity(contours1, contours2)
        similarities['size'] = size_similarity
        
        # TODO 一级筛选：如果尺寸差异过大，直接返回低相似度
        if size_similarity < size_tolerance:
            similarities.update({
                'geometric': 0.0,
                'hu_moments': 0.0,
                'fourier': 0.0,
                'overall': size_similarity
            })
            return similarities
        
        # TODO 计算形状特征相似度
        geometric_sim = self.calculate_geometric_similarity(contours1, contours2)
        hu_sim = self.calculate_hu_similarity(contours1, contours2)
        fourier_sim = self.calculate_fourier_similarity(contours1, contours2)
        
        similarities.update({
            'geometric': geometric_sim,
            'hu_moments': hu_sim,
            'fourier': fourier_sim
        })
        
        # TODO 计算最终相似度
        shape_weights = {
            'geometric': 0.55,
            'hu_moments': 0.05,
            'fourier': 0.4
        }
        
        shape_similarity = sum(shape_weights[k] * similarities[k] for k in shape_weights)
        
        # TODO 最终相似度 = 尺寸相似度 × 形状相似度
        size_weight, shape_weight = 0.1, 0.9
        similarities['overall'] = size_similarity * size_weight + shape_similarity * shape_weight
        
        return similarities

    @staticmethod
    def compare_contours_approx(contours1: dict, contours2: dict, rel_tol=0.01, abs_tol=0.1) -> dict:
        # 主特征用相对误差
        keys = ['area', 'perimeter', 'aspect_ratio', 'circularity', 'solidity']
        all_close = True
        for k in keys:
            v1 = float(contours1.get(k, 0))
            v2 = float(contours2.get(k, 0))
            if abs(v1 - v2) / (abs(v1) + 1e-6) > rel_tol:
                all_close = False
                break
        # Hu矩、傅里叶用绝对误差
        hu1 = np.array(contours1.get('hu_moments', []))
        hu2 = np.array(contours2.get('hu_moments', []))
        if hu1.shape == hu2.shape and np.all(np.abs(hu1 - hu2) < abs_tol):
            pass
        else:
            all_close = False
        f1 = np.array(contours1.get('fourier_descriptors', []))
        f2 = np.array(contours2.get('fourier_descriptors', []))
        if f1.shape == f2.shape and np.all(np.abs(f1 - f2) < abs_tol):
            pass
        else:
            all_close = False
        if all_close:
            return {'overall': 1.0, 'size': 1.0, 'geometric': 1.0, 'hu_moments': 1.0, 'fourier': 1.0}
        # 否则走原有逻辑
        return SimilarityCalculator().compare_contours(contours1, contours2)

class DatabaseInterface:
    """数据库接口类"""
    
    def __init__(self, database_path=Config.DATABASE_PATH):
        self.database_path = database_path
        self.templates_dir = Path(Config.TEMPLATES_DIR)
    
    def load_all_templates(self):
        """加载所有模板数据"""
        if not Path(self.database_path).exists():
            logger.warning(f"数据库文件不存在: {self.database_path}")
            return {}
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # 检查是否有增强的特征列
            cursor.execute("PRAGMA table_info(templates)")
            columns = [column[1] for column in cursor.fetchall()]
            has_contours = 'contours_json' in columns
            
            if has_contours:
                # 使用增强的数据库结构
                cursor.execute('''
                    SELECT tooth_id, contour_file, contours_json, geometric_weights, 
                           similarity_weights, num_contours, total_area
                    FROM templates WHERE contours_json IS NOT NULL
                ''')
                
                templates = {}
                for row in cursor.fetchall():
                    tooth_id, contour_file, contours_json, geo_weights, sim_weights, num_contours, total_area = row
                    
                    # 解析特征数据
                    contours_data = json.loads(contours_json) if contours_json else []
                    
                    # 转换为match.py兼容格式
                    compatible_contours = []
                    for feature in contours_data:
                        converted = self._convert_to_match_format(feature)
                        compatible_contours.append(converted)
                    
                    templates[tooth_id] = {
                        'contours': compatible_contours,
                        'contour_file': contour_file,
                        'num_contours': num_contours,
                        'total_area': total_area,
                        'geometric_weights': json.loads(geo_weights) if geo_weights else None,
                        'similarity_weights': json.loads(sim_weights) if sim_weights else None
                    }
                
            else:
                # 使用基础数据库结构，从文件加载特征
                cursor.execute('''
                    SELECT tooth_id, contour_file, num_contours, total_area
                    FROM templates
                ''')
                
                templates = {}
                for tooth_id, contour_file, num_contours, total_area in cursor.fetchall():
                    # 尝试加载特征文件
                    contours = self._load_contours_from_file(tooth_id)
                    if contours:
                        templates[tooth_id] = {
                            'contours': contours,
                            'contour_file': contour_file,
                            'num_contours': num_contours,
                            'total_area': total_area
                        }
            
            logger.info(f"📚 已加载 {len(templates)} 个模板，共 {sum(len(t['contours']) for t in templates.values())} 个轮廓特征")
            return templates
            
        except Exception as e:
            logger.error(f"❌ 加载模板失败: {e}")
            return {}
        finally:
            conn.close()
    
    def _convert_to_match_format(self, contour_dict):
        """将单个contour字典转换为match.py兼容格式"""
        features = contour_dict['features']
        return {
            'area': features['area'],
            'perimeter': features['perimeter'],
            'aspect_ratio': features['aspect_ratio'],
            'circularity': features['circularity'],
            'solidity': features['solidity'],
            'corner_count': features['corner_count'],
            'hu_moments': np.array(features['hu_moments']),
            'fourier_descriptors': np.array(features['fourier_descriptors'])
        }
    
    def _load_contours_from_file(self, tooth_id):
        """从特征文件加载特征"""
        contours_file = self.templates_dir / "contours" / f"{tooth_id}.json"
        
        if not contours_file.exists():
            logger.warning(f"特征文件不存在: {contours_file}")
            return []
        
        try:
            with open(contours_file, 'r', encoding='utf-8') as f:
                contours_data = json.load(f)
            
            compatible_contours = []
            for contour in contours_data['contours']:
                converted = self._convert_to_match_format(contour)
                compatible_contours.append(converted)
            
            return compatible_contours
            
        except Exception as e:
            logger.error(f"❌ 加载特征文件失败: {e}")
            return []
    
    def save_match_result(self, template_id, query_image_path, query_contour_idx, similarities):
        """保存匹配结果到数据库"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # 检查是否有匹配记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT,
                    query_image_path TEXT,
                    query_contour_idx INTEGER,
                    similarity_overall REAL,
                    similarity_size REAL,
                    similarity_geometric REAL,
                    similarity_hu REAL,
                    similarity_fourier REAL,
                    match_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO match_records 
                (template_id, query_image_path, query_contour_idx, 
                 similarity_overall, similarity_size, similarity_geometric, 
                 similarity_hu, similarity_fourier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_id, query_image_path, query_contour_idx,
                similarities['overall'], similarities['size'], similarities['geometric'],
                similarities['hu_moments'], similarities['fourier']
            ))
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ 保存匹配结果失败: {e}")
        finally:
            conn.close()

def load_features_templates(features_dir="templates/features"):
    templates = {}
    for fname in os.listdir(features_dir):
        if fname.endswith("_features.json"):
            tooth_id = fname.split("_features.json")[0].upper()
            with open(os.path.join(features_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            templates[tooth_id] = data["features"]
    return templates

class ToothMatcher:
    """牙齿匹配器主类 - 增强版（集成尺度标定）"""
    
    def __init__(self, scale_mode: str = Config.SCALE_CALIBRATION_MODE, 
                 reference_size_mm: float = Config.REFERENCE_SIZE_MM):
        # 原有组件
        self.feature_extractor = ContourFeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.fourier_analyzer = FourierAnalyzer()
        self.db_interface = DatabaseInterface()
        self.templates = load_features_templates()
        self.current_image_path = None
        self.highlight_template = None  # (template_id, template_contour_idx)
        self._db_match_line_boxes = []  # 存储匹配区每行的bbox和match_id
        self.match_highlight_idx = None  # 当前色块下高亮的数据库匹配索引
        
        # 尺度标定相关组件
        self.scale_mode = scale_mode  # "auto", "manual", "traditional"
        self.reference_size_mm = reference_size_mm
        self.reference_obj = ReferenceObject(size_mm=reference_size_mm)
        self.reference_detector = ReferenceDetector(self.reference_obj)
        self.calibration_result = None  # 当前图像的标定结果
        self.manual_pixel_per_mm = None  # 手动指定的尺度比例
        
        logger.info(f"🦷 ToothMatcher初始化完成 - 尺度模式: {scale_mode}, 参考物尺寸: {reference_size_mm}mm")

    def load_templates(self):
        """加载模板库"""
        self.templates = load_features_templates()
        return len(self.templates) > 0
    
    def set_manual_scale(self, pixel_per_mm: float):
        """手动设置尺度比例"""
        self.manual_pixel_per_mm = pixel_per_mm
        self.scale_mode = "manual"
        logger.info(f"📏 手动设置尺度比例: {pixel_per_mm:.4f} px/mm")
    
    def get_effective_pixel_per_mm(self) -> Optional[float]:
        """获取有效的像素/毫米比例"""
        if self.scale_mode == "manual" and self.manual_pixel_per_mm:
            return self.manual_pixel_per_mm
        elif self.scale_mode == "auto" and self.calibration_result:
            if self.calibration_result.confidence >= Config.SCALE_CONFIDENCE_THRESHOLD:
                return self.calibration_result.pixel_per_mm
        return None
    
    def normalize_features_by_scale(self, features: dict, pixel_per_mm: float) -> dict:
        """根据尺度比例归一化特征
        
        Args:
            features: 原始特征字典
            pixel_per_mm: 像素/毫米比例
        
        Returns:
            归一化后的特征字典
        """
        if not Config.ENABLE_SCALE_NORMALIZATION or pixel_per_mm <= 0:
            return features.copy()
        
        normalized_features = features.copy()
        
        # 面积特征：除以 (pixel_per_mm)²
        area_scale_factor = pixel_per_mm ** 2
        if 'area' in normalized_features:
            normalized_features['area'] = normalized_features['area'] / area_scale_factor
        if 'area_norm' in normalized_features:
            normalized_features['area_norm'] = normalized_features['area_norm'] / area_scale_factor
        
        # 周长特征：除以 pixel_per_mm
        perimeter_scale_factor = pixel_per_mm
        if 'perimeter' in normalized_features:
            normalized_features['perimeter'] = normalized_features['perimeter'] / perimeter_scale_factor
        if 'perimeter_norm' in normalized_features:
            normalized_features['perimeter_norm'] = normalized_features['perimeter_norm'] / perimeter_scale_factor
        
        # 形状特征保持不变（天然尺度不变）
        # aspect_ratio, circularity, solidity, corner_count 不需要处理
        
        # Hu矩保持不变（天然尺度不变）
        # hu_moments 不需要处理
        
        # 傅里叶描述符需要特殊处理（尺度归一化）
        if 'fourier_descriptors' in normalized_features:
            fourier_descriptors = np.array(normalized_features['fourier_descriptors'])
            # 傅里叶描述符的第一个系数（DC分量）与尺度相关，其他系数相对尺度不变
            if len(fourier_descriptors) > 0:
                # 对DC分量进行归一化，其他保持不变
                fourier_descriptors[0] = fourier_descriptors[0] / pixel_per_mm
                if len(fourier_descriptors) > Config.FOURIER_ORDER:
                    fourier_descriptors[Config.FOURIER_ORDER] = fourier_descriptors[Config.FOURIER_ORDER] / pixel_per_mm
                normalized_features['fourier_descriptors'] = fourier_descriptors.tolist()
        
        # 添加归一化标记
        normalized_features['_scale_normalized'] = True
        normalized_features['_pixel_per_mm'] = pixel_per_mm
        
        return normalized_features
    
    def denormalize_features_by_scale(self, normalized_features: dict, target_pixel_per_mm: float) -> dict:
        """将归一化的特征反归一化到目标尺度
        
        Args:
            normalized_features: 归一化的特征字典
            target_pixel_per_mm: 目标像素/毫米比例
        
        Returns:
            反归一化的特征字典
        """
        if not normalized_features.get('_scale_normalized', False) or target_pixel_per_mm <= 0:
            return normalized_features.copy()
        
        denormalized_features = normalized_features.copy()
        
        # 面积特征：乘以 (target_pixel_per_mm)²
        area_scale_factor = target_pixel_per_mm ** 2
        if 'area' in denormalized_features:
            denormalized_features['area'] = denormalized_features['area'] * area_scale_factor
        if 'area_norm' in denormalized_features:
            denormalized_features['area_norm'] = denormalized_features['area_norm'] * area_scale_factor
        
        # 周长特征：乘以 target_pixel_per_mm
        perimeter_scale_factor = target_pixel_per_mm
        if 'perimeter' in denormalized_features:
            denormalized_features['perimeter'] = denormalized_features['perimeter'] * perimeter_scale_factor
        if 'perimeter_norm' in denormalized_features:
            denormalized_features['perimeter_norm'] = denormalized_features['perimeter_norm'] * perimeter_scale_factor
        
        # 傅里叶描述符反归一化
        if 'fourier_descriptors' in denormalized_features:
            fourier_descriptors = np.array(denormalized_features['fourier_descriptors'])
            if len(fourier_descriptors) > 0:
                fourier_descriptors[0] = fourier_descriptors[0] * target_pixel_per_mm
                if len(fourier_descriptors) > Config.FOURIER_ORDER:
                    fourier_descriptors[Config.FOURIER_ORDER] = fourier_descriptors[Config.FOURIER_ORDER] * target_pixel_per_mm
                denormalized_features['fourier_descriptors'] = fourier_descriptors.tolist()
        
        # 更新归一化标记
        denormalized_features['_pixel_per_mm'] = target_pixel_per_mm
        
        return denormalized_features
    
    def match_against_database(self, query_features_list, threshold=Config.SIMILARITY_THRESHOLD):
        """与数据库模板进行匹配（支持尺度归一化）"""
        if not self.templates:
            logger.warning("❌ 未加载模板数据，请先使用 BuildTheLab 创建模板")
            return {}
        
        effective_pixel_per_mm = self.get_effective_pixel_per_mm()
        scale_normalized = Config.ENABLE_SCALE_NORMALIZATION and effective_pixel_per_mm is not None
        
        if scale_normalized:
            logger.info(f"🔄 使用尺度归一化匹配 (比例: {effective_pixel_per_mm:.4f} px/mm)")
        else:
            logger.info("🔄 使用传统匹配模式")
        
        all_matches = {}
        for query_idx, query_features in enumerate(query_features_list):
            query_matches = []
            
            # 检查查询特征是否已归一化
            query_is_normalized = query_features.get('_scale_normalized', False)
            
            for template_id, template_features_list in self.templates.items():
                for template_idx, template_features in enumerate(template_features_list):
                    
                    # 准备用于比较的特征
                    if scale_normalized and query_is_normalized:
                        # 查询已归一化，需要将模板也归一化到相同标准
                        # 注意：这里假设模板特征是原始尺度，需要根据实际情况调整
                        comparison_template_features = template_features.copy()
                        comparison_query_features = query_features.copy()
                    else:
                        # 传统模式，直接比较
                        comparison_template_features = template_features
                        comparison_query_features = query_features
                    
                    # 计算相似度
                    similarities = self.similarity_calculator.compare_contours_approx(
                        comparison_query_features, comparison_template_features, rel_tol=0.01, abs_tol=0.1)
                    
                    # 添加尺度信息到相似度结果中
                    if scale_normalized:
                        similarities['scale_info'] = {
                            'pixel_per_mm': effective_pixel_per_mm,
                            'confidence': self.calibration_result.confidence if self.calibration_result else 0.0,
                            'scale_mode': self.scale_mode
                        }
                    
                    if similarities['overall'] >= threshold:
                        match_info = {
                            'template_id': template_id,
                            'template_contour_idx': template_idx,
                            'similarity': similarities['overall'],
                            'details': similarities,
                            'query_contour_idx': query_idx,
                            'scale_normalized': scale_normalized
                        }
                        query_matches.append(match_info)
                        
                        # 保存匹配结果到数据库
                        if self.current_image_path:
                            self.db_interface.save_match_result(
                                template_id, self.current_image_path, query_idx, similarities
                            )
            
            # 按相似度排序
            query_matches.sort(key=lambda x: x['similarity'], reverse=True)
            all_matches[f'query_{query_idx}'] = query_matches
            
            # 输出匹配统计
            if query_matches:
                best_match = query_matches[0]
                logger.info(f"轮廓 {query_idx}: 最佳匹配 {best_match['template_id']}-{best_match['template_contour_idx']+1} "
                          f"(相似度: {best_match['similarity']:.3f})")
            else:
                logger.info(f"轮廓 {query_idx}: 无匹配结果")
        
        return all_matches
    
    def find_similar_contours(self, target_contours: dict, all_contours: list, 
                             threshold: float = Config.SIMILARITY_THRESHOLD,
                             size_tolerance: float = Config.SIZE_TOLERANCE) -> list:
        """找到与目标轮廓相似的所有轮廓（当前图像内部）"""
        similar_contours = []
        
        for i, contours in enumerate(all_contours):
            if contours == target_contours:
                continue
            
            similarities = self.similarity_calculator.compare_contours(
                target_contours, contours, size_tolerance)
            
            if similarities['overall'] >= threshold:
                similar_contours.append({
                    'index': i,
                    'similarity': similarities['overall'],
                    'details': similarities
                })
        
        similar_contours.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_contours
    
    def process_image(self, image_path: str):
        """处理图像的主函数（集成尺度标定）"""
        self.current_image_path = image_path
        
        # 验证文件路径
        if not Path(image_path).exists():
            logger.error(f"图像文件不存在: {image_path}")
            return
        
        # 加载模板库
        if not self.load_templates():
            logger.warning("⚠️ 未找到模板库，仅显示当前图像轮廓分析")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error("图片读取失败")
            return
        
        # ===== 尺度标定阶段 =====
        self.calibration_result = None
        if self.scale_mode == "auto":
            logger.info("🔍 开始自动尺度标定...")
            self.calibration_result = self.reference_detector.detect_reference_object(img)
            
            if self.calibration_result.pixel_per_mm > 0:
                logger.info(f"✅ 尺度标定成功!")
                logger.info(f"   比例系数: {self.calibration_result.pixel_per_mm:.4f} px/mm")
                logger.info(f"   置信度: {self.calibration_result.confidence:.3f}")
                logger.info(f"   参考物位置: {self.calibration_result.reference_position}")
            else:
                logger.warning(f"❌ 尺度标定失败: {self.calibration_result.error_message}")
                if Config.ENABLE_SCALE_NORMALIZATION:
                    logger.warning("🔄 自动降级到传统模式")
                    self.scale_mode = "traditional"
        elif self.scale_mode == "manual":
            logger.info(f"📏 使用手动尺度设置: {self.manual_pixel_per_mm:.4f} px/mm")
        else:
            logger.info("🔄 使用传统模式（无尺度标定）")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        picked_colors = self._pick_colors(img, hsv)
        
        if not picked_colors:
            logger.warning("未选取颜色")
            return
        
        # 创建掩码并提取轮廓
        mask = self._create_mask(hsv, picked_colors)
        color_extract = cv2.bitwise_and(img, img, mask=mask)
        
        # 处理轮廓
        valid_contours, all_contours = self._process_contours(mask)
        
        if not valid_contours:
            logger.warning("未检测到有效轮廓")
            return
        
        logger.info(f"检测到 {len(valid_contours)} 个有效轮廓")
        
        # ===== 特征归一化阶段 =====
        effective_pixel_per_mm = self.get_effective_pixel_per_mm()
        normalized_query_features_list = []
        
        for i, contour_info in enumerate(valid_contours):
            original_features = contour_info['contours']
            
            if effective_pixel_per_mm and Config.ENABLE_SCALE_NORMALIZATION:
                # 进行尺度归一化
                normalized_features = self.normalize_features_by_scale(original_features, effective_pixel_per_mm)
                logger.debug(f"轮廓 {i}: 原始面积={original_features.get('area', 0):.0f}, "
                           f"归一化面积={normalized_features.get('area', 0):.2f}")
            else:
                # 传统模式，不进行归一化
                normalized_features = original_features.copy()
                normalized_features['_scale_normalized'] = False
            
            normalized_query_features_list.append(normalized_features)
            # 同时更新contour_info中的特征
            contour_info['normalized_contours'] = normalized_features
        
        # 与数据库进行匹配（使用归一化后的特征）
        matches = self.match_against_database(normalized_query_features_list)
        
        # 显示交互式界面
        self._show_interactive_display(color_extract, valid_contours, all_contours, matches)
    
    def _pick_colors(self, img: np.ndarray, hsv: np.ndarray) -> list:
        """颜色选择 - 自动调整显示大小"""
        picked = []
        original_img = img.copy()
        original_hsv = hsv.copy()
        
        # 获取屏幕尺寸的估计值（保守估计）
        max_width = 1200
        max_height = 800
        
        # 计算缩放比例
        h, w = img.shape[:2]
        scale_w = max_width / w if w > max_width else 1.0
        scale_h = max_height / h if h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        # 如果需要缩放，则缩放图像
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(img, (new_w, new_h))
            display_hsv = cv2.resize(hsv, (new_w, new_h))
            logger.info(f"图像缩放: {w}x{h} -> {new_w}x{new_h} (缩放比例: {scale:.2f})")
        else:
            display_img = img
            display_hsv = hsv
            scale = 1.0
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 将显示坐标转换回原始图像坐标
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                
                # 确保坐标在原始图像范围内
                orig_x = max(0, min(orig_x, original_img.shape[1] - 1))
                orig_y = max(0, min(orig_y, original_img.shape[0] - 1))
                
                color = original_hsv[orig_y, orig_x]
                logger.info(f"选中点 显示坐标:({x},{y}) -> 原始坐标:({orig_x},{orig_y}) HSV: {color}")
                picked.append(color)
                
                # 在图像上标记选中点
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), 2)
                cv2.putText(display_img, f"{len(picked)}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("点击选取目标区域颜色 (按空格键完成选择)", display_img)
        
        # 创建窗口并设置可调整大小
        window_name = "点击选取目标区域颜色 (按空格键完成选择)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_img)
        cv2.setMouseCallback(window_name, on_mouse)
        
        print("🎯 颜色选择说明:")
        print("  • 点击图像中的目标区域来选择颜色")
        print("  • 可以选择多个颜色点")
        print("  • 按空格键或ESC键完成选择")
        print("  • 按R键重置")
        print("  • 按Q键取消并退出")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == 27:  # 空格键或ESC键完成选择
                if picked:
                    print(f"✅ 完成选择，共选择了 {len(picked)} 个颜色点")
                    break
                else:
                    print("⚠️ 请先选择至少一个颜色点")
            elif key == ord('q') or key == ord('Q'):  # Q键取消
                print("❌ 取消颜色选择")
                picked = []
                break
            elif key == ord('r'):  # R键重置
                picked = []
                display_img = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()
                cv2.imshow(window_name, display_img)
                print("🔄 已重置选择")
        
        cv2.destroyAllWindows()
        
        if picked:
            print(f"✅ 颜色选择完成！已选择 {len(picked)} 个颜色点")
            # 显示选择的颜色信息
            for i, color in enumerate(picked):
                print(f"  点{i+1}: HSV({color[0]}, {color[1]}, {color[2]})")
        else:
            print("❌ 未选择任何颜色，程序将退出")
        
        return picked
    
    def _create_mask(self, hsv: np.ndarray, picked_colors: list) -> np.ndarray:
        """创建颜色掩码"""
        hsv_arr = np.array(picked_colors)
        h, s, v = np.mean(hsv_arr, axis=0).astype(int)
        logger.info(f"HSV picked: {h}, {s}, {v}")
        
        tolerance = Config.DEFAULT_HSV_TOLERANCE
        
        lower = np.array([
            max(0, h - tolerance['h']), 
            max(0, s - tolerance['s']), 
            max(0, v - tolerance['v']-10)
        ])
        upper = np.array([
            min(179, h + tolerance['h']), 
            min(255, s + tolerance['s']+10), 
            min(255, v + tolerance['v'])
        ])
        
        logger.info(f"HSV范围 - lower: {lower}, upper: {upper}")
        return cv2.inRange(hsv, lower, upper)
    
    def _process_contours(self, mask: np.ndarray) -> tuple:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_contours = []
        all_contours = []
        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            max_area = max(areas)
            min_area = min(areas)
            if max_area > 0 and max_area / max(min_area, 1e-6) > 100:
                area_threshold = max_area / 100
                filtered = [(i, c) for i, c in enumerate(contours) if cv2.contourArea(c) >= area_threshold]
                contours = [c for i, c in filtered]
        # 获取图像shape
        image_shape = None
        if hasattr(self, 'current_image_path') and self.current_image_path is not None:
            img = cv2.imread(self.current_image_path)
            if img is not None:
                image_shape = img.shape
        for i, contour in enumerate(contours):
            if contour.shape[0] < Config.MIN_CONTOUR_POINTS:
                continue
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            points = contour[:, 0, :]
            contours = self.feature_extractor.extract_all_contours(contour, points, image_shape=image_shape)
            # if i == 1:  # 假设你要比对第1个色块
            #      print("【调试】当前色块特征：", contours)
            valid_contours.append({
                'contour': contour,
                'points': points,
                'area': area,
                'length': length,
                'idx': i,
                'bbox': cv2.boundingRect(contour),  # 添加包围框信息
                'contours': contours
            })
            all_contours.append(contours)
        return valid_contours, all_contours
    
    def _show_interactive_display(self, color_extract: np.ndarray, 
                             valid_contours: list, all_contours: list, matches):
        n_contours = len(valid_contours)
        linewidth = max(0.5, 2 - 0.03 * n_contours)
        show_legend = n_contours <= 15
        
        # 调整布局：删除色块放大视图，放大模板原图预览
        if self.templates:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 改为2x3布局
            # 重新分配子图
            ax_img, ax_fit, ax_template_preview = axes[0]  # 上排：颜色提取、轮廓显示、模板原图预览(放大)
            ax_db_matches, ax_stats, ax_history = axes[1]  # 下排：数据库匹配、统计、历史
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 改为1x3布局
            ax_img, ax_fit, ax_template_preview = axes
            ax_db_matches = ax_stats = ax_history = None
        
        # 设置各子图标题
        ax_img.set_title("颜色提取结果", fontproperties=myfont)
        ax_img.imshow(cv2.cvtColor(color_extract, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
        
        ax_fit.set_title("轮廓显示", fontproperties=myfont)
        ax_fit.axis('equal')
        ax_fit.invert_yaxis()
        ax_fit.grid(True)
        
        # 放大的模板原图预览区
        ax_template_preview.set_title("模板原图预览", fontproperties=myfont, fontsize=14)
        ax_template_preview.axis('off')
        
        # 初始化数据库匹配信息
        if self.templates:
            if ax_db_matches is not None:
                ax_db_matches.set_title("数据库匹配结果", fontproperties=myfont)
                ax_db_matches.axis('off')
            if ax_stats is not None:
                ax_stats.set_title("模板库统计", fontproperties=myfont)
                ax_stats.axis('off')
            if ax_history is not None:
                ax_history.set_title("匹配历史", fontproperties=myfont)
                ax_history.axis('off')
            
            # 显示模板库统计和尺度标定信息
            total_templates = len(self.templates)
            total_contours = sum(len(t) for t in self.templates.values())
            
            # 构建统计文本（包含尺度标定信息）
            stats_text = f"📊 系统状态:\n"
            stats_text += f"{'='*25}\n"
            
            # 尺度标定状态
            stats_text += f"🔍 尺度标定:\n"
            stats_text += f"  模式: {self.scale_mode}\n"
            
            if self.scale_mode == "auto":
                if self.calibration_result and self.calibration_result.pixel_per_mm > 0:
                    stats_text += f"  状态: ✅ 成功\n"
                    stats_text += f"  比例: {self.calibration_result.pixel_per_mm:.4f} px/mm\n"
                    stats_text += f"  置信度: {self.calibration_result.confidence:.3f}\n"
                    stats_text += f"  参考物尺寸: {self.reference_size_mm}mm\n"
                else:
                    stats_text += f"  状态: ❌ 失败\n"
                    if self.calibration_result:
                        stats_text += f"  错误: {self.calibration_result.error_message}\n"
            elif self.scale_mode == "manual":
                stats_text += f"  状态: 📏 手动设置\n"
                stats_text += f"  比例: {self.manual_pixel_per_mm:.4f} px/mm\n"
            else:
                stats_text += f"  状态: 🔄 传统模式\n"
            
            stats_text += f"\n📚 模板库:\n"
            stats_text += f"  总模板数: {total_templates}\n"
            stats_text += f"  总轮廓数: {total_contours}\n"
            stats_text += f"  归一化: {'✅' if Config.ENABLE_SCALE_NORMALIZATION else '❌'}\n\n"
            
            stats_text += f"📋 模板列表:\n"
            for i, (template_id, data) in enumerate(list(self.templates.items())[:8]):
                stats_text += f"  {i+1}. {template_id} ({len(data)}个轮廓)\n"
            if total_templates > 8:
                stats_text += f"  ... 还有 {total_templates-8} 个模板\n"
            
            if ax_stats is not None:
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                             fontsize=9, verticalalignment='top', fontproperties=myfont)
        
        selected_idx = [0]
        self.match_highlight_idx = None
        self.highlight_template = None
        
        def draw_all(highlight_idx=None):
            print("[DRAW_ALL] 调用栈:")
            print("[DRAW_ALL] 当前高亮模板:", self.highlight_template)
            print("[DRAW_ALL] 当前match_highlight_idx:", self.match_highlight_idx)
            print("[DRAW_ALL] matches keys:", list(matches.keys()))
            key = f'query_{highlight_idx}'
            print("[DRAW_ALL] 当前色块key:", key, "匹配列表长度:", len(matches.get(key, [])))
            
            # 更新轮廓显示（移除ax_zoom参数）
            self._draw_contours_enhanced(ax_fit, valid_contours, all_contours, 
                                       highlight_idx, linewidth, show_legend, fig,
                                       ax_db_matches if self.templates else None, matches)
            
            # 更新放大的模板原图预览区
            ax_template_preview.clear()
            ax_template_preview.set_title("模板原图预览", fontproperties=myfont, fontsize=14)
            ax_template_preview.axis('off')
            
            if self.highlight_template is not None and self.templates:
                template_id, template_contour_idx = self.highlight_template
                print("[DRAW_ALL] 模板原图区高亮分支:", template_id, template_contour_idx)
                
                # 加载原图
                img_path = f"templates/images/{template_id}.png"
                print("[DRAW_ALL] 原图路径:", img_path)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax_template_preview.imshow(img_rgb)
                    
                    # 加载轮廓点
                    contour_json = f"templates/contours/{template_id}.json"
                    print("[DRAW_ALL] 轮廓json路径:", contour_json)
                    if os.path.exists(contour_json):
                        with open(contour_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'contours' in data and 0 <= template_contour_idx < len(data['contours']):
                            points = np.array(data['contours'][template_contour_idx]['points'])
                            print("[DRAW_ALL] points.shape:", points.shape)
                            try:
                                # 使用更明显的高亮效果
                                ax_template_preview.fill(points[:,0], points[:,1], 
                                                       color='red', alpha=0.6, zorder=10, 
                                                       label=f'匹配轮廓: {template_contour_idx+1}')
                                ax_template_preview.plot(points[:,0], points[:,1], 
                                                       color='darkred', linewidth=3, zorder=11)
                                print("[DRAW_ALL] 轮廓绘制完成")
                                
                                # 添加标注信息（黑色字体，无背景框）
                                center_x, center_y = np.mean(points, axis=0)
                                ax_template_preview.text(center_x, center_y, str(template_contour_idx+1), 
                                                       fontsize=16, fontweight='bold', 
                                                       color='black', ha='center', va='center', 
                                                       zorder=12)
                                
                            except Exception as e:
                                print("[DRAW_ALL] 轮廓绘制异常:", e)
                        else:
                            print("[DRAW_ALL] 轮廓点索引超界或无contours")
                    else:
                        print("[DRAW_ALL] 轮廓json文件不存在")
                    
                    # 添加模板信息标题
                    info_text = f"模板: {template_id}\n轮廓: {template_contour_idx+1}"
                    ax_template_preview.text(0.02, 0.98, info_text, 
                                           transform=ax_template_preview.transAxes,
                                           fontsize=12, fontweight='bold', color='blue',
                                           ha='left', va='top', fontproperties=myfont,
                                           bbox=dict(facecolor='white', alpha=0.8, 
                                                   edgecolor='blue', boxstyle='round,pad=0.3'))
                else:
                    print(f"[DRAW_ALL] 未找到原图: {img_path}")
                    ax_template_preview.text(0.5, 0.5, f"未找到模板图像\n{template_id}", 
                                           ha='center', va='center', fontsize=14, color='red',
                                           fontproperties=myfont)
            else:
                print("[DRAW_ALL] 无模板高亮分支")
                # 显示使用说明
                help_text = ("🦷 模板原图预览区\n\n"
                            "📖 使用方法:\n"
                            "• ←→ 切换色块\n"
                            "• ↓ 选择匹配项\n"
                            "• 点击匹配项查看模板\n\n"
                            "💡 此区域将显示匹配到的\n"
                            "模板原始图像和轮廓位置")
                ax_template_preview.text(0.5, 0.5, help_text, ha='center', va='center', 
                                       fontsize=12, color='gray', fontproperties=myfont,
                                       bbox=dict(facecolor='lightgray', alpha=0.3, 
                                               boxstyle='round,pad=0.5'))
            
            fig.canvas.draw_idle()
        
        def on_key(event):
            print(f"[ON_KEY] 按键: {event.key}, 当前选中色块: {selected_idx[0]}, match_highlight_idx: {self.match_highlight_idx}")
            
            if event.key == 'right':
                selected_idx[0] = (selected_idx[0] + 1) % n_contours
                self.match_highlight_idx = None
                self.highlight_template = None
                print(f"[ON_KEY] 切换到色块 {selected_idx[0]}")
                draw_all(highlight_idx=selected_idx[0])
                
            elif event.key == 'left':
                selected_idx[0] = (selected_idx[0] - 1) % n_contours
                self.match_highlight_idx = None
                self.highlight_template = None
                print(f"[ON_KEY] 切换到色块 {selected_idx[0]}")
                draw_all(highlight_idx=selected_idx[0])
                
            elif event.key in ['escape', 'up']:
                if self.match_highlight_idx is not None or self.highlight_template is not None:
                    self.match_highlight_idx = None
                    self.highlight_template = None
                    print("[ON_KEY] 取消匹配高亮，返回色块高亮")
                    draw_all(highlight_idx=selected_idx[0])
                
            elif event.key == 'down':
                current_key = f'query_{selected_idx[0]}'
                match_list = matches.get(current_key, [])
                
                if not match_list:
                    print(f"[ON_KEY] 色块 {selected_idx[0]} 无匹配项")
                    return
                
                if self.match_highlight_idx is None:
                    self.match_highlight_idx = 0
                    print(f"[ON_KEY] 选中第一个匹配项 (索引0)")
                else:
                    self.match_highlight_idx = (self.match_highlight_idx + 1) % len(match_list)
                    print(f"[ON_KEY] 切换到匹配项 {self.match_highlight_idx}")
                
                if 0 <= self.match_highlight_idx < len(match_list):
                    match = match_list[self.match_highlight_idx]
                    self.highlight_template = (match['template_id'], match['template_contour_idx'])
                    print(f"[ON_KEY] 设置高亮模板: {self.highlight_template}")
                
                draw_all(highlight_idx=selected_idx[0])
            
            elif event.key == 'q':
                print("[ON_KEY] 退出程序")
                plt.close()
            
            else:
                print(f"[ON_KEY] 未处理的按键: {event.key}")
        
        def on_db_match_click(event):
            if ax_db_matches is None or event.inaxes != ax_db_matches:
                return
            
            if not hasattr(self, '_db_match_line_boxes') or not self._db_match_line_boxes:
                return
            
            click_x, click_y = event.xdata, event.ydata
            if click_x is None or click_y is None:
                return
            
            for idx, (bbox, match_id) in enumerate(self._db_match_line_boxes):
                x0, y0, x1, y1 = bbox
                if x0 <= click_x <= x1 and y0 <= click_y <= y1:
                    self.highlight_template = match_id
                    self.match_highlight_idx = idx
                    draw_all(highlight_idx=selected_idx[0])
                    return
        
        def on_click(event):
            # 更新点击检测，移除ax_zoom相关判断
            if self.templates and event.inaxes not in [ax_img, ax_fit, ax_db_matches]:
                pass
            on_db_match_click(event)
        
        # 绑定事件
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # 初始显示
        draw_all(highlight_idx=selected_idx[0])
        
        plt.tight_layout()
        plt.show()
    
    def _draw_contours_enhanced(self, ax, valid_contours, all_contours, highlight_idx, 
                               linewidth, show_legend, fig, ax_db_matches, matches):
        """增强版轮廓绘制方法"""
        ax.clear()
        ax.set_title("轮廓显示", fontproperties=myfont)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(valid_contours)))
        
        # 绘制所有轮廓
        for i, contour_info in enumerate(valid_contours):
            contour = contour_info['contour']
            points = contour_info['points']
            area = contour_info['area']
            
            color = colors[i]
            alpha = 0.6 if i == highlight_idx else 0.4
            edge_alpha = 1.0 if i == highlight_idx else 0.7
            linewidth_current = linewidth * 3 if i == highlight_idx else linewidth * 2
            
            # 绘制填充轮廓（类似您图片中的效果）
            ax.fill(points[:, 0], points[:, 1], color=color, 
                   alpha=alpha, label=f'色块 {i+1} (面积:{area:.0f})')
            
            # 绘制轮廓边框
            ax.plot(points[:, 0], points[:, 1], color=color, 
                   linewidth=linewidth_current, alpha=edge_alpha)
            
            # 标注色块编号（黑色字体，无背景框）
            center = np.mean(points, axis=0)
            ax.text(center[0], center[1], str(i+1), 
                   fontsize=10, ha='center', va='center', 
                   fontweight='bold', color='black')
        
        # 高亮显示匹配模板轮廓（如果存在）
        if self.highlight_template and highlight_idx is not None:
            template_id, template_contour_idx = self.highlight_template
            
            # 在轮廓图上添加匹配指示
            if highlight_idx < len(valid_contours):
                contour_info = valid_contours[highlight_idx]
                points = contour_info['points']
                center = np.mean(points, axis=0)
                
                # 添加匹配指示标记（纯红圆点）
                ax.plot(center[0], center[1], 'o', markersize=2, 
                       color='red', 
                       label=f'匹配: {template_id}-{template_contour_idx+1}')
        
        if show_legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                     fontsize=8, prop=myfont)
        
        # 更新数据库匹配显示
        if ax_db_matches is not None and matches:
            self._update_db_matches_display(ax_db_matches, matches, highlight_idx)
    
    def _update_db_matches_display(self, ax, matches, highlight_idx):
        """更新数据库匹配显示"""
        ax.clear()
        ax.set_title("数据库匹配结果", fontproperties=myfont)
        ax.axis('off')
        
        if highlight_idx is None:
            ax.text(0.5, 0.5, "请选择一个色块查看匹配结果", 
                   ha='center', va='center', fontproperties=myfont,
                   transform=ax.transAxes)
            return
        
        query_key = f'query_{highlight_idx}'
        query_matches = matches.get(query_key, [])
        
        if not query_matches:
            ax.text(0.5, 0.5, f"色块 {highlight_idx+1} 无匹配结果", 
                   ha='center', va='center', fontproperties=myfont,
                   transform=ax.transAxes, color='red')
            return
        
        # 显示匹配结果
        y_pos = 0.95
        line_height = 0.08
        
        ax.text(0.05, y_pos, f"色块 {highlight_idx+1} 的匹配结果:", 
               fontsize=14, fontweight='bold', fontproperties=myfont,
               transform=ax.transAxes)
        y_pos -= line_height
        
        self._db_match_line_boxes = []  # 重置点击区域
        
        for i, match in enumerate(query_matches[:10]):  # 最多显示10个匹配
            similarity = match['similarity']
            template_id = match['template_id']
            template_idx = match['template_contour_idx']
            
            # 高亮当前选中的匹配项
            if i == self.match_highlight_idx:
                bg_color = 'yellow'
                text_color = 'black'
                alpha = 0.8
            else:
                bg_color = 'lightblue' if i % 2 == 0 else 'white'
                text_color = 'black'
                alpha = 0.3
            
            # 添加背景框
            bbox = dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=alpha)
            
            match_text = f"{i+1}. {template_id}-{template_idx+1}: {similarity:.3f}"
            text_obj = ax.text(0.05, y_pos, match_text, 
                              fontsize=10, fontproperties=myfont,
                              transform=ax.transAxes, color=text_color,
                              bbox=bbox)
            
            # 记录点击区域
            bbox_coords = (0.05, y_pos - line_height/2, 0.95, y_pos + line_height/2)
            match_id = (template_id, template_idx)
            self._db_match_line_boxes.append((bbox_coords, match_id))
            
            y_pos -= line_height
            
            if y_pos < 0.1:  # 避免超出显示区域
                break


class ToothAreaCalculator:
    """牙齿区域面积计算器"""
    
    def __init__(self, pixel_per_mm: float):
        self.pixel_per_mm = pixel_per_mm
        
    def calculate_tooth_area(self, image: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """计算牙齿（白色区域）面积并返回处理过程图像"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 不反转，直接使用二值图像（白色牙齿区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {
                    'total_area_pixels': 0,
                    'total_area_mm2': 0,
                    'total_perimeter_pixels': 0,
                    'total_perimeter_mm': 0,
                    'contour_count': 0,
                    'largest_area_pixels': 0,
                    'largest_area_mm2': 0,
                    'largest_perimeter_pixels': 0,
                    'largest_perimeter_mm': 0,
                    'error': 'No white tooth regions found'
                }, cleaned, image.copy()
            
            total_area_pixels = 0
            total_perimeter_pixels = 0
            largest_area_pixels = 0
            largest_perimeter_pixels = 0
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 牙齿应该较大
                    perimeter = cv2.arcLength(contour, True)
                    total_area_pixels += area
                    total_perimeter_pixels += perimeter
                    valid_contours.append(contour)
                    
                    if area > largest_area_pixels:
                        largest_area_pixels = area
                        largest_perimeter_pixels = perimeter
            
            total_area_mm2 = total_area_pixels / (self.pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (self.pixel_per_mm ** 2)
            total_perimeter_mm = total_perimeter_pixels / self.pixel_per_mm
            largest_perimeter_mm = largest_perimeter_pixels / self.pixel_per_mm
            
            # 创建结果图像
            result_image = image.copy()
            cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
                
                # 添加标注
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result_image, f"Area: {largest_area_mm2:.1f}mm²", 
                              (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_image, f"Perimeter: {largest_perimeter_mm:.1f}mm", 
                              (cx-50, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            result = {
                'total_area_pixels': total_area_pixels,
                'total_area_mm2': total_area_mm2,
                'total_perimeter_pixels': total_perimeter_pixels,
                'total_perimeter_mm': total_perimeter_mm,
                'contour_count': len(valid_contours),
                'largest_area_pixels': largest_area_pixels,
                'largest_area_mm2': largest_area_mm2,
                'largest_perimeter_pixels': largest_perimeter_pixels,
                'largest_perimeter_mm': largest_perimeter_mm,
                'pixel_per_mm': self.pixel_per_mm
            }
            
            return result, cleaned, result_image
            
        except Exception as e:
            logger.error(f"牙齿面积计算失败: {e}")
            return {
                'total_area_pixels': 0,
                'total_area_mm2': 0,
                'contour_count': 0,
                'largest_area_pixels': 0,
                'largest_area_mm2': 0,
                'error': str(e)
            }, image, image


class ToothMatcherGUI:
    """牙齿匹配器GUI界面 - 整合了模板匹配和面积分析功能"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦷 牙齿匹配与分析系统")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 核心组件
        self.tooth_matcher = ToothMatcher()
        
        # 数据存储
        self.current_image = None
        self.original_image = None
        self.current_image_path = None
        self.calibration_result = None
        self.area_result = None
        self.match_results = None
        self.valid_contours = None
        
        # 颜色选择相关数据
        self.selected_colors = []  # 存储选择的颜色点
        self.current_hsv = None  # 当前图像的HSV版本
        self.current_mask = None  # 当前生成的掩码
        self.hsv_tolerance = {'h': 15, 's': 60, 'v': 60}  # HSV容忍度
        
        # GUI配置变量
        self.reference_size = tk.DoubleVar(value=10.0)
        self.scale_mode = tk.StringVar(value="auto")
        self.similarity_threshold = tk.DoubleVar(value=0.99)
        self.enable_area_analysis = tk.BooleanVar(value=True)
        self.enable_color_selection = tk.BooleanVar(value=True)  # 启用颜色选择模式
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 控制面板
        self.setup_control_panel(main_frame)
        
        # 图像显示区域
        self.setup_image_panel(main_frame)
        
        # 结果显示区域
        self.setup_result_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 第一行：文件操作
        row1_frame = ttk.Frame(control_frame)
        row1_frame.grid(row=0, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(row1_frame, text="📁 选择图像", command=self.select_image).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(row1_frame, text="🔍 开始分析", command=self.start_analysis, style="Accent.TButton").grid(row=0, column=1, padx=(0, 10))
        ttk.Button(row1_frame, text="� 颜色选择分析", command=self.open_color_selection_tab).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(row1_frame, text="�🎯 详细匹配分析", command=self.start_detailed_analysis).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(row1_frame, text="💾 保存结果", command=self.save_results).grid(row=0, column=4, padx=(0, 10))
        
        # 状态标签
        self.status_label = ttk.Label(row1_frame, text="请选择图像文件开始分析", foreground="blue")
        self.status_label.grid(row=0, column=5, padx=(20, 0))
        
        # 第二行：参数配置
        row2_frame = ttk.Frame(control_frame)
        row2_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E))
        
        # 尺度标定设置
        ttk.Label(row2_frame, text="尺度模式:").grid(row=0, column=0, padx=(0, 5))
        scale_combo = ttk.Combobox(row2_frame, textvariable=self.scale_mode, width=10, state="readonly")
        scale_combo['values'] = ('auto', 'manual', 'traditional')
        scale_combo.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(row2_frame, text="参考物尺寸(mm):").grid(row=0, column=2, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.reference_size, width=8).grid(row=0, column=3, padx=(0, 10))
        
        # 相似度阈值
        ttk.Label(row2_frame, text="相似度阈值:").grid(row=0, column=4, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.similarity_threshold, width=8).grid(row=0, column=5, padx=(0, 10))
        
        # 面积分析开关
        ttk.Checkbutton(row2_frame, text="启用面积分析", variable=self.enable_area_analysis).grid(row=0, column=6, padx=(10, 0))
        
        # 颜色选择模式开关
        ttk.Checkbutton(row2_frame, text="颜色选择模式", variable=self.enable_color_selection).grid(row=0, column=7, padx=(10, 0))
        
    def setup_image_panel(self, parent):
        """设置图像显示面板"""
        image_frame = ttk.LabelFrame(parent, text="图像显示", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 创建Notebook用于多标签页显示
        self.image_notebook = ttk.Notebook(image_frame)
        self.image_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像标签页
        self.original_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.original_tab, text="原始图像")
        
        self.original_canvas = tk.Canvas(self.original_tab, bg='white', width=500, height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 颜色选择标签页（新增）
        self.setup_color_selection_tab()
        
        # 轮廓检测标签页
        self.contour_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.contour_tab, text="轮廓检测")
        
        self.contour_canvas = tk.Canvas(self.contour_tab, bg='white', width=500, height=400)
        self.contour_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 面积分析标签页
        self.area_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.area_tab, text="面积分析")
        
        self.area_canvas = tk.Canvas(self.area_tab, bg='white', width=500, height=400)
        self.area_canvas.pack(fill=tk.BOTH, expand=True)
        
    def setup_color_selection_tab(self):
        """设置颜色选择标签页"""
        # 创建颜色选择标签页
        self.color_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.color_tab, text="颜色选择")
        
        # 创建主容器（左右分割）
        main_container = ttk.Frame(self.color_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：matplotlib canvas
        left_frame = ttk.LabelFrame(main_container, text="图像选择区", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 创建matplotlib figure和canvas
        self.color_fig = Figure(figsize=(6, 5), dpi=80)
        self.color_ax = self.color_fig.add_subplot(111)
        self.color_ax.set_title("点击选择牙齿颜色", fontsize=12)
        self.color_ax.axis('off')
        
        # 嵌入matplotlib canvas到tkinter
        self.color_canvas = FigureCanvasTkAgg(self.color_fig, left_frame)
        self.color_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 绑定点击事件
        self.color_canvas.mpl_connect('button_press_event', self.on_color_click)
        
        # 添加控制按钮
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="🔄 重置选择", command=self.reset_color_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="✅ 完成选择", command=self.complete_color_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🎯 开始分析", command=self.start_color_based_analysis).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="🔍 预览效果", command=self.preview_color_analysis).pack(side=tk.LEFT, padx=(5, 0))
        
        # 右侧：控制面板
        right_frame = ttk.LabelFrame(main_container, text="选择控制", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.configure(width=300)
        
        # 已选择的颜色点显示
        colors_frame = ttk.LabelFrame(right_frame, text="📍 已选择的颜色点", padding="5")
        colors_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建颜色列表显示
        self.colors_listbox = tk.Listbox(colors_frame, height=6, font=("Consolas", 9))
        colors_scrollbar = ttk.Scrollbar(colors_frame, orient=tk.VERTICAL, command=self.colors_listbox.yview)
        self.colors_listbox.configure(yscrollcommand=colors_scrollbar.set)
        
        self.colors_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        colors_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右键菜单（删除颜色点）
        self.colors_listbox.bind("<Button-3>", self.show_color_context_menu)
        
        # HSV范围调整
        hsv_frame = ttk.LabelFrame(right_frame, text="🎨 HSV范围调整", padding="5")
        hsv_frame.pack(fill=tk.X, pady=(0, 10))
        
        # H容忍度
        ttk.Label(hsv_frame, text="H容忍度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.h_tolerance = tk.IntVar(value=self.hsv_tolerance['h'])
        h_scale = ttk.Scale(hsv_frame, from_=0, to=50, variable=self.h_tolerance, 
                           orient=tk.HORIZONTAL, command=self.update_mask_preview)
        h_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.h_label = ttk.Label(hsv_frame, text=str(self.h_tolerance.get()))
        self.h_label.grid(row=0, column=2, pady=2)
        
        # S容忍度
        ttk.Label(hsv_frame, text="S容忍度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.s_tolerance = tk.IntVar(value=self.hsv_tolerance['s'])
        s_scale = ttk.Scale(hsv_frame, from_=0, to=100, variable=self.s_tolerance,
                           orient=tk.HORIZONTAL, command=self.update_mask_preview)
        s_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.s_label = ttk.Label(hsv_frame, text=str(self.s_tolerance.get()))
        self.s_label.grid(row=1, column=2, pady=2)
        
        # V容忍度
        ttk.Label(hsv_frame, text="V容忍度:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.v_tolerance = tk.IntVar(value=self.hsv_tolerance['v'])
        v_scale = ttk.Scale(hsv_frame, from_=0, to=100, variable=self.v_tolerance,
                           orient=tk.HORIZONTAL, command=self.update_mask_preview)
        v_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.v_label = ttk.Label(hsv_frame, text=str(self.v_tolerance.get()))
        self.v_label.grid(row=2, column=2, pady=2)
        
        # 配置列权重
        hsv_frame.columnconfigure(1, weight=1)
        
        # 实时预览
        preview_frame = ttk.LabelFrame(right_frame, text="🔍 实时预览", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建预览的matplotlib canvas
        self.preview_fig = Figure(figsize=(3, 2.5), dpi=60)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax.set_title("掩码预览", fontsize=10)
        self.preview_ax.axis('off')
        
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 预览统计信息
        self.preview_stats_label = ttk.Label(preview_frame, text="选择颜色点开始预览", 
                                           font=("Arial", 9), foreground="gray")
        self.preview_stats_label.pack(pady=(5, 0))
        
    def setup_result_panel(self, parent):
        """设置结果显示面板"""
        result_frame = ttk.LabelFrame(parent, text="分析结果", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建Notebook用于多个结果标签页
        self.result_notebook = ttk.Notebook(result_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 模板匹配结果标签页
        self.match_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.match_tab, text="模板匹配")
        
        self.match_text = tk.Text(self.match_tab, height=12, width=60, font=("Consolas", 10))
        match_scrollbar = ttk.Scrollbar(self.match_tab, orient=tk.VERTICAL, command=self.match_text.yview)
        self.match_text.configure(yscrollcommand=match_scrollbar.set)
        
        self.match_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        match_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 标定结果标签页
        self.calib_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.calib_tab, text="尺度标定")
        
        self.calib_text = tk.Text(self.calib_tab, height=12, width=60, font=("Consolas", 10))
        calib_scrollbar = ttk.Scrollbar(self.calib_tab, orient=tk.VERTICAL, command=self.calib_text.yview)
        self.calib_text.configure(yscrollcommand=calib_scrollbar.set)
        
        self.calib_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        calib_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 面积计算结果标签页
        self.area_result_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.area_result_tab, text="面积分析")
        
        self.area_text = tk.Text(self.area_result_tab, height=12, width=60, font=("Consolas", 10))
        area_scrollbar = ttk.Scrollbar(self.area_result_tab, orient=tk.VERTICAL, command=self.area_text.yview)
        self.area_text.configure(yscrollcommand=area_scrollbar.set)
        
        self.area_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        area_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("错误", "无法读取图像文件")
                    return
                
                self.current_image = self.original_image.copy()
                self.current_image_path = file_path
                self.display_image(self.original_canvas, self.current_image)
                self.status_label.config(text=f"已加载: {Path(file_path).name}", foreground="green")
                
                # 生成HSV版本用于颜色选择
                self.current_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
                
                # 更新颜色选择标签页
                self.update_color_selection_display()
                
                # 清空结果显示
                self.clear_results()
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def start_analysis(self):
        """开始分析"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择图像文件")
            return
        
        # 在后台线程中执行分析
        self.status_label.config(text="正在分析...", foreground="orange")
        self.root.config(cursor="wait")
        
        thread = threading.Thread(target=self._analysis_worker)
        thread.daemon = True
        thread.start()
    
    def start_detailed_analysis(self):
        """启动详细的matplotlib交互分析"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择图像文件")
            return
        
        if not self.current_image_path:
            messagebox.showerror("错误", "图像路径无效")
            return
        
        # 确认启动详细分析
        result = messagebox.askyesno(
            "启动详细分析", 
            "即将启动matplotlib交互式分析界面。\n\n"
            "注意：\n"
            "• 这将打开一个新的matplotlib窗口\n"
            "• 需要手动选择颜色区域\n"
            "• 可以进行详细的模板匹配\n\n"
            "是否继续？"
        )
        
        if result:
            try:
                # 更新匹配器配置
                self.tooth_matcher.scale_mode = self.scale_mode.get()
                self.tooth_matcher.reference_size_mm = self.reference_size.get()
                self.tooth_matcher.reference_obj = ReferenceObject(size_mm=self.reference_size.get())
                self.tooth_matcher.reference_detector = ReferenceDetector(self.tooth_matcher.reference_obj)
                Config.SIMILARITY_THRESHOLD = self.similarity_threshold.get()
                
                # 在新线程中启动详细分析
                thread = threading.Thread(target=self._detailed_analysis_worker)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                messagebox.showerror("错误", f"启动详细分析失败: {str(e)}")
    
    def _detailed_analysis_worker(self):
        """详细分析工作线程"""
        try:
            # 执行完整的匹配分析（包括matplotlib交互）
            self.tooth_matcher.process_image(self.current_image_path)
            
        except Exception as e:
            logger.error(f"详细分析失败: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"详细分析失败: {str(e)}"))
    
    def _analysis_worker(self):
        """后台分析工作线程"""
        try:
            # 更新匹配器配置
            self.tooth_matcher.scale_mode = self.scale_mode.get()
            self.tooth_matcher.reference_size_mm = self.reference_size.get()
            self.tooth_matcher.reference_obj = ReferenceObject(size_mm=self.reference_size.get())
            self.tooth_matcher.reference_detector = ReferenceDetector(self.tooth_matcher.reference_obj)
            Config.SIMILARITY_THRESHOLD = self.similarity_threshold.get()
            
            # 重新加载模板库
            self.tooth_matcher.load_templates()
            
            # 如果启用了面积分析，则进行面积计算
            if self.enable_area_analysis.get():
                self._perform_area_analysis()
            else:
                # 如果没有启用面积分析，但有颜色选择，则基于颜色选择进行轮廓分析
                if hasattr(self, 'selected_colors') and self.selected_colors:
                    logger.info("🎨 基于颜色选择进行轮廓分析")
                    self._perform_user_selected_color_analysis_with_matching()
                else:
                    logger.info("⚠️ 未选择颜色且未启用面积分析，跳过轮廓检测")
            
            # 在主线程中更新UI
            self.root.after(0, self.update_gui_results)
            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"分析失败: {str(e)}", foreground="red"))
            self.root.after(0, lambda: self.root.config(cursor=""))
    
    def _perform_area_analysis(self):
        """执行面积分析 - 基于用户颜色选择或轮廓检测结果"""
        try:
            # 检测参考物
            reference_obj = ReferenceObject(size_mm=self.reference_size.get())
            detector = ReferenceDetector(reference_obj)
            self.calibration_result = detector.detect_reference_object(self.original_image)
            
            if self.calibration_result.pixel_per_mm <= 0:
                logger.warning(f"标定失败: {self.calibration_result.error_message}")
                return
            
            # 如果用户已经进行了颜色选择，基于选择结果计算面积
            if hasattr(self, 'selected_colors') and self.selected_colors:
                logger.info("📐 基于用户颜色选择计算面积")
                self._calculate_area_from_color_selection()
            elif hasattr(self, 'valid_contours') and self.valid_contours:
                logger.info("📐 基于检测到的轮廓计算面积")
                self._calculate_area_from_contours()
            else:
                # 否则使用传统的自动检测方法（Otsu阈值）
                logger.info("📐 使用自动检测方法计算面积")
                calculator = ToothAreaCalculator(self.calibration_result.pixel_per_mm)
                self.area_result, binary_image, result_image = calculator.calculate_tooth_area(self.original_image)
                
                # 保存图像以便在GUI中显示
                self.binary_image = binary_image
                self.area_result_image = result_image
            
            # 执行基础的轮廓检测以显示在GUI中（仅在没有用户选择时）
            if not (hasattr(self, 'selected_colors') and self.selected_colors):
                self._perform_basic_contour_analysis()
            
        except Exception as e:
            logger.error(f"面积分析失败: {e}")
    
    def _calculate_area_from_color_selection(self):
        """基于用户颜色选择计算面积"""
        try:
            # 重新生成掩码
            if hasattr(self, 'current_mask') and self.current_mask is not None:
                mask = self.current_mask
            else:
                # 如果没有current_mask，重新生成
                mask = self.generate_mask_from_selections()
            
            if mask is None:
                logger.error("无法生成颜色掩码")
                return
            
            # 基于掩码计算面积
            total_area_pixels = np.sum(mask == 255)
            total_perimeter_pixels = 0
            
            # 检测轮廓以计算周长
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            # 构建与ToothMatcher兼容的轮廓数据结构
            valid_contours = []
            for i, contour in enumerate(raw_contours):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                bbox = cv2.boundingRect(contour)
                points = len(contour)
                
                valid_contours.append({
                    'contour': contour,
                    'points': points,
                    'area': area,
                    'length': perimeter,
                    'idx': i,
                    'bbox': bbox,
                    'contours': raw_contours
                })
            
            # 设置GUI需要的轮廓数据
            self.valid_contours = valid_contours
            
            largest_area_pixels = 0
            largest_perimeter_pixels = 0
            
            for contour_info in valid_contours:
                contour = contour_info['contour']
                perimeter = contour_info['length']
                total_perimeter_pixels += perimeter
                
                area = contour_info['area']
                if area > largest_area_pixels:
                    largest_area_pixels = area
                    largest_perimeter_pixels = perimeter
            
            # 转换为毫米单位
            pixel_per_mm = self.calibration_result.pixel_per_mm
            total_area_mm2 = total_area_pixels / (pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (pixel_per_mm ** 2)
            total_perimeter_mm = total_perimeter_pixels / pixel_per_mm
            largest_perimeter_mm = largest_perimeter_pixels / pixel_per_mm
            
            # 创建结果图像  
            result_image = self.original_image.copy()
            if valid_contours:
                contour_list = [c['contour'] for c in valid_contours]
                cv2.drawContours(result_image, contour_list, -1, (0, 255, 0), 2)
                
                if valid_contours:
                    largest_contour_info = max(valid_contours, key=lambda x: x['area'])
                    cv2.drawContours(result_image, [largest_contour_info['contour']], -1, (0, 0, 255), 3)
            
            self.area_result = {
                'total_area_pixels': total_area_pixels,
                'total_area_mm2': total_area_mm2,
                'total_perimeter_pixels': total_perimeter_pixels,
                'total_perimeter_mm': total_perimeter_mm,
                'contour_count': len(valid_contours),
                'largest_area_pixels': largest_area_pixels,
                'largest_area_mm2': largest_area_mm2,
                'largest_perimeter_pixels': largest_perimeter_pixels,
                'largest_perimeter_mm': largest_perimeter_mm,
                'pixel_per_mm': pixel_per_mm
            }
            
            self.binary_image = mask
            self.area_result_image = result_image
            
            logger.info(f"✅ 基于颜色选择计算面积: {total_area_mm2:.2f} mm²，{len(valid_contours)} 个轮廓")
            
        except Exception as e:
            logger.error(f"基于颜色选择计算面积失败: {e}")
    
    def _calculate_area_from_contours(self):
        """基于检测到的轮廓计算面积"""
        try:
            if not hasattr(self, 'valid_contours') or not self.valid_contours:
                logger.warning("没有有效轮廓用于面积计算")
                return
            
            total_area_pixels = 0
            total_perimeter_pixels = 0
            largest_area_pixels = 0
            largest_perimeter_pixels = 0
            
            contours_list = []
            for contour_info in self.valid_contours:
                contour = contour_info['contour']
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                total_area_pixels += area
                total_perimeter_pixels += perimeter
                contours_list.append(contour)
                
                if area > largest_area_pixels:
                    largest_area_pixels = area
                    largest_perimeter_pixels = perimeter
            
            # 转换为毫米单位
            pixel_per_mm = self.calibration_result.pixel_per_mm
            total_area_mm2 = total_area_pixels / (pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (pixel_per_mm ** 2)
            total_perimeter_mm = total_perimeter_pixels / pixel_per_mm
            largest_perimeter_mm = largest_perimeter_pixels / pixel_per_mm
            
            # 创建结果图像
            result_image = self.original_image.copy()
            cv2.drawContours(result_image, contours_list, -1, (0, 255, 0), 2)
            
            if contours_list:
                largest_contour = max(contours_list, key=cv2.contourArea)
                cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
            
            self.area_result = {
                'total_area_pixels': total_area_pixels,
                'total_area_mm2': total_area_mm2,
                'total_perimeter_pixels': total_perimeter_pixels,
                'total_perimeter_mm': total_perimeter_mm,
                'contour_count': len(self.valid_contours),
                'largest_area_pixels': largest_area_pixels,
                'largest_area_mm2': largest_area_mm2,
                'largest_perimeter_pixels': largest_perimeter_pixels,
                'largest_perimeter_mm': largest_perimeter_mm,
                'pixel_per_mm': pixel_per_mm
            }
            
            self.binary_image = None  # 没有二值图像
            self.area_result_image = result_image
            
            logger.info(f"✅ 基于轮廓计算面积: {total_area_mm2:.2f} mm²，{len(self.valid_contours)} 个轮廓")
            
        except Exception as e:
            logger.error(f"基于轮廓计算面积失败: {e}")
    
    def _perform_basic_contour_analysis(self):
        """执行基础轮廓检测用于GUI显示 - 改进版本，调用ToothMatcher核心功能"""
        try:
            # 如果用户已选择颜色，使用选择的颜色进行分析
            if hasattr(self, 'selected_colors') and self.selected_colors:
                logger.info("🎨 使用用户选择的颜色进行轮廓分析")
                self._perform_user_selected_color_analysis_with_matching()
            elif hasattr(self, 'picked_colors') and self.picked_colors:
                logger.info("🎨 使用用户选择的颜色进行轮廓分析(picked_colors)")
                self._perform_user_selected_color_analysis_with_matching()
            else:
                # 否则使用智能自适应方法
                logger.info("🤖 使用智能自适应方法进行轮廓分析")
                self._perform_smart_color_analysis_with_matching()
                
        except Exception as e:
            logger.error(f"基础轮廓分析失败: {e}")
            # 使用原图作为备选方案
            self.contour_image = self.original_image
            if hasattr(self, 'selected_colors') and self.selected_colors:
                self._perform_user_selected_color_analysis_with_matching()
            else:
                # 使用智能自适应颜色检测替代硬编码值
                self._perform_smart_color_analysis_with_matching()
            
        except Exception as e:
            logger.error(f"基础轮廓分析失败: {e}")
            # 如果失败，使用原图作为轮廓图
            self.contour_image = self.original_image
    
    def _perform_user_selected_color_analysis_with_matching(self):
        """使用用户选择的颜色进行分析 - 集成ToothMatcher核心功能"""
        if not hasattr(self, 'selected_colors') or not self.selected_colors:
            self._perform_smart_color_analysis_with_matching()
            return
            
        try:
            logger.info(f"🎨 使用用户选择的 {len(self.selected_colors)} 个颜色进行完整轮廓分析")
            
            # 模拟ToothMatcher的颜色选择结果
            picked_colors = []
            for color_info in self.selected_colors:
                # color_info是字典格式：{'position': (x, y), 'hsv': [h, s, v], 'timestamp': time}
                picked_colors.append(np.array(color_info['hsv']))
            
            # 调用ToothMatcher的核心逻辑
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            
            # 使用ToothMatcher的_create_mask方法
            mask = self.tooth_matcher._create_mask(hsv, picked_colors)
            
            # 生成颜色提取图像
            self.color_extract = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            
            # 调用ToothMatcher的_process_contours方法进行轮廓检测和特征提取
            self.valid_contours, all_contours = self.tooth_matcher._process_contours(mask)
            
            if self.valid_contours:
                logger.info(f"✅ 检测到 {len(self.valid_contours)} 个有效轮廓")
                
                # 提取特征列表用于模板匹配
                query_features_list = []
                for contour_info in self.valid_contours:
                    original_features = contour_info['contours']
                    
                    # 应用尺度归一化（如果可用）
                    effective_pixel_per_mm = self.tooth_matcher.get_effective_pixel_per_mm()
                    if effective_pixel_per_mm and Config.ENABLE_SCALE_NORMALIZATION:
                        normalized_features = self.tooth_matcher.normalize_features_by_scale(
                            original_features, effective_pixel_per_mm)
                    else:
                        normalized_features = original_features.copy()
                        normalized_features['_scale_normalized'] = False
                    
                    query_features_list.append(normalized_features)
                    # 更新轮廓信息
                    contour_info['normalized_contours'] = normalized_features
                
                # 与数据库进行模板匹配
                self.match_results = self.tooth_matcher.match_against_database(query_features_list)
                
                # 生成带轮廓的可视化图像
                self.contour_image = self._create_contour_visualization()
                
            else:
                logger.warning("⚠️ 未检测到有效轮廓")
                self.contour_image = self.color_extract
                self.match_results = {}
            
        except Exception as e:
            logger.error(f"用户颜色选择分析失败: {e}")
            self.contour_image = self.original_image
            self.match_results = {}
    
    def _perform_smart_color_analysis_with_matching(self):
        """智能自适应颜色检测 - 集成ToothMatcher核心功能"""
        try:
            logger.info("🤖 使用智能自适应模式进行完整轮廓分析")
            
            # 方法1：使用Otsu阈值分割
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作清理噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 生成颜色提取图像
            self.color_extract = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            
            # 调用ToothMatcher的_process_contours方法进行轮廓检测和特征提取
            self.valid_contours, all_contours = self.tooth_matcher._process_contours(mask)
            
            if self.valid_contours:
                logger.info(f"✅ 检测到 {len(self.valid_contours)} 个有效轮廓")
                
                # 提取特征列表用于模板匹配
                query_features_list = []
                for contour_info in self.valid_contours:
                    original_features = contour_info['contours']
                    
                    # 应用尺度归一化（如果可用）
                    effective_pixel_per_mm = self.tooth_matcher.get_effective_pixel_per_mm()
                    if effective_pixel_per_mm and Config.ENABLE_SCALE_NORMALIZATION:
                        normalized_features = self.tooth_matcher.normalize_features_by_scale(
                            original_features, effective_pixel_per_mm)
                    else:
                        normalized_features = original_features.copy()
                        normalized_features['_scale_normalized'] = False
                    
                    query_features_list.append(normalized_features)
                    # 更新轮廓信息
                    contour_info['normalized_contours'] = normalized_features
                
                # 与数据库进行模板匹配
                self.match_results = self.tooth_matcher.match_against_database(query_features_list)
                
                # 生成带轮廓的可视化图像
                self.contour_image = self._create_contour_visualization()
                
            else:
                logger.warning("⚠️ 未检测到有效轮廓")
                self.contour_image = self.color_extract
                self.match_results = {}
            
        except Exception as e:
            logger.error(f"智能颜色分析失败: {e}")
            # 最后备选方案：使用原图
            self.contour_image = self.original_image
            self.match_results = {}
    
    def _create_contour_visualization(self):
        """创建轮廓可视化图像，用于GUI显示"""
        try:
            # 基于原图创建可视化图像
            vis_image = self.original_image.copy()
            
            if self.valid_contours:
                for idx, contour_info in enumerate(self.valid_contours):
                    contour = contour_info['contour']
                    bbox = contour_info['bbox']
                    
                    # 绘制轮廓（绿色）
                    cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
                    
                    # 绘制包围框（蓝色）
                    x, y, w, h = bbox
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    
                    # 添加索引标签
                    cv2.putText(vis_image, f"#{idx+1}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                logger.info(f"✅ 已创建包含 {len(self.valid_contours)} 个轮廓的可视化图像")
            else:
                logger.warning("⚠️ 无有效轮廓用于可视化")
            
            return vis_image
            
        except Exception as e:
            logger.error(f"创建轮廓可视化失败: {e}")
            return self.original_image
    
    def update_gui_results(self):
        """更新GUI结果显示"""
        try:
            # 显示轮廓图像（如果有的话）
            if hasattr(self, 'contour_image'):
                self.display_image(self.contour_canvas, self.contour_image)
            
            # 显示面积分析图像
            if hasattr(self, 'area_result_image'):
                self.display_image(self.area_canvas, self.area_result_image)
            
            # 显示各种结果
            self.display_match_results()
            self.display_calibration_results()
            self.display_area_results()
            
            self.status_label.config(text="GUI分析完成", foreground="green")
            self.root.config(cursor="")
            
        except Exception as e:
            logger.error(f"更新GUI结果失败: {e}")
            self.status_label.config(text=f"更新结果失败: {str(e)}", foreground="red")
            self.root.config(cursor="")
    
    def display_image(self, canvas, image, is_gray=False):
        """在Canvas上显示图像"""
        try:
            if image is None:
                return
                
            if is_gray and len(image.shape) == 2:
                # 灰度图像
                pil_image = Image.fromarray(image)
            else:
                # 彩色图像，从BGR转换为RGB
                if len(image.shape) == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                else:
                    pil_image = Image.fromarray(image)
            
            # 调整图像大小以适应Canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas尚未初始化，使用默认大小
                canvas_width, canvas_height = 500, 400
            
            img_width, img_height = pil_image.size
            
            # 计算缩放比例
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h) * 0.9  # 留一些边距
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 清空Canvas并显示图像
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
            
            # 保存引用以防止垃圾回收
            canvas.image = photo
            
        except Exception as e:
            logger.error(f"显示图像失败: {e}")
    
    def display_match_results(self):
        """显示模板匹配结果"""
        self.match_text.delete(1.0, tk.END)
        
        result_text = "🔍 模板匹配分析结果\n"
        result_text += "=" * 40 + "\n\n"
        
        if hasattr(self.tooth_matcher, 'templates') and self.tooth_matcher.templates:
            result_text += f"📚 模板库信息:\n"
            result_text += f"   模板总数: {len(self.tooth_matcher.templates)}\n"
            
            total_contours = sum(len(data) for data in self.tooth_matcher.templates.values())
            result_text += f"   轮廓总数: {total_contours}\n\n"
            
            result_text += f"⚙️ 分析设置:\n"
            result_text += f"   尺度模式: {self.scale_mode.get()}\n"
            result_text += f"   相似度阈值: {self.similarity_threshold.get():.3f}\n"
            result_text += f"   尺度归一化: {'启用' if Config.ENABLE_SCALE_NORMALIZATION else '禁用'}\n\n"
            
            result_text += "💡 使用说明:\n"
            result_text += "   - 使用matplotlib界面进行详细的交互式匹配分析\n"
            result_text += "   - 键盘控制: ←→ 切换轮廓, ↓ 查看匹配项\n"
            result_text += "   - 点击匹配项可查看模板详情\n"
        else:
            result_text += "❌ 未找到模板库\n"
            result_text += "💡 请检查模板文件是否存在于 templates/features/ 目录\n"
        
        self.match_text.insert(tk.END, result_text)
    
    def display_calibration_results(self):
        """显示标定结果"""
        self.calib_text.delete(1.0, tk.END)
        
        if self.calibration_result:
            result_text = f"""🎯 参考物检测结果
{'='*40}
✅ 检测状态: 成功
📏 参考物尺寸: {self.reference_size.get():.1f} mm
📐 像素尺寸: {self.calibration_result.reference_pixel_size:.2f} pixels
🔄 比例系数: {self.calibration_result.pixel_per_mm:.4f} px/mm
🎯 置信度: {self.calibration_result.confidence:.3f} ({self.calibration_result.confidence*100:.1f}%)

📍 参考物位置:
   X: {self.calibration_result.reference_position[0]}
   Y: {self.calibration_result.reference_position[1]} 
   宽度: {self.calibration_result.reference_position[2]}
   高度: {self.calibration_result.reference_position[3]}
"""
        else:
            result_text = "❌ 暂无标定结果\n"
            result_text += "💡 请启用面积分析功能进行自动标定"
        
        self.calib_text.insert(tk.END, result_text)
    
    def display_area_results(self):
        """显示面积计算结果 - 改进版本，强调单位"""
        self.area_text.delete(1.0, tk.END)
        
        if self.area_result and 'error' not in self.area_result:
            # 获取关键数据
            total_area_mm2 = self.area_result['total_area_mm2']
            total_area_pixels = self.area_result['total_area_pixels']
            largest_area_mm2 = self.area_result['largest_area_mm2']
            pixel_per_mm = self.area_result['pixel_per_mm']
            contour_count = self.area_result['contour_count']
            
            result_text = f"""🦷 牙齿面积和周长分析结果
{'='*50}

� 【面积测量结果】（平方毫米 mm²）
   ✅ 总牙齿面积: {total_area_mm2:.2f} mm²
   ✅ 最大牙齿面积: {largest_area_mm2:.2f} mm²
   📊 平均牙齿面积: {(total_area_mm2/contour_count) if contour_count > 0 else 0:.2f} mm²

� 【周长测量结果】（毫米 mm）
   ✅ 总牙齿周长: {self.area_result.get('total_perimeter_mm', 0):.2f} mm
   ✅ 最大牙齿周长: {self.area_result.get('largest_perimeter_mm', 0):.2f} mm
   📊 平均牙齿周长: {(self.area_result.get('total_perimeter_mm', 0)/contour_count) if contour_count > 0 else 0:.2f} mm

🔢 【检测统计】
   检测到的牙齿数量: {contour_count}
   最大牙齿占比: {(largest_area_mm2/total_area_mm2*100) if total_area_mm2 > 0 else 0:.1f}%

🔧 【技术参数】
   像素面积（原始值）: {total_area_pixels:.0f} pixels
   像素周长（原始值）: {self.area_result.get('total_perimeter_pixels', 0):.0f} pixels
   比例系数: {pixel_per_mm:.4f} pixels/mm
   
📊 【换算说明】
   1 mm² = {pixel_per_mm**2:.1f} pixels²
   面积换算: {total_area_pixels:.0f} pixels² ÷ {pixel_per_mm**2:.1f} = {total_area_mm2:.2f} mm²

💡 【结果说明】
   - 绿色轮廓: 所有检测到的牙齿区域
   - 红色轮廓: 最大的牙齿
   - ⚠️  注意: 结果以毫米单位显示，基于参考物标定
   - 📏 参考: 标准牙齿面积通常在 {total_area_mm2/contour_count if contour_count > 0 else 50:.0f} mm² 左右
"""
        elif self.area_result and 'error' in self.area_result:
            result_text = f"""❌ 面积计算失败
{'='*30}
错误信息: {self.area_result['error']}

💡 可能的解决方案:
   1. 检查图像中是否有红色参考物
   2. 确保参考物大小设置正确
   3. 检查图像质量和光照条件
"""
        else:
            result_text = """❌ 暂无面积计算结果
{'='*30}
💡 使用说明:
   1. 首先加载图像
   2. 启用面积分析功能 ✓
   3. 点击"开始分析"按钮
   4. 确保图像中包含红色参考物
"""
        
        self.area_text.insert(tk.END, result_text)
    
    def clear_results(self):
        """清空结果显示"""
        self.match_text.delete(1.0, tk.END)
        self.calib_text.delete(1.0, tk.END)
        self.area_text.delete(1.0, tk.END)
        
        # 清空颜色选择相关数据
        self.selected_colors = []
        self.current_mask = None
        
        # 更新颜色选择预览
        if hasattr(self, 'colors_listbox'):
            self.colors_listbox.delete(0, tk.END)
        if hasattr(self, 'preview_stats_label'):
            self.preview_stats_label.config(text="选择颜色点开始预览")
        
        # 清空图像显示
        for canvas in [self.contour_canvas, self.area_canvas]:
            canvas.delete("all")
            if hasattr(canvas, 'image'):
                del canvas.image
    
    def save_results(self):
        """保存分析结果"""
        if not hasattr(self, 'area_result') and not hasattr(self, 'match_results'):
            messagebox.showwarning("警告", "请先进行分析")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".json",
            filetypes=[
                ("JSON文件", "*.json"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 准备保存的数据
                save_data = {
                    "image_path": self.current_image_path,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "settings": {
                        "scale_mode": self.scale_mode.get(),
                        "reference_size_mm": self.reference_size.get(),
                        "similarity_threshold": self.similarity_threshold.get(),
                        "enable_area_analysis": self.enable_area_analysis.get()
                    }
                }
                
                if hasattr(self, 'calibration_result') and self.calibration_result:
                    save_data["calibration_results"] = {
                        "pixel_per_mm": self.calibration_result.pixel_per_mm,
                        "reference_pixel_size": self.calibration_result.reference_pixel_size,
                        "reference_position": self.calibration_result.reference_position,
                        "confidence": self.calibration_result.confidence
                    }
                
                if hasattr(self, 'area_result') and self.area_result:
                    save_data["area_results"] = self.area_result
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                else:
                    # 保存为文本格式
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("🦷 牙齿匹配与分析系统报告\n")
                        f.write("="*50 + "\n\n")
                        
                        f.write(f"📸 图像路径: {self.current_image_path}\n")
                        f.write(f"🕒 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        f.write("⚙️ 分析设置:\n")
                        f.write(f"   尺度模式: {self.scale_mode.get()}\n")
                        f.write(f"   参考物尺寸: {self.reference_size.get():.1f} mm\n")
                        f.write(f"   相似度阈值: {self.similarity_threshold.get():.3f}\n")
                        f.write(f"   面积分析: {'启用' if self.enable_area_analysis.get() else '禁用'}\n\n")
                        
                        if hasattr(self, 'calibration_result') and self.calibration_result:
                            f.write("📏 标定结果:\n")
                            f.write(f"   比例系数: {self.calibration_result.pixel_per_mm:.4f} px/mm\n")
                            f.write(f"   置信度: {self.calibration_result.confidence:.3f}\n\n")
                        
                        if hasattr(self, 'area_result') and self.area_result:
                            f.write("📐 面积分析结果:\n")
                            f.write(f"   总面积: {self.area_result['total_area_mm2']:.2f} mm²\n")
                            f.write(f"   最大区域: {self.area_result['largest_area_mm2']:.2f} mm²\n")
                            f.write(f"   区域数量: {self.area_result['contour_count']}\n")
                
                messagebox.showinfo("成功", f"结果已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    # ===== 颜色选择相关方法 =====
    
    def open_color_selection_tab(self):
        """打开颜色选择标签页"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择图像文件")
            return
        
        # 切换到颜色选择标签页
        self.image_notebook.select(self.color_tab)
        self.status_label.config(text="请在图像上点击选择牙齿颜色", foreground="blue")
    
    def on_color_click(self, event):
        """处理颜色选择点击事件"""
        if event.inaxes != self.color_ax or self.current_hsv is None:
            return
        
        # 获取点击坐标
        x, y = int(event.xdata), int(event.ydata)
        
        # 确保坐标在图像范围内
        h, w = self.current_hsv.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # 提取HSV值
            hsv_value = self.current_hsv[y, x]
            
            # 添加到选择列表
            color_info = {
                'position': (x, y),
                'hsv': hsv_value.tolist(),
                'timestamp': time.time()
            }
            self.selected_colors.append(color_info)
            
            # 更新显示
            self.update_color_selection_display()
            self.update_mask_preview()
            
            logger.info(f"选择颜色点 ({x},{y}): HSV{hsv_value}")
    
    def update_color_selection_display(self):
        """更新颜色选择显示"""
        if self.original_image is None:
            return
            
        # 清空并重新绘制图像
        self.color_ax.clear()
        
        # 显示原始图像
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.color_ax.imshow(rgb_image)
        self.color_ax.set_title(f"点击选择牙齿颜色 (已选择 {len(self.selected_colors)} 个点)", fontsize=12)
        self.color_ax.axis('off')
        
        # 标记选择的点
        for i, color_info in enumerate(self.selected_colors):
            x, y = color_info['position']
            # 绘制绿色圆圈
            circle = plt.Circle((x, y), 5, color='lime', fill=False, linewidth=2)
            self.color_ax.add_patch(circle)
            # 添加序号
            self.color_ax.text(x+8, y-8, str(i+1), color='lime', fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        self.color_canvas.draw()
        
        # 更新颜色列表
        self.colors_listbox.delete(0, tk.END)
        for i, color_info in enumerate(self.selected_colors):
            hsv = color_info['hsv']
            self.colors_listbox.insert(tk.END, f"点{i+1}: HSV({hsv[0]:3d},{hsv[1]:3d},{hsv[2]:3d})")
    
    def update_mask_preview(self, *args):
        """更新掩码预览"""
        if not self.selected_colors or self.current_hsv is None:
            # 清空预览
            self.preview_ax.clear()
            self.preview_ax.set_title("掩码预览", fontsize=10)
            self.preview_ax.text(0.5, 0.5, "请先选择颜色点", ha='center', va='center', 
                               transform=self.preview_ax.transAxes, fontsize=12, color='gray')
            self.preview_ax.axis('off')
            self.preview_canvas.draw()
            self.preview_stats_label.config(text="选择颜色点开始预览")
            return
        
        # 更新HSV容忍度数值显示
        self.h_label.config(text=str(self.h_tolerance.get()))
        self.s_label.config(text=str(self.s_tolerance.get()))
        self.v_label.config(text=str(self.v_tolerance.get()))
        
        # 更新容忍度字典
        self.hsv_tolerance = {
            'h': self.h_tolerance.get(),
            's': self.s_tolerance.get(),
            'v': self.v_tolerance.get()
        }
        
        # 生成掩码
        self.current_mask = self.generate_mask_from_selections()
        
        if self.current_mask is not None:
            # 显示掩码预览
            self.preview_ax.clear()
            self.preview_ax.imshow(self.current_mask, cmap='gray')
            self.preview_ax.set_title("掩码预览", fontsize=10)
            self.preview_ax.axis('off')
            self.preview_canvas.draw()
            
            # 计算统计信息
            total_pixels = self.current_mask.shape[0] * self.current_mask.shape[1]
            selected_pixels = np.sum(self.current_mask == 255)
            coverage = (selected_pixels / total_pixels) * 100
            
            # 更新统计信息
            stats_text = f"覆盖率: {coverage:.1f}%\n选中像素: {selected_pixels:,}\n总像素: {total_pixels:,}"
            self.preview_stats_label.config(text=stats_text)
    
    def generate_mask_from_selections(self):
        """根据选择的颜色生成掩码"""
        if not self.selected_colors or self.current_hsv is None:
            return None
        
        try:
            # 计算所有选择颜色的平均HSV值
            hsv_values = np.array([color['hsv'] for color in self.selected_colors])
            h_mean, s_mean, v_mean = np.mean(hsv_values, axis=0).astype(int)
            
            # 应用容忍度
            lower = np.array([
                max(0, h_mean - self.hsv_tolerance['h']),
                max(0, s_mean - self.hsv_tolerance['s']),
                max(0, v_mean - self.hsv_tolerance['v'] - 10)
            ])
            upper = np.array([
                min(179, h_mean + self.hsv_tolerance['h']),
                min(255, s_mean + self.hsv_tolerance['s'] + 10),
                min(255, v_mean + self.hsv_tolerance['v'])
            ])
            
            # 生成掩码
            mask = cv2.inRange(self.current_hsv, lower, upper)
            
            logger.info(f"生成掩码: 平均HSV({h_mean},{s_mean},{v_mean}), "
                       f"范围 lower{lower} upper{upper}")
            
            return mask
            
        except Exception as e:
            logger.error(f"生成掩码失败: {e}")
            return None
    
    def reset_color_selection(self):
        """重置颜色选择"""
        self.selected_colors = []
        self.current_mask = None
        self.update_color_selection_display()
        self.update_mask_preview()
        logger.info("已重置颜色选择")
    
    def complete_color_selection(self):
        """完成颜色选择"""
        if not self.selected_colors:
            messagebox.showwarning("警告", "请先选择至少一个颜色点")
            return
        
        # 生成最终掩码
        final_mask = self.generate_mask_from_selections()
        if final_mask is not None:
            # 显示成功消息
            coverage = (np.sum(final_mask == 255) / (final_mask.shape[0] * final_mask.shape[1])) * 100
            messagebox.showinfo("完成", f"颜色选择完成！\n"
                                       f"选择了 {len(self.selected_colors)} 个颜色点\n"
                                       f"掩码覆盖率: {coverage:.1f}%\n\n"
                                       f"现在可以进行轮廓分析")
            
            # 切换到轮廓检测标签页
            self.image_notebook.select(self.contour_tab)
            
        logger.info(f"完成颜色选择: {len(self.selected_colors)} 个点")
    
    def preview_color_analysis(self):
        """预览颜色分析效果"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        try:
            # 执行轮廓分析预览
            self._perform_basic_contour_analysis()
            
            # 显示预览结果
            if hasattr(self, 'contour_image'):
                self.display_image(self.contour_canvas, self.contour_image)
                
                # 更新状态信息
                if hasattr(self, 'selected_colors') and self.selected_colors:
                    mode_text = f"用户选择（{len(self.selected_colors)}个颜色点）"
                else:
                    mode_text = "智能自适应检测"
                    
                self.status_label.config(text=f"预览完成 - 使用{mode_text}", foreground="blue")
                messagebox.showinfo("预览完成", f"轮廓分析预览已更新\n分析模式: {mode_text}")
            
        except Exception as e:
            logger.error(f"预览失败: {e}")
            messagebox.showerror("预览失败", f"无法生成预览: {str(e)}")
    
    def start_color_based_analysis(self):
        """基于颜色选择开始分析"""
        if not self.selected_colors or self.current_mask is None:
            messagebox.showwarning("警告", "请先完成颜色选择")
            return
        
        # 在后台线程中执行分析
        self.status_label.config(text="正在基于颜色选择进行分析...", foreground="orange")
        self.root.config(cursor="wait")
        
        thread = threading.Thread(target=self._color_based_analysis_worker)
        thread.daemon = True
        thread.start()
    
    def _color_based_analysis_worker(self):
        """基于颜色选择的分析工作线程"""
        try:
            # 更新匹配器配置
            self.tooth_matcher.scale_mode = self.scale_mode.get()
            self.tooth_matcher.reference_size_mm = self.reference_size.get()
            self.tooth_matcher.reference_obj = ReferenceObject(size_mm=self.reference_size.get())
            self.tooth_matcher.reference_detector = ReferenceDetector(self.tooth_matcher.reference_obj)
            Config.SIMILARITY_THRESHOLD = self.similarity_threshold.get()
            
            # 重新加载模板库
            self.tooth_matcher.load_templates()
            
            # 使用颜色选择生成的掩码进行轮廓分析
            if self.current_mask is not None:
                self._perform_color_based_contour_analysis()
            
            # 如果启用了面积分析，则进行面积计算
            if self.enable_area_analysis.get():
                self._perform_area_analysis()
            
            # 在主线程中更新UI
            self.root.after(0, self.update_gui_results)
            
        except Exception as e:
            logger.error(f"基于颜色选择的分析失败: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"分析失败: {str(e)}", foreground="red"))
            self.root.after(0, lambda: self.root.config(cursor=""))
    
    def _perform_color_based_contour_analysis(self):
        """基于颜色选择执行轮廓检测分析"""
        try:
            # 使用已生成的掩码提取颜色区域
            self.color_extract = cv2.bitwise_and(self.original_image, self.original_image, mask=self.current_mask)
            
            # 进行轮廓检测
            contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # 过滤小轮廓
            areas = [cv2.contourArea(c) for c in contours]
            if areas:
                max_area = max(areas)
                min_area = min(areas)
                if max_area > 0 and max_area / max(min_area, 1e-6) > 100:
                    area_threshold = max_area / 100
                    contours = [c for c in contours if cv2.contourArea(c) >= area_threshold]
            
            # 提取特征
            valid_contours = []
            for i, contour in enumerate(contours):
                if contour.shape[0] < Config.MIN_CONTOUR_POINTS:
                    continue
                
                area = cv2.contourArea(contour)
                length = cv2.arcLength(contour, True)
                points = contour[:, 0, :]
                
                # 提取所有特征
                features = self.tooth_matcher.feature_extractor.extract_all_contours(
                    contour, points, image_shape=self.original_image.shape)
                
                valid_contours.append({
                    'contour': contour,
                    'points': points,
                    'area': area,
                    'length': length,
                    'idx': i,
                    'contours': features
                })
            
            self.valid_contours = valid_contours
            
            # 与数据库进行匹配
            if self.tooth_matcher.templates and valid_contours:
                features_list = [vc['contours'] for vc in valid_contours]
                self.match_results = self.tooth_matcher.match_against_database(features_list)
            
            # 保存轮廓图像用于显示
            self.contour_image = self.color_extract
            
            logger.info(f"基于颜色选择检测到 {len(valid_contours)} 个有效轮廓")
            
        except Exception as e:
            logger.error(f"基于颜色选择的轮廓分析失败: {e}")
            # 如果失败，使用原图作为轮廓图
            self.contour_image = self.original_image
    
    def show_color_context_menu(self, event):
        """显示颜色点右键菜单"""
        selection = self.colors_listbox.curselection()
        if not selection:
            return
        
        # 创建右键菜单
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="删除此颜色点", 
                               command=lambda: self.delete_color_point(selection[0]))
        
        # 显示菜单
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def delete_color_point(self, index):
        """删除指定的颜色点"""
        if 0 <= index < len(self.selected_colors):
            removed_color = self.selected_colors.pop(index)
            logger.info(f"删除颜色点 {index+1}: HSV{removed_color['hsv']}")
            
            # 更新显示
            self.update_color_selection_display()
            self.update_mask_preview()
    
    def run(self):
        """运行GUI应用"""
        self.root.mainloop()


if __name__ == "__main__":
    """主执行入口（支持尺度标定参数和GUI模式）"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='牙齿匹配系统 - 集成尺度标定和GUI界面')
    parser.add_argument('--gui', action='store_true', help='启动GUI界面模式')
    parser.add_argument('image_path', nargs='?', default=PHOTO_PATH, 
                       help='图像路径 (默认: %(default)s)')
    parser.add_argument('--scale-mode', choices=['auto', 'manual', 'traditional'], 
                       default=Config.SCALE_CALIBRATION_MODE,
                       help='尺度标定模式 (默认: %(default)s)')
    parser.add_argument('--reference-size', type=float, default=Config.REFERENCE_SIZE_MM,
                       help='参考物尺寸(mm) (默认: %(default)s)')
    parser.add_argument('--manual-scale', type=float, help='手动指定像素/毫米比例')
    parser.add_argument('--no-scale-norm', action='store_true', 
                       help='禁用尺度归一化')
    parser.add_argument('--threshold', type=float, default=Config.SIMILARITY_THRESHOLD,
                       help='相似度阈值 (默认: %(default)s)')
    
    args = parser.parse_args()
    
    # 如果指定了GUI模式，启动GUI
    if args.gui:
        print("🦷 启动牙齿匹配与分析系统 GUI界面")
        try:
            app = ToothMatcherGUI()
            app.run()
        except Exception as e:
            print(f"❌ GUI启动失败: {e}")
            traceback.print_exc()
        sys.exit(0)
    
    # 传统命令行模式
    # 应用配置
    if args.no_scale_norm:
        Config.ENABLE_SCALE_NORMALIZATION = False
    
    Config.SIMILARITY_THRESHOLD = args.threshold
    
    print(f"🦷 牙齿匹配系统启动 (集成尺度标定)")
    print(f"📸 图像路径: {args.image_path}")
    print(f"🔍 尺度模式: {args.scale_mode}")
    print(f"📏 参考物尺寸: {args.reference_size}mm")
    print(f"🔄 尺度归一化: {'启用' if Config.ENABLE_SCALE_NORMALIZATION else '禁用'}")
    print(f"🎯 相似度阈值: {args.threshold}")
    
    # 创建匹配器
    matcher = ToothMatcher(scale_mode=args.scale_mode, 
                          reference_size_mm=args.reference_size)
    
    # 设置手动尺度（如果指定）
    if args.manual_scale:
        matcher.set_manual_scale(args.manual_scale)
    
    try:
        matcher.process_image(args.image_path)
        print("✅ 处理完成")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        traceback.print_exc()
