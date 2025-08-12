import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import sqlite3
import os
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
# 高性能库导入
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# === 1. 移植特征提取相关类 ===
import logging
from numpy.linalg import lstsq
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PHOTO_PATH = r'c:\Users\Jason\Desktop\tooth\Tooth_5.png'

class FourierAnalyzer:import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import sqlite3
import os
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
# 高性能库导入
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# === 1. 移植特征提取相关类 ===
import logging
from numpy.linalg import lstsq
from sklearn.metrics.pairwise import cosine_similarity

# === 新增：标定相关导入 ===
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PHOTO_PATH = r'F:\python\.vscode\toothIDMatch\images\Tooth_10.png'

# ===== 尺度标定相关类定义（移植并适配长方形标定物）=====
@dataclass
class ReferenceObject:
    """参考物规格定义 - 适配134mm×9mm长方形标定物"""
    size_mm: Tuple[float, float] = (134.0, 9.0)  # (长度, 宽度) 单位：mm
    color_hsv_range: Dict = None
    
    def __post_init__(self):
        if self.color_hsv_range is None:
            # 默认红色范围，可根据实际标定物颜色调整
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
    reference_pixel_size: Tuple[float, float]  # (长边像素, 短边像素)
    reference_position: Tuple[int, int, int, int]
    confidence: float
    long_edge_mm: float = 134.0  # 长边物理尺寸
    short_edge_mm: float = 9.0   # 短边物理尺寸
    error_message: str = ""

class ReferenceDetector:
    """参考物检测器 - 专门针对134mm×9mm长方形标定物"""
    
    def __init__(self, reference_obj: ReferenceObject):
        self.reference_obj = reference_obj
        
    def detect_reference_object(self, image: np.ndarray) -> CalibrationResult:
        """检测图像中的长方形参考物并计算标定参数"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = self._create_color_mask(hsv)
            mask = self._clean_mask(mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return CalibrationResult(0, (0, 0), (0, 0, 0, 0), 0, error_message="未检测到参考物颜色")
            
            best_contour = self._find_best_reference_contour(contours)
            
            if best_contour is None:
                return CalibrationResult(0, (0, 0), (0, 0, 0, 0), 0, error_message="未找到符合条件的长方形参考物")
            
            return self._calculate_calibration(best_contour)
            
        except Exception as e:
            logger.error(f"参考物检测失败: {e}")
            return CalibrationResult(0, (0, 0), (0, 0, 0, 0), 0, error_message=f"检测异常: {str(e)}")
    
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
        """找到最佳的长方形参考物轮廓"""
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # 长方形面积阈值比正方形大
                continue
            
            features = self._analyze_contour_shape(contour)
            score = self._evaluate_rectangle_candidate(features)
            
            if score > 0.6:  # 长方形检测要求更高的分数
                candidates.append((contour, score, features))
        
        if not candidates:
            return None
        
        best_contour, best_score, best_features = max(candidates, key=lambda x: x[1])
        logger.info(f"🎯 找到最佳长方形候选，评分: {best_score:.3f}")
        return best_contour
    
    def _analyze_contour_shape(self, contour: np.ndarray) -> Dict:
        """分析轮廓形状特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0  # 长边/短边
        
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
            'bounding_rect': (x, y, w, h),
            'long_edge': max(w, h),
            'short_edge': min(w, h)
        }
    
    def _evaluate_rectangle_candidate(self, features: Dict) -> float:
        """评估长方形参考物候选的质量"""
        score = 0.0
        
        # 长方形比例检查：134:9 = 16.75:1
        aspect_ratio = features['aspect_ratio']
        expected_ratio = 134.0 / 9.0  # 16.75
        ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio
        
        if ratio_error < 0.1:  # 误差小于10%
            score += 0.4
        elif ratio_error < 0.2:  # 误差小于20%
            score += 0.2
        
        # 矩形度检查
        rectangularity = features['rectangularity']
        if rectangularity > 0.8:
            score += 0.3
        elif rectangularity > 0.7:
            score += 0.2
        
        # 实体度检查
        solidity = features['solidity']
        if solidity > 0.85:
            score += 0.2
        elif solidity > 0.75:
            score += 0.1
        
        # 面积合理性检查
        area = features['area']
        if 500 <= area <= 20000:  # 长方形预期面积范围
            score += 0.1
        
        logger.debug(f"长方形评估: 比例={aspect_ratio:.2f}(期望{expected_ratio:.2f}), "
                    f"矩形度={rectangularity:.3f}, 实体度={solidity:.3f}, 总分={score:.3f}")
        
        return min(score, 1.0)
    
    def _calculate_calibration(self, contour: np.ndarray) -> CalibrationResult:
        """计算长方形标定参数"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # 确定长边和短边
        long_edge_pixels = max(w, h)
        short_edge_pixels = min(w, h)
        
        # 使用长边作为主要标定基准
        pixel_per_mm = long_edge_pixels / self.reference_obj.size_mm[0]  # 长边：134mm
        
        # 交叉验证：短边应该约等于 9 * pixel_per_mm
        expected_short_edge = self.reference_obj.size_mm[1] * pixel_per_mm  # 9 * pixel_per_mm
        short_edge_error = abs(short_edge_pixels - expected_short_edge) / expected_short_edge
        
        # 计算置信度（基于长宽比和交叉验证）
        aspect_ratio = long_edge_pixels / short_edge_pixels if short_edge_pixels > 0 else 0
        expected_aspect = self.reference_obj.size_mm[0] / self.reference_obj.size_mm[1]
        aspect_error = abs(aspect_ratio - expected_aspect) / expected_aspect
        
        # 综合置信度计算
        confidence = 1.0 - (aspect_error * 0.6 + short_edge_error * 0.4)
        confidence = max(0.0, min(1.0, confidence))
        
        # 添加详细调试信息
        logger.info(f"📏 长方形标定物检测结果:")
        logger.info(f"   检测位置: ({x}, {y}), 尺寸: {w}×{h} 像素")
        logger.info(f"   长边: {long_edge_pixels} px (对应 134 mm)")
        logger.info(f"   短边: {short_edge_pixels} px (对应 9 mm)")
        logger.info(f"   比例系数: {pixel_per_mm:.4f} px/mm")
        logger.info(f"   实际比例: {aspect_ratio:.2f} (期望: {expected_aspect:.2f})")
        logger.info(f"   短边验证误差: {short_edge_error:.3f}")
        logger.info(f"   置信度: {confidence:.3f}")
        logger.info(f"   面积换算公式: 像素面积 ÷ {pixel_per_mm:.1f}² = 真实面积(mm²)")
        
        return CalibrationResult(
            pixel_per_mm=pixel_per_mm,
            reference_pixel_size=(long_edge_pixels, short_edge_pixels),
            reference_position=(x, y, w, h),
            confidence=confidence,
            long_edge_mm=self.reference_obj.size_mm[0],
            short_edge_mm=self.reference_obj.size_mm[1]
        )

# ===== 物理特征归一化处理器 =====
class PhysicalFeatureNormalizer:
    """物理特征归一化处理器"""
    
    def __init__(self, pixel_per_mm: float, calibration_confidence: float = 1.0):
        self.pixel_per_mm = pixel_per_mm
        self.calibration_confidence = calibration_confidence
        
    def normalize_geometric_features(self, features: dict) -> dict:
        """归一化几何特征"""
        normalized = features.copy()
        
        # 面积归一化: pixel² -> mm²
        if 'area' in features:
            normalized['area_mm2'] = features['area'] / (self.pixel_per_mm ** 2)
        
        # 周长归一化: pixel -> mm
        if 'perimeter' in features:
            normalized['perimeter_mm'] = features['perimeter'] / self.pixel_per_mm
        
        # 边界框归一化
        if 'bounding_rect' in features:
            x, y, w, h = features['bounding_rect']
            normalized['bounding_rect_mm'] = (
                x / self.pixel_per_mm,
                y / self.pixel_per_mm,
                w / self.pixel_per_mm,
                h / self.pixel_per_mm
            )
        
        # 保留原始像素特征用于调试
        normalized['_original_pixel_features'] = {
            'area_px': features.get('area', 0),
            'perimeter_px': features.get('perimeter', 0)
        }
        
        # 添加归一化元数据
        normalized['_normalization_info'] = {
            'pixel_per_mm': self.pixel_per_mm,
            'confidence': self.calibration_confidence,
            'normalized': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return normalized
    
    def normalize_contour_points(self, contour: np.ndarray) -> np.ndarray:
        """归一化轮廓点坐标到物理尺度"""
        normalized_contour = contour.astype(float)
        normalized_contour[:, :, 0] /= self.pixel_per_mm  # x坐标
        normalized_contour[:, :, 1] /= self.pixel_per_mm  # y坐标
        return normalized_contour
    
    def normalize_fourier_descriptors(self, descriptors: np.ndarray, fourier_order: int = 80) -> np.ndarray:
        """归一化傅里叶描述符"""
        if len(descriptors) == 0:
            return descriptors
        
        normalized_descriptors = descriptors.copy()
        
        # DC分量（0阶）需要尺度归一化
        if len(normalized_descriptors) > 0:
            normalized_descriptors[0] /= self.pixel_per_mm
        
        # 如果有Y坐标的DC分量也需要归一化
        if len(normalized_descriptors) > fourier_order:
            normalized_descriptors[fourier_order] /= self.pixel_per_mm
        
        return normalized_descriptors

# ===== 尺度标定相关类定义结束 =====

class FourierAnalyzer:
    @staticmethod
    def fit_fourier_series(data: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
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
        A = np.ones((len(t), 2 * order + 1))
        for k in range(1, order + 1):
            A[:, 2 * k - 1] = np.cos(k * t)
            A[:, 2 * k] = np.sin(k * t)
        return A @ coeffs

    def analyze_contour(self, points: np.ndarray, order: int = 80, center_normalize: bool = True) -> dict:
        try:
            x = points[:, 0].astype(float)
            y = points[:, 1].astype(float)
            center_x = np.mean(x)
            center_y = np.mean(y)
            if center_normalize:
                x_normalized = x - center_x
                y_normalized = y - center_y
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
            coeffs_x = self.fit_fourier_series(x_normalized, t, order)
            coeffs_y = self.fit_fourier_series(y_normalized, t, order)
            t_dense = np.linspace(0, 2 * np.pi, N * 4)
            x_fit_normalized = self.evaluate_fourier_series(coeffs_x, t_dense, order)
            y_fit_normalized = self.evaluate_fourier_series(coeffs_y, t_dense, order)
            if center_normalize:
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
            return {}  # 修正：始终返回dict

class ContourFeatureExtractor:
    def __init__(self):
        self.fourier_analyzer = FourierAnalyzer()

    def extract_geometric_features(self, contour: np.ndarray, image_shape=None) -> dict:
        features = {}
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
        features.update({
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
        return features

    def extract_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        try:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
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
        try:
            fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
            if fourier_data is not None:
                coeffs_x = fourier_data['coeffs_x']
                coeffs_y = fourier_data['coeffs_y']
                fourier_features = np.concatenate([coeffs_x[:11], coeffs_y[:11]])
                return fourier_features
            else:
                return np.zeros(22)
        except Exception as e:
            logger.error(f"傅里叶描述符提取失败: {e}")
            return np.zeros(22)

    def extract_all_features(self, contour: np.ndarray, points: np.ndarray, image_shape=None) -> dict:
        features = {}
        geometric_features = self.extract_geometric_features(contour, image_shape=image_shape)
        features.update(geometric_features)
        features['hu_moments'] = self.extract_hu_moments(contour)
        features['fourier_descriptors'] = self.extract_fourier_descriptors(points)
        fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
        if fourier_data is not None:
            features['fourier_x_fit'] = fourier_data['x_fit'].tolist()
            features['fourier_y_fit'] = fourier_data['y_fit'].tolist()
        return features

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False

class ToothTemplateBuilder:
    def __init__(self, database_path="tooth_templates.db", templates_dir="templates"):
        self.database_path = database_path
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        (self.templates_dir / "contours").mkdir(exist_ok=True)
        (self.templates_dir / "images").mkdir(exist_ok=True)
        (self.templates_dir / "features").mkdir(exist_ok=True)  # 确保features目录存在
        
        # 原有组件
        self.feature_extractor = ContourFeatureExtractor()
        self.current_image = None
        
        # 新增：标定相关组件
        self.reference_detector = ReferenceDetector(
            ReferenceObject(size_mm=(134.0, 9.0))  # 134mm×9mm长方形标定物
        )
        self.feature_normalizer = None  # 动态初始化，基于标定结果
        self.calibration_mode = "auto"  # auto, manual, traditional
        
        # 质量控制参数
        self.min_calibration_confidence = 0.8
        self.require_calibration = True
        self.enable_normalization = True
        
        # 初始化数据库（包含新的标定表）
        self.init_database()
        
        logger.info(f"🏗️ ToothTemplateBuilder初始化完成")
        logger.info(f"   标定模式: {self.calibration_mode}")
        logger.info(f"   标定物规格: 134mm×9mm 长方形")
        logger.info(f"   最小置信度: {self.min_calibration_confidence}")
        logger.info(f"   特征归一化: {'启用' if self.enable_normalization else '禁用'}")
    
    def init_database(self):
        """初始化数据库，包含标定和归一化相关表"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 原有的模板表（扩展字段）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tooth_id TEXT UNIQUE NOT NULL,
                name TEXT,
                image_path TEXT,
                contour_file TEXT,
                num_contours INTEGER,
                total_area REAL,
                
                -- 新增：标定相关字段
                pixel_per_mm REAL,
                calibration_confidence REAL,
                reference_position TEXT,
                physical_area_mm2 REAL,
                physical_perimeter_mm REAL,
                is_normalized BOOLEAN DEFAULT 0,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 新增：标定记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tooth_id TEXT,
                pixel_per_mm REAL,
                confidence REAL,
                reference_type TEXT DEFAULT 'rectangle_134x9',
                reference_position TEXT,
                long_edge_pixels REAL,
                short_edge_pixels REAL,
                calibration_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                calibration_method TEXT DEFAULT 'auto'
            )
        ''')
        
        # 新增：物理特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS physical_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tooth_id TEXT,
                contour_idx INTEGER,
                area_mm2 REAL,
                perimeter_mm REAL,
                bbox_width_mm REAL,
                bbox_height_mm REAL,
                centroid_x_mm REAL,
                centroid_y_mm REAL,
                is_normalized BOOLEAN DEFAULT 1
            )
        ''')
        
        # 新增：归一化元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS normalization_metadata (
                tooth_id TEXT PRIMARY KEY,
                normalization_version TEXT DEFAULT 'v2.0_physical',
                source_image_resolution TEXT,
                pixel_density_dpi REAL,
                processing_pipeline_version TEXT DEFAULT '1.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 检查是否需要添加新列到现有表（向后兼容）
        try:
            cursor.execute("ALTER TABLE templates ADD COLUMN pixel_per_mm REAL")
            cursor.execute("ALTER TABLE templates ADD COLUMN calibration_confidence REAL")
            cursor.execute("ALTER TABLE templates ADD COLUMN reference_position TEXT")
            cursor.execute("ALTER TABLE templates ADD COLUMN physical_area_mm2 REAL")
            cursor.execute("ALTER TABLE templates ADD COLUMN physical_perimeter_mm REAL")
            cursor.execute("ALTER TABLE templates ADD COLUMN is_normalized BOOLEAN DEFAULT 0")
            logger.info("📊 数据库结构已升级，支持标定功能")
        except sqlite3.OperationalError:
            # 列已存在，忽略错误
            pass
        
        conn.commit()
        conn.close()
        logger.info(f"✅ 数据库初始化完成: {self.database_path}")
        logger.info("📋 数据库表结构:")
        logger.info("   • templates: 模板基本信息 (已扩展)")
        logger.info("   • calibration_records: 标定记录")
        logger.info("   • physical_features: 物理特征数据")
        logger.info("   • normalization_metadata: 归一化元数据")

    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.database_path)

    def get_next_tooth_id(self):
        """生成下一个连续的牙模编号"""
        contours_dir = self.templates_dir / "contours"
        if not contours_dir.exists():
            return "TOOTH_001"
        
        existing_files = list(contours_dir.glob("TOOTH_*.json"))
        if not existing_files:
            return "TOOTH_001"
        
        # 提取编号并找到最大值
        max_num = 0
        for file in existing_files:
            try:
                num_str = file.stem.split('_')[1]  # TOOTH_001 -> 001
                num = int(num_str)
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue
        
        return f"TOOTH_{max_num + 1:03d}"

    def serialize_contours(self, valid_contours, tooth_id=None, image_path=None, hsv_info=None, auto_save=False):
        """序列化轮廓数据 - 重构版本，集成标定和归一化功能
        
        新的工作流程：
        1. 载入原始图像
        2. 自动检测长方形标定物 (134mm×9mm)
        3. 计算pixel_per_mm比例和置信度
        4. 进行颜色选择和轮廓提取  
        5. 对所有特征进行物理归一化
        6. 保存归一化后的模板数据
        """
        try:
            if tooth_id is None:
                tooth_id = self.get_next_tooth_id()
            
            logger.info(f"🚀 开始处理模板 {tooth_id}，{len(valid_contours)} 个轮廓")
            
            # ===== 第1步：标定物检测 =====
            calibration_result = None
            if self.enable_normalization and image_path:
                logger.info("🔍 开始长方形标定物检测...")
                try:
                    image = cv2.imread(str(image_path)) if isinstance(image_path, (str, Path)) else self.current_image
                    if image is not None:
                        calibration_result = self.reference_detector.detect_reference_object(image)
                        
                        if calibration_result.pixel_per_mm > 0:
                            logger.info(f"✅ 标定成功! 置信度: {calibration_result.confidence:.3f}")
                            
                            # 检查置信度是否满足要求
                            if calibration_result.confidence < self.min_calibration_confidence:
                                logger.warning(f"⚠️ 标定置信度 {calibration_result.confidence:.3f} 低于阈值 {self.min_calibration_confidence}")
                                if self.require_calibration:
                                    logger.error("❌ 由于置信度不足，终止处理")
                                    return False
                                else:
                                    logger.info("🔄 继续处理但不启用归一化")
                                    calibration_result = None
                        else:
                            logger.warning(f"❌ 标定失败: {calibration_result.error_message}")
                            if self.require_calibration:
                                logger.error("❌ 由于标定失败，终止处理")
                                return False
                            else:
                                logger.info("🔄 降级到传统模式")
                                calibration_result = None
                    else:
                        logger.warning("⚠️ 无法载入图像，跳过标定")
                        calibration_result = None
                except Exception as e:
                    logger.error(f"❌ 标定过程异常: {e}")
                    if self.require_calibration:
                        return False
                    calibration_result = None
            
            # ===== 第2步：初始化特征归一化器 =====
            if calibration_result and calibration_result.pixel_per_mm > 0:
                self.feature_normalizer = PhysicalFeatureNormalizer(
                    calibration_result.pixel_per_mm,
                    calibration_result.confidence
                )
                normalization_enabled = True
                logger.info(f"📏 特征归一化器已初始化 (比例: {calibration_result.pixel_per_mm:.4f} px/mm)")
            else:
                self.feature_normalizer = None
                normalization_enabled = False
                logger.info("📏 使用传统模式（无归一化）")
            
            # ===== 第3步：构建模板数据结构 =====
            template_data = {
                "tooth_id": tooth_id,
                "image_path": str(image_path) if image_path else None,
                "created_at": datetime.now().isoformat(),
                "hsv_info": hsv_info,
                "num_contours": len(valid_contours),
                "contours": [],
                
                # 新增：标定信息
                "calibration_info": {
                    "enabled": normalization_enabled,
                    "pixel_per_mm": calibration_result.pixel_per_mm if calibration_result else None,
                    "confidence": calibration_result.confidence if calibration_result else None,
                    "reference_position": calibration_result.reference_position if calibration_result else None,
                    "reference_type": "rectangle_134x9",
                    "method": self.calibration_mode
                } if calibration_result else {
                    "enabled": False,
                    "method": "traditional"
                },
                
                # 版本信息
                "template_version": "v2.0_normalized" if normalization_enabled else "v1.0_pixel",
                "processing_pipeline": "1.0"
            }
            
            # ===== 第4步：处理每个轮廓 =====
            total_area_pixels = 0
            total_area_mm2 = 0
            physical_features_data = []  # 用于保存到物理特征表
            
            for i, contour_info in enumerate(valid_contours):
                points = contour_info['points']
                contour = contour_info['contour']
                x, y, w, h = cv2.boundingRect(contour)
                
                # 提取原始特征
                original_features = self.feature_extractor.extract_all_features(
                    contour, points, 
                    image_shape=self.current_image.shape if hasattr(self, 'current_image') and self.current_image is not None else None
                )
                
                # 物理归一化（如果启用）
                if self.feature_normalizer:
                    normalized_features = self.feature_normalizer.normalize_geometric_features(original_features)
                    # 归一化傅里叶描述符
                    if 'fourier_descriptors' in original_features:
                        normalized_features['fourier_descriptors'] = self.feature_normalizer.normalize_fourier_descriptors(
                            original_features['fourier_descriptors']
                        ).tolist()
                    
                    # 物理尺寸统计
                    total_area_mm2 += normalized_features.get('area_mm2', 0)
                    
                    # 保存物理特征数据
                    physical_features_data.append({
                        'contour_idx': i,
                        'area_mm2': normalized_features.get('area_mm2', 0),
                        'perimeter_mm': normalized_features.get('perimeter_mm', 0),
                        'bbox_width_mm': normalized_features.get('bounding_rect_mm', (0,0,0,0))[2],
                        'bbox_height_mm': normalized_features.get('bounding_rect_mm', (0,0,0,0))[3],
                        'centroid_x_mm': normalized_features.get('bounding_rect_mm', (0,0,0,0))[0],
                        'centroid_y_mm': normalized_features.get('bounding_rect_mm', (0,0,0,0))[1]
                    })
                else:
                    normalized_features = original_features
                
                # 构建轮廓数据
                contour_data = {
                    "idx": i,
                    "original_idx": contour_info['idx'],
                    "points": points.tolist(),
                    "area": float(contour_info['area']),
                    "perimeter": float(contour_info['length']),
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    
                    # 特征数据（原始+归一化）
                    "features": {
                        # 原始像素特征
                        "pixel_features": {
                            "area": float(original_features['area']),
                            "perimeter": float(original_features['perimeter']),
                            "aspect_ratio": float(original_features['aspect_ratio']),
                            "circularity": float(original_features['circularity']),
                            "solidity": float(original_features['solidity']),
                            "corner_count": int(original_features['corner_count']),
                            "hu_moments": original_features['hu_moments'].tolist(),
                            "fourier_descriptors": original_features['fourier_descriptors'].tolist()
                        },
                        
                        # 物理特征（如果已归一化）
                        "physical_features": {
                            "area_mm2": normalized_features.get('area_mm2', None),
                            "perimeter_mm": normalized_features.get('perimeter_mm', None),
                            "bbox_mm": list(normalized_features.get('bounding_rect_mm', (None, None, None, None))),
                        } if self.feature_normalizer else {},
                        
                        # 尺度无关特征
                        "scale_invariant_features": {
                            "aspect_ratio": float(original_features['aspect_ratio']),
                            "circularity": float(original_features['circularity']),
                            "solidity": float(original_features['solidity']),
                            "hu_moments": original_features['hu_moments'].tolist()
                        },
                        
                        # 归一化的傅里叶描述符
                        "normalized_fourier_descriptors": normalized_features.get('fourier_descriptors', original_features['fourier_descriptors']).tolist() if isinstance(normalized_features.get('fourier_descriptors', original_features['fourier_descriptors']), np.ndarray) else normalized_features.get('fourier_descriptors', original_features['fourier_descriptors'])
                    }
                }
                
                template_data["contours"].append(contour_data)
                total_area_pixels += contour_info['area']
            
            template_data["total_area"] = float(total_area_pixels)
            template_data["total_area_mm2"] = float(total_area_mm2) if normalization_enabled else None
            
            # ===== 第5步：保存模板数据 =====
            # 保存JSON文件
            json_filename = f"{tooth_id}.json"
            json_path = self.templates_dir / "contours" / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            # 保存特征文件到 features 目录
            save_features_only(template_data["contours"], tooth_id)
            
            # 保存轮廓图像（PNG格式）
            png_filename = f"{tooth_id}.png"
            png_path = self.templates_dir / "images" / png_filename
            png_path.parent.mkdir(exist_ok=True)
            
            if hasattr(self, 'current_image') and self.current_image is not None:
                contour_img = self.current_image.copy()
                for contour_info in valid_contours:
                    cv2.drawContours(contour_img, [contour_info['contour']], -1, (0, 255, 0), 2)
                
                # 如果有标定结果，标注标定物位置
                if calibration_result and calibration_result.pixel_per_mm > 0:
                    x, y, w, h = calibration_result.reference_position
                    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(contour_img, f"Calibration: {calibration_result.pixel_per_mm:.2f}px/mm", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imwrite(str(png_path), contour_img)
            
            # ===== 第6步：保存到数据库 =====
            self._save_to_database_extended(template_data, json_filename, image_path, 
                                          calibration_result, physical_features_data)
            
            # ===== 第7步：记录处理结果 =====
            save_type = "自动保存" if auto_save else "手动保存"
            if normalization_enabled:
                logger.info(f"✅ 归一化模板已{save_type}: {tooth_id}")
                logger.info(f"   📊 {len(valid_contours)}个轮廓, 总面积: {total_area_mm2:.2f} mm²")
                logger.info(f"   📏 标定精度: {calibration_result.pixel_per_mm:.4f} px/mm (置信度: {calibration_result.confidence:.3f})")
            else:
                logger.info(f"✅ 传统模板已{save_type}: {tooth_id}")
                logger.info(f"   📊 {len(valid_contours)}个轮廓, 总面积: {total_area_pixels:.0f} px²")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模板保存失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_to_database_extended(self, template_data, json_filename, image_path, 
                                 calibration_result, physical_features_data):
        """扩展的数据库保存方法，支持标定和物理特征数据"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # 保存主模板记录（扩展版）
            cursor.execute('''
                INSERT OR REPLACE INTO templates 
                (tooth_id, name, image_path, contour_file, num_contours, total_area,
                 pixel_per_mm, calibration_confidence, reference_position, 
                 physical_area_mm2, physical_perimeter_mm, is_normalized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_data["tooth_id"],
                f"牙齿模型 {template_data['tooth_id']}",
                str(image_path) if image_path else None,
                json_filename,
                template_data["num_contours"],
                template_data["total_area"],
                calibration_result.pixel_per_mm if calibration_result else None,
                calibration_result.confidence if calibration_result else None,
                json.dumps(calibration_result.reference_position) if calibration_result else None,
                template_data.get("total_area_mm2", None),
                None,  # total_perimeter_mm 需要计算
                1 if calibration_result else 0
            ))
            
            # 保存标定记录
            if calibration_result:
                cursor.execute('''
                    INSERT INTO calibration_records 
                    (tooth_id, pixel_per_mm, confidence, reference_type, reference_position,
                     long_edge_pixels, short_edge_pixels, calibration_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    template_data["tooth_id"],
                    calibration_result.pixel_per_mm,
                    calibration_result.confidence,
                    'rectangle_134x9',
                    json.dumps(calibration_result.reference_position),
                    calibration_result.reference_pixel_size[0],
                    calibration_result.reference_pixel_size[1],
                    self.calibration_mode
                ))
            
            # 保存物理特征数据
            for feature_data in physical_features_data:
                cursor.execute('''
                    INSERT INTO physical_features 
                    (tooth_id, contour_idx, area_mm2, perimeter_mm, bbox_width_mm, 
                     bbox_height_mm, centroid_x_mm, centroid_y_mm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    template_data["tooth_id"],
                    feature_data['contour_idx'],
                    feature_data['area_mm2'],
                    feature_data['perimeter_mm'],
                    feature_data['bbox_width_mm'],
                    feature_data['bbox_height_mm'],
                    feature_data['centroid_x_mm'],
                    feature_data['centroid_y_mm']
                ))
            
            # 保存归一化元数据
            cursor.execute('''
                INSERT OR REPLACE INTO normalization_metadata 
                (tooth_id, normalization_version, processing_pipeline_version)
                VALUES (?, ?, ?)
            ''', (
                template_data["tooth_id"],
                template_data.get("template_version", "v1.0_pixel"),
                template_data.get("processing_pipeline", "1.0")
            ))
            
            conn.commit()
            logger.info("✅ 扩展数据库记录已保存")
            
        except Exception as e:
            logger.error(f"❌ 数据库保存失败: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_to_database(self, template_data, json_filename, image_path):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO templates 
                (tooth_id, name, image_path, contour_file, num_contours, total_area)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                template_data["tooth_id"],
                f"牙齿模型 {template_data['tooth_id']}",
                image_path,
                json_filename,
                template_data["num_contours"],
                template_data["total_area"]
            ))
            conn.commit()
            print(f"✅ 数据库记录已保存")
        except Exception as e:
            print(f"❌ 数据库保存失败: {e}")
        finally:
            conn.close()

    def list_templates(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('SELECT tooth_id, num_contours, total_area, created_at FROM templates ORDER BY created_at DESC')
        templates = cursor.fetchall()
        conn.close()
        
        if templates:
            print("\n📋 已保存的牙齿模板:")
            print("-" * 50)
            for tooth_id, num_contours, total_area, created_at in templates:
                print(f"ID: {tooth_id:<15} | 轮廓: {num_contours:<3} | 面积: {total_area:<8.1f}")
        else:
            print("📭 暂无保存的模板")
        return templates

    def load_saved_contours(self, tooth_id):
        """加载已保存的轮廓数据用于比对
        Args:
            tooth_id: 牙模ID
        Returns:
            dict: 包含轮廓信息的字典，失败返回None
        """
        json_path = self.templates_dir / "contours" / f"{tooth_id}.json"
        if not json_path.exists():
            print(f"❌ 模板文件不存在: {tooth_id}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            print(f"✅ 成功加载模板: {tooth_id}")
            return template_data
        except Exception as e:
            print(f"❌ 加载模板失败: {e}")
            return None

    def compare_with_saved_template(self, current_contours, template_tooth_id):
        """简单的轮廓比对示例
        Args:
            current_contours: 当前检测到的轮廓列表
            template_tooth_id: 要比对的模板ID
        Returns:
            dict: 比对结果
        """
        template_data = self.load_saved_contours(template_tooth_id)
        if not template_data:
            return {"success": False, "error": "无法加载模板"}
        
        current_count = len(current_contours)
        template_count = template_data['num_contours']
        
        # 简单的数量和面积比对
        current_total_area = sum(info['area'] for info in current_contours)
        template_total_area = template_data['total_area']
        
        area_similarity = min(current_total_area, template_total_area) / max(current_total_area, template_total_area)
        count_match = current_count == template_count
        
        result = {
            "success": True,
            "template_id": template_tooth_id,
            "current_count": current_count,
            "template_count": template_count,
            "count_match": count_match,
            "current_area": current_total_area,
            "template_area": template_total_area,
            "area_similarity": area_similarity,
            "is_similar": area_similarity > 0.8 and count_match
        }
        
        print(f"\n📊 轮廓比对结果:")
        print(f"   模板ID: {template_tooth_id}")
        print(f"   轮廓数量: {current_count} vs {template_count} ({'✅ 匹配' if count_match else '❌ 不匹配'})")
        print(f"   总面积: {current_total_area:.1f} vs {template_total_area:.1f}")
        print(f"   面积相似度: {area_similarity:.3f}")
        print(f"   整体相似: {'✅ 是' if result['is_similar'] else '❌ 否'}")
        
        return result

    def list_all_saved_templates(self):
        """列出所有已保存的模板ID"""
        contours_dir = self.templates_dir / "contours"
        if not contours_dir.exists():
            return []
        
        template_files = list(contours_dir.glob("TOOTH_*.json"))
        template_ids = [f.stem for f in template_files]
        
        if template_ids:
            print(f"\n📁 找到 {len(template_ids)} 个已保存模板:")
            for tid in sorted(template_ids):
                print(f"   - {tid}")
        
        return sorted(template_ids)

class BatchToothProcessor:
    """批量牙齿图像处理器 - 基于现有的ToothTemplateBuilder"""
    
    def __init__(self, input_dir: str = "images", templates_dir: str = "templates", 
                 database_path: str = "tooth_templates.db"):
        self.input_dir = Path(input_dir)
        self.templates_dir = Path(templates_dir)
        self.database_path = database_path
        self.builder = ToothTemplateBuilder(database_path, str(templates_dir))
        
        # 支持的图像格式
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # 批量处理状态
        self.processed_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []  # (文件名, 错误信息)
        self.skipped_files: List[str] = []
        
        # 颜色模板缓存
        self.color_template: Optional[Dict] = None
        
        print(f"🚀 批量处理器初始化完成")
        print(f"   📁 输入目录: {self.input_dir}")
        print(f"   📄 模板目录: {self.templates_dir}")
        print(f"   🗄️ 数据库: {self.database_path}")
    
    def scan_image_files(self) -> List[Path]:
        """扫描输入目录中的所有支持的图像文件"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        
        image_files = []
        for ext in self.supported_formats:
            pattern = str(self.input_dir / f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = str(self.input_dir / f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        image_files = [Path(f) for f in image_files]
        image_files = sorted(set(image_files))  # 去重并排序
        
        print(f"📸 发现 {len(image_files)} 个图像文件:")
        for i, file in enumerate(image_files[:10], 1):  # 只显示前10个
            print(f"   {i:2d}. {file.name}")
        if len(image_files) > 10:
            print(f"   ... 还有 {len(image_files) - 10} 个文件")
        
        return image_files
    
    def is_already_processed(self, image_path: Path) -> bool:
        """检查图像是否已经被处理过（通过数据库查询）"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT tooth_id FROM templates WHERE image_path = ?', (str(image_path),))
            result = cursor.fetchone()
            return result is not None
        except Exception:
            return False
        finally:
            conn.close()
    
    def get_color_template_from_first_image(self, first_image_path: Path) -> Optional[Dict]:
        """从第一张图像获取颜色模板（交互式选择）"""
        print(f"\n🎨 请在第一张图像中选择目标颜色:")
        print(f"📸 {first_image_path.name}")
        
        img = cv2.imread(str(first_image_path))
        if img is None:
            print(f"❌ 无法读取图像: {first_image_path}")
            return None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        picked = []
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = hsv[y, x]
                print(f"选中点HSV: {color}")
                picked.append(color)
        
        cv2.imshow("点击选取目标区域颜色 (ESC退出, 多点选择后按ESC)", img)
        cv2.setMouseCallback("点击选取目标区域颜色 (ESC退出, 多点选择后按ESC)", on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if not picked:
            print("❌ 未选取颜色")
            return None
        
        # 计算HSV平均值
        hsv_arr = np.array(picked)
        h_mean, s_mean, v_mean = np.mean(hsv_arr, axis=0).astype(int)
        
        # 创建颜色模板
        color_template = {
            'h_mean': int(h_mean),
            's_mean': int(s_mean),
            'v_mean': int(v_mean),
            'lower': [0, 0, 0],  # 可以根据需要调整
            'upper': [15, 60, 61],  # 可以根据需要调整
            'picked_points': len(picked)
        }
        
        print(f"✅ 颜色模板创建成功:")
        print(f"   HSV均值: ({h_mean}, {s_mean}, {v_mean})")
        print(f"   选取点数: {len(picked)}")
        
        return color_template
    
    def process_single_image_with_template(self, image_path: Path, 
                                         color_template: Dict, 
                                         show_interactive: bool = False) -> bool:
        """使用颜色模板自动处理单张图像"""
        try:
            print(f"🔄 处理中: {image_path.name}")
            
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 应用颜色模板进行HSV掩码
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array(color_template['lower'])
            upper = np.array(color_template['upper'])
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            
            # 智能分离
            mask_processed = choose_separation_method(mask)
            
            # 轮廓检测
            contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid_contours = []
            
            for i, contour in enumerate(contours):
                if contour.shape[0] < 20:
                    continue
                area = cv2.contourArea(contour)
                length = cv2.arcLength(contour, True)
                valid_contours.append({
                    'contour': contour,
                    'points': contour[:, 0, :],
                    'area': area,
                    'length': length,
                    'idx': i
                })
            
            if not valid_contours:
                raise ValueError("未检测到有效轮廓")
            
            # 生成牙齿ID
            tooth_id = self.builder.get_next_tooth_id()
            
            # 创建HSV信息
            hsv_info = {
                'h_mean': color_template['h_mean'],
                's_mean': color_template['s_mean'],
                'v_mean': color_template['v_mean'],
                'lower': color_template['lower'],
                'upper': color_template['upper']
            }
            
            # 自动保存（不显示交互界面）
            success = self.builder.serialize_contours(
                valid_contours, tooth_id, str(image_path), hsv_info, auto_save=True
            )
            
            if success:
                print(f"✅ {image_path.name} -> {tooth_id} ({len(valid_contours)}个轮廓)")
                return True
            else:
                raise ValueError("保存失败")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {image_path.name}: {error_msg}")
            self.failed_files.append((str(image_path), error_msg))
            return False
    
    def process_batch(self, skip_processed: bool = True, 
                     interactive_first: bool = True,
                     show_progress: bool = True) -> Dict:
        """批量处理所有图像"""
        print(f"\n🚀 开始批量处理...")
        print("=" * 60)
        
        # 扫描图像文件
        image_files = self.scan_image_files()
        if not image_files:
            print("❌ 没有找到可处理的图像文件")
            return self._generate_report()
        
        # 过滤已处理的文件
        if skip_processed:
            unprocessed_files = []
            for img_file in image_files:
                if self.is_already_processed(img_file):
                    self.skipped_files.append(str(img_file))
                    print(f"⏭️  跳过已处理: {img_file.name}")
                else:
                    unprocessed_files.append(img_file)
            image_files = unprocessed_files
        
        if not image_files:
            print("✅ 所有图像都已处理完成")
            return self._generate_report()
        
        print(f"\n📊 待处理图像: {len(image_files)} 个")
        
        # 获取颜色模板
        if interactive_first and self.color_template is None:
            self.color_template = self.get_color_template_from_first_image(image_files[0])
            if self.color_template is None:
                print("❌ 无法获取颜色模板，批量处理终止")
                return self._generate_report()
        
        # 处理所有图像
        total_files = len(image_files)
        for i, img_file in enumerate(image_files, 1):
            if show_progress:
                print(f"\n📈 进度: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            success = self.process_single_image_with_template(
                img_file, self.color_template, show_interactive=False
            )
            
            if success:
                self.processed_files.append(str(img_file))
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """生成批量处理报告"""
        total_found = len(self.processed_files) + len(self.failed_files) + len(self.skipped_files)
        
        report = {
            'total_found': total_found,
            'processed': len(self.processed_files),
            'failed': len(self.failed_files),
            'skipped': len(self.skipped_files),
            'success_rate': len(self.processed_files) / max(1, total_found - len(self.skipped_files)) * 100,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files
        }
        
        # 打印报告
        print(f"\n" + "=" * 60)
        print(f"🎉 批量处理完成！")
        print(f"=" * 60)
        print(f"📊 处理统计:")
        print(f"   🔍 发现文件: {report['total_found']} 个")
        print(f"   ✅ 成功处理: {report['processed']} 个")
        print(f"   ❌ 处理失败: {report['failed']} 个")
        print(f"   ⏭️  跳过文件: {report['skipped']} 个")
        print(f"   📈 成功率: {report['success_rate']:.1f}%")
        
        if self.failed_files:
            print(f"\n❌ 失败文件详情:")
            for file_path, error in self.failed_files:
                print(f"   • {Path(file_path).name}: {error}")
        
        return report

def process_image_with_color_template(image_path: str, color_template: Dict, 
                                    tooth_id: Optional[str] = None) -> bool:
    """修改后的颜色处理函数，支持预设颜色模板"""
    builder = ToothTemplateBuilder()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 图片读取失败: {image_path}")
        return False
    
    # 使用预设的颜色模板
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(color_template['lower'])
    upper = np.array(color_template['upper'])
    
    hsv_info = {
        'h_mean': color_template['h_mean'],
        's_mean': color_template['s_mean'], 
        'v_mean': color_template['v_mean'],
        'lower': color_template['lower'],
        'upper': color_template['upper']
    }
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # 其余处理逻辑与原函数相同...
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    mask_processed = choose_separation_method(mask)
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_contours = []
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        valid_contours.append({
            'contour': contour,
            'points': contour[:, 0, :],
            'area': area,
            'length': length,
            'idx': i
        })
    
    if not valid_contours:
        print("❌ 未检测到有效轮廓")
        return False
    
    if tooth_id is None:
        tooth_id = builder.get_next_tooth_id()
    
    success = builder.serialize_contours(valid_contours, tooth_id, image_path, hsv_info, auto_save=True)
    if success:
        print(f"✅ 自动处理完成: {tooth_id} ({len(valid_contours)}个轮廓)")
    
    return success

def pick_color_and_draw_edge(image_path, tooth_id=None):
    """牙齿颜色选择和轮廓提取 - 集成标定功能的新版本
    
    新的工作流程：
    1. 载入图像
    2. 自动检测长方形标定物 (134mm×9mm)
    3. 计算pixel_per_mm比例和置信度
    4. 进行颜色选择和轮廓提取  
    5. 应用物理尺度处理
    6. 保存归一化后的模板数据
    """
    # 初始化模板建立器
    builder = ToothTemplateBuilder()
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 图片读取失败")
        return
    
    # 保存原始图像到builder中
    builder.current_image = img
    
    logger.info(f"🚀 开始处理图像: {image_path}")
    logger.info(f"   图像尺寸: {img.shape[1]}×{img.shape[0]}")
    
    # ===== 第1步：长方形标定物检测 =====
    logger.info("🔍 开始检测134mm×9mm长方形标定物...")
    
    # 显示原始图像以供用户检查标定物
    display_img = img.copy()
    cv2.imshow("图像预览 - 请确认标定物清晰可见 (按任意键继续)", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 执行标定检测
    calibration_result = builder.reference_detector.detect_reference_object(img)
    
    if calibration_result.pixel_per_mm > 0:
        logger.info(f"✅ 标定成功!")
        logger.info(f"   像素比例: {calibration_result.pixel_per_mm:.4f} px/mm")
        logger.info(f"   置信度: {calibration_result.confidence:.3f}")
        logger.info(f"   长边: {calibration_result.reference_pixel_size[0]:.0f} px (134mm)")
        logger.info(f"   短边: {calibration_result.reference_pixel_size[1]:.0f} px (9mm)")
        
        # 显示标定结果
        calib_img = img.copy()
        x, y, w, h = calibration_result.reference_position
        cv2.rectangle(calib_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(calib_img, f"Calibration: {calibration_result.pixel_per_mm:.2f}px/mm", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(calib_img, f"Confidence: {calibration_result.confidence:.3f}", 
                   (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("标定结果 - 红框标出标定物 (按任意键继续)", calib_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 检查置信度
        if calibration_result.confidence < builder.min_calibration_confidence:
            logger.warning(f"⚠️ 标定置信度 {calibration_result.confidence:.3f} 低于阈值 {builder.min_calibration_confidence}")
            response = input("是否继续处理？(y/n): ").lower().strip()
            if response != 'y':
                logger.info("❌ 用户取消处理")
                return
        
        # 初始化特征归一化器
        builder.feature_normalizer = PhysicalFeatureNormalizer(
            calibration_result.pixel_per_mm,
            calibration_result.confidence
        )
        normalization_enabled = True
        
    else:
        logger.warning(f"❌ 标定失败: {calibration_result.error_message}")
        response = input("是否继续使用传统模式？(y/n): ").lower().strip()
        if response != 'y':
            logger.info("❌ 用户取消处理")
            return
        
        logger.info("🔄 继续使用传统模式（无尺度归一化）")
        builder.feature_normalizer = None
        normalization_enabled = False
    
    # ===== 第2步：颜色选择 =====
    logger.info("🎨 开始颜色选择...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    picked = []
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = hsv[y, x]
            logger.info(f"选中点HSV: {color}")
            picked.append(color)
            # 在图像上标记选中点
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(display_img, f"{len(picked)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("点击选取目标区域颜色 (ESC退出)", display_img)
    
    # 创建显示图像副本
    display_img = img.copy()
    
    # 如果有标定结果，在显示图像上标出标定物
    if normalization_enabled:
        x, y, w, h = calibration_result.reference_position
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(display_img, "Calibration Object", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imshow("点击选取目标区域颜色 (ESC退出)", display_img)
    cv2.setMouseCallback("点击选取目标区域颜色 (ESC退出)", on_mouse)
    
    print("🎯 颜色选择说明:")
    print("  • 点击图像中的目标区域来选择颜色")
    print("  • 可以选择多个颜色点")
    print("  • 按ESC键完成选择")
    if normalization_enabled:
        print("  • 蓝框标出了用于尺度标定的长方形标定物")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if not picked:
        logger.warning("❌ 未选取颜色")
        return
    
    # ===== 第3步：HSV处理和掩码生成 =====
    hsv_arr = np.array(picked)
    h, s, v = np.mean(hsv_arr, axis=0).astype(int)
    logger.info(f"📊 颜色统计: HSV平均值 = ({h}, {s}, {v})")
    
    # 自适应HSV范围设置
    tolerance = {'h': 15, 's': 60, 'v': 60}
    lower = np.array([
        max(0, h - tolerance['h']), 
        max(0, s - tolerance['s']), 
        max(0, v - tolerance['v'])
    ])
    upper = np.array([
        min(179, h + tolerance['h']), 
        min(255, s + tolerance['s']), 
        min(255, v + tolerance['v'])
    ])
    
    logger.info(f"🎨 HSV范围: lower={lower}, upper={upper}")
    
    # 保存HSV信息（包含标定信息）
    hsv_info = {
        'h_mean': int(h), 's_mean': int(s), 'v_mean': int(v),
        'lower': lower.tolist(), 'upper': upper.tolist(),
        'picked_points': len(picked),
        'calibration_enabled': normalization_enabled
    }
    
    if normalization_enabled:
        # 添加标定信息到HSV信息中
        hsv_info['calibration_info'] = {
            'pixel_per_mm': calibration_result.pixel_per_mm,
            'confidence': calibration_result.confidence,
            'reference_position': calibration_result.reference_position
        }
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # ===== 第4步：形态学处理和轮廓提取 =====
    logger.info("🔧 执行形态学操作...")
    
    # 先进行开运算去除噪声
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 智能选择分离方法
    mask_processed = choose_separation_method(mask)
    
    # 显示分离效果对比
    show_separation_comparison(mask, mask_processed, image_path)
    
    color_extract = cv2.bitwise_and(img, img, mask=mask_processed)
    
    # ===== 第5步：轮廓检测和特征提取 =====
    logger.info("🔍 检测轮廓...")
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_contours = []
    
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        valid_contours.append({
            'contour': contour,
            'points': contour[:, 0, :],
            'area': area,
            'length': length,
            'idx': i
        })
    
    logger.info(f"✅ 检测到 {len(valid_contours)} 个有效轮廓")
    if not valid_contours:
        logger.error("❌ 未检测到有效轮廓")
        return
    
    # ===== 第6步：准备显示和保存 =====
    n_contours = len(valid_contours)
    linewidth = max(0.5, 2 - 0.03 * n_contours)
    show_legend = n_contours <= 15
    
    # 自动生成牙齿ID（连续编号）
    if tooth_id is None:
        tooth_id = builder.get_next_tooth_id()
    
    # 物理尺寸统计（如果启用了归一化）
    if normalization_enabled:
        total_area_mm2 = sum(contour['area'] for contour in valid_contours) / (calibration_result.pixel_per_mm ** 2)
        total_perimeter_mm = sum(contour['length'] for contour in valid_contours) / calibration_result.pixel_per_mm
        logger.info(f"📏 物理尺寸预览:")
        logger.info(f"   总面积: {total_area_mm2:.2f} mm²")
        logger.info(f"   总周长: {total_perimeter_mm:.2f} mm")
    else:
        total_area_px = sum(contour['area'] for contour in valid_contours)
        total_perimeter_px = sum(contour['length'] for contour in valid_contours)
        logger.info(f"📊 像素尺寸:")
        logger.info(f"   总面积: {total_area_px:.0f} px²")
        logger.info(f"   总周长: {total_perimeter_px:.0f} px")
    
    # ===== 第7步：交互式显示 =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_img, ax_calib = axes[0]
    ax_contour, ax_zoom = axes[1]
    
    # 原始图像
    ax_img.set_title("原始图像", fontsize=14)
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    
    # 标定结果显示
    if normalization_enabled:
        ax_calib.set_title("标定结果", fontsize=14)
        calib_display = img.copy()
        x, y, w, h = calibration_result.reference_position
        cv2.rectangle(calib_display, (x, y), (x+w, y+h), (0, 0, 255), 3)
        ax_calib.imshow(cv2.cvtColor(calib_display, cv2.COLOR_BGR2RGB))
        ax_calib.text(0.02, 0.98, f"比例: {calibration_result.pixel_per_mm:.3f} px/mm\n置信度: {calibration_result.confidence:.3f}\n长边: {calibration_result.reference_pixel_size[0]:.0f}px → 134mm\n短边: {calibration_result.reference_pixel_size[1]:.0f}px → 9mm", 
                     transform=ax_calib.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_calib.axis('off')
    else:
        ax_calib.set_title("传统模式", fontsize=14)
        ax_calib.text(0.5, 0.5, "未启用尺度标定\n使用像素单位", 
                     transform=ax_calib.transAxes, fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax_calib.axis('off')
    
    # 轮廓显示
    ax_contour.set_title(f"轮廓检测结果 - 牙齿ID: {tooth_id}", fontsize=14)
    ax_contour.axis('equal')
    ax_contour.invert_yaxis()
    ax_contour.grid(True)
    
    # 放大视图
    ax_zoom.set_title("选中轮廓放大视图", fontsize=14)
    ax_zoom.axis('equal')
    ax_zoom.invert_yaxis()
    ax_zoom.grid(True)
    
    selected_idx = [0]  # 用列表包裹以便闭包修改
    saved = [False]  # 保存状态
    
    # ===== 第8步：自动保存模板 =====
    logger.info(f"� 开始保存模板 {tooth_id}...")
    success = builder.serialize_contours(valid_contours, tooth_id, image_path, hsv_info, auto_save=True)
    if success:
        saved[0] = True
        if normalization_enabled:
            logger.info(f"✅ 归一化模板已保存: {tooth_id}")
            logger.info(f"   🔬 包含标定信息和物理特征")
        else:
            logger.info(f"✅ 传统模板已保存: {tooth_id}")
    else:
        print(f"❌ 自动保存失败")
    
    def draw_all(highlight_idx=None):
        # 中间图：显示全部轮廓
        ax_contour.clear()
        ax_contour.set_title(f"全部轮廓显示 - 牙齿ID: {tooth_id}")
        ax_contour.axis('equal')
        ax_contour.invert_yaxis()
        ax_contour.grid(True)
        
        # 在原图上叠加所有轮廓
        img_display = img.copy()
        
        # 准备颜色列表
        colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        cmap = plt.get_cmap('tab10')
        colors_plt = cmap(np.linspace(0, 1, max(len(valid_contours), 10)))
        
        for j, info in enumerate(valid_contours):
            contour = info['contour']
            color_bgr = colors_bgr[j % len(colors_bgr)]
            
            if highlight_idx is not None and j == highlight_idx:
                # 高亮显示选中的轮廓
                cv2.drawContours(img_display, [contour], -1, (0, 0, 255), 3)
                # 添加标记点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img_display, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(img_display, f'{j+1}', (cx-8, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # 普通显示其他轮廓
                cv2.drawContours(img_display, [contour], -1, color_bgr, 2)
                # 添加编号
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(img_display, f'{j+1}', (cx-5, cy+3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ax_contour.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        ax_contour.axis('off')
        
        # 右边图：显示选中轮廓的放大视图
        ax_zoom.clear()
        if highlight_idx is not None:
            info = valid_contours[highlight_idx]
            contour = info['contour']
            
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            margin = max(20, max(w, h) * 0.1)  # 自适应边距
            
            # 从原图中裁剪区域
            x1 = max(0, int(x - margin))
            y1 = max(0, int(y - margin))
            x2 = min(img.shape[1], int(x + w + margin))
            y2 = min(img.shape[0], int(y + h + margin))
            
            cropped_img = img[y1:y2, x1:x2].copy()
            
            # 调整轮廓坐标到裁剪图像的坐标系
            adjusted_contour = contour.copy()
            adjusted_contour[:, 0, 0] -= x1
            adjusted_contour[:, 0, 1] -= y1
            
            # 在裁剪图像上绘制轮廓
            cv2.drawContours(cropped_img, [adjusted_contour], -1, (0, 0, 255), 3)
            # 创建半透明填充效果
            overlay = cropped_img.copy()
            cv2.fillPoly(overlay, [adjusted_contour], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, cropped_img, 0.7, 0, cropped_img)
            
            ax_zoom.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            ax_zoom.set_title(f"选中轮廓 {highlight_idx+1} - 面积: {info['area']:.1f} | 周长: {info['length']:.1f}")
        else:
            # 如果没有选中轮廓，显示提示信息
            ax_zoom.text(0.5, 0.5, '点击轮廓查看放大视图\n←→ 键切换轮廓\nq 键退出\n\n✅ 模板已自动保存', 
                        ha='center', va='center', transform=ax_zoom.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax_zoom.set_title("轮廓放大视图")
        
        ax_zoom.axis('off')
        
        # 状态信息显示
        if highlight_idx is not None:
            info = valid_contours[highlight_idx]
            status = "✅ 已自动保存" if saved[0] else "❌ 未保存"
            status_text = f"状态: {status} | 当前: {highlight_idx+1}/{len(valid_contours)} | 面积: {info['area']:.1f} | 周长: {info['length']:.1f}"
        else:
            status = "✅ 已自动保存" if saved[0] else "❌ 未保存"
            status_text = f"状态: {status} | 共 {len(valid_contours)} 个轮廓 | 操作: ←→切换 q退出"
        
        fig.suptitle(status_text, fontsize=12, y=0.02)
        
        fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes != ax_contour:
            return
        
        # 获取点击坐标（需要转换到图像坐标系）
        if event.xdata is None or event.ydata is None:
            return
            
        # 由于ax_contour显示的是图像，坐标系与原图一致
        x, y = int(event.xdata), int(event.ydata)
        
        # 检查点击是否在图像范围内
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return
        
        found = False
        for j, info in enumerate(valid_contours):
            # 检查点是否在轮廓内
            if cv2.pointPolygonTest(info['contour'], (x, y), False) >= 0:
                selected_idx[0] = j
                draw_all(highlight_idx=j)
                found = True
                print(f"✅ 选中轮廓 {j+1}")
                break
        
        if not found:
            print("未选中任何轮廓")
    
    def on_key(event):
        if event.key == 'right':
            selected_idx[0] = (selected_idx[0] + 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'left':
            selected_idx[0] = (selected_idx[0] - 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'q':
            plt.close()
    
    draw_all(highlight_idx=0 if valid_contours else None)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为状态信息留出空间
    plt.show()
    
    # 显示已保存的模板列表
    builder.list_templates()

def ultra_separate_connected_objects(mask):
    """
    超强黏连分离算法 - 仅使用OpenCV，无需额外依赖
    """
    print("🚀 启动超强分离算法（OpenCV版本）...")
    
    # 步骤1: 清理噪声
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # 步骤2: 多策略分离尝试
    best_result = mask_clean
    max_components = 1
    
    # 策略1: 激进腐蚀分离
    erosion_configs = [
        (1, 3), (2, 3), (3, 3), (4, 3),  # 小核多次迭代
        (1, 5), (2, 5), (3, 5),          # 中核
        (1, 7), (2, 7)                   # 大核
    ]
    
    for iterations, kernel_size in erosion_configs:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask_clean, kernel, iterations=iterations)
        
        # 检查是否成功分离
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        if num_labels > max_components:
            max_components = num_labels
            print(f"💪 找到更好分离: {num_labels-1} 个区域 (腐蚀{iterations}次,核{kernel_size}x{kernel_size})")
            
            # 恢复各个区域
            result_mask = np.zeros_like(mask_clean)
            
            for i in range(1, num_labels):  # 跳过背景
                # 获取当前区域
                component = (labels == i).astype(np.uint8) * 255
                
                # 渐进膨胀恢复
                restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (min(kernel_size, 5), min(kernel_size, 5)))
                restored = cv2.dilate(component, restore_kernel, iterations=min(iterations, 2))
                
                # 限制在原始区域内
                restored = cv2.bitwise_and(restored, mask_clean)
                
                result_mask = cv2.bitwise_or(result_mask, restored)
            
            best_result = result_mask
    
    print(f"✅ 超强分离完成！最终分离出 {max_components-1} 个独立区域")
    return best_result

def force_separation_with_morphology(mask):
    """
    强制形态学分离 - 当分水岭失败时的终极备选方案
    """
    print("🔧 启动强制形态学分离...")
    original_mask = mask.copy()
    best_result = mask.copy()
    max_components = 1
    
    # 极度激进的腐蚀策略
    erosion_configs = [
        (1, (3, 3)), (2, (3, 3)), (3, (3, 3)), (4, (3, 3)), (5, (3, 3)),
        (1, (5, 5)), (2, (5, 5)), (3, (5, 5)),
        (1, (7, 7)), (2, (7, 7)),
        (1, (9, 9))
    ]
    
    for iterations, kernel_size in erosion_configs:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        eroded = cv2.erode(original_mask, kernel, iterations=iterations)
        
        # 检查连通分量
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        if num_labels > max_components:
            max_components = num_labels
            result_mask = np.zeros_like(mask)
            
            for i in range(1, num_labels):
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # 渐进式膨胀恢复
                restore_iterations = min(iterations, 3)  # 限制恢复强度
                kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (restore_iterations*2+1, restore_iterations*2+1))
                restored = cv2.dilate(component_mask, kernel_restore, iterations=restore_iterations)
                
                # 限制在扩展的原始区域内
                expanded_original = cv2.dilate(original_mask, np.ones((3,3), np.uint8), iterations=2)
                restored = cv2.bitwise_and(restored, expanded_original)
                
                result_mask = cv2.bitwise_or(result_mask, restored)
            
            best_result = result_mask.copy()
            print(f"💪 形态学方案找到 {max_components-1} 个区域 (腐蚀{iterations}次,核{kernel_size})")
    
    print(f"✅ 强制分离完成，最终分离出 {max_components-1} 个区域")
    return best_result
    """
    超强黏连分离算法 - 针对牙齿模型优化
    """
    # 步骤1: 预处理 - 去除小噪声和平滑
    mask_bool = mask > 0
    mask_clean = remove_small_objects(mask_bool, min_size=30, connectivity=2)
    mask_clean = binary_opening(mask_clean, disk(1))  # 减少开运算强度
    mask_clean = mask_clean.astype(np.uint8) * 255
    
    # 步骤2: 高精度距离变换
    dist_transform = distance_transform_edt(mask_clean)
    
    # 步骤3: 更激进的参数设置 - 专门针对牙齿黏连
    img_area = mask_clean.shape[0] * mask_clean.shape[1]
    max_dist = np.max(dist_transform)
    
    # 更激进的参数，强制分离黏连牙齿
    if img_area > 500000:  # 大图像
        min_distance = 2  # 极小
        threshold_abs = max_dist * 0.05  # 更低
        threshold_rel = 0.02
    elif img_area > 100000:  # 中等图像
        min_distance = 1
        threshold_abs = max_dist * 0.03
        threshold_rel = 0.01
    else:  # 小图像
        min_distance = 1
        threshold_abs = max_dist * 0.01
        threshold_rel = 0.005
    
    print(f"🔍 距离变换最大值: {max_dist:.2f}")
    print(f"📊 参数设置 - 最小距离: {min_distance}, 阈值: {threshold_abs:.2f}")
    
    # 步骤4: 寻找局部最大值作为分离种子
    local_maxima = peak_local_max(
        dist_transform,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    
    print(f"🎯 检测到 {len(local_maxima)} 个高质量分离种子点")
    
    if len(local_maxima) == 0:
        print("⚠️ 未找到分离点，降低阈值重试...")
        # 降低阈值重试
        local_maxima = peak_local_max(
            dist_transform,
            min_distance=max(min_distance//2, 3),
            threshold_abs=threshold_abs * 0.5,
            threshold_rel=threshold_rel * 0.5
        )
        print(f"🔄 重试后检测到 {len(local_maxima)} 个种子点")
    
    if len(local_maxima) == 0:
        print("❌ 仍未找到分离点，使用备选方案")
        return advanced_separate_connected_objects(mask_clean)
    
    # 步骤5: 创建高质量标记图像
    markers = np.zeros_like(mask_clean, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxima):
        markers[y, x] = i + 1
    
    # 使用形态学膨胀扩展标记，但控制扩展程度
    expansion_size = max(1, min_distance // 4)
    markers = ndimage.binary_dilation(
        markers > 0, 
        structure=disk(expansion_size)
    ).astype(np.int32)
    
    # 重新标记连通分量
    markers = label(markers)
    
    # 步骤6: 高性能分水岭分割
    labels = watershed(-dist_transform, markers, mask=mask_clean)
    
    # 步骤7: 智能后处理
    result_mask = np.zeros_like(mask_clean)
    regions = regionprops(labels)
    
    min_area = 100  # 最小区域面积
    processed_regions = 0
    
    for region in regions:
        if region.area < min_area:
            continue
            
        # 获取区域mask
        region_mask = (labels == region.label).astype(np.uint8) * 255
        
        # 形态学闭运算填补空洞，使用自适应核大小
        close_size = max(1, int(np.sqrt(region.area) * 0.05))
        kernel_close = disk(close_size)
        region_mask = ndimage.binary_closing(region_mask, structure=kernel_close)
        region_mask = region_mask.astype(np.uint8) * 255
        
        # 合并到结果
        result_mask = cv2.bitwise_or(result_mask, region_mask)
        processed_regions += 1
    
    print(f"✅ 高性能分离完成！生成 {processed_regions} 个独立高质量区域")
    return result_mask

def advanced_separate_connected_objects(mask):
    """
    高级分离方法：结合多种形态学操作，不依赖额外库
    """
    # 方法1: 基于腐蚀-膨胀的分离
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel_erode, iterations=2)
    
    # 寻找连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    if num_labels <= 1:  # 没有找到分离的区域
        print("⚠️ 腐蚀后未找到分离区域，尝试更强的分离")
        return erosion_dilation_separation(mask)
    
    result_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # 跳过背景
        # 获取当前连通分量
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # 对每个分量进行膨胀恢复
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(component_mask, kernel_dilate, iterations=2)
        
        # 与原始mask取交集，避免过度膨胀
        dilated = cv2.bitwise_and(dilated, mask)
        
        result_mask = cv2.bitwise_or(result_mask, dilated)
    
    print(f"✅ 腐蚀-膨胀分离完成，生成 {num_labels-1} 个区域")
    return result_mask

def erosion_dilation_separation(mask):
    """
    渐进式腐蚀分离算法
    """
    original_mask = mask.copy()
    best_result = mask.copy()
    max_components = 1
    
    # 尝试不同强度的腐蚀
    for iterations in range(1, 6):
        for kernel_size in [(3,3), (5,5), (7,7)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded = cv2.erode(original_mask, kernel, iterations=iterations)
            
            # 检查连通分量
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            
            if num_labels > max_components:
                max_components = num_labels
                # 恢复各个分量
                result_mask = np.zeros_like(mask)
                
                for i in range(1, num_labels):
                    component_mask = (labels == i).astype(np.uint8) * 255
                    
                    # 膨胀恢复，但限制在原始区域内
                    kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iterations*2+1, iterations*2+1))
                    restored = cv2.dilate(component_mask, kernel_restore, iterations=iterations)
                    restored = cv2.bitwise_and(restored, original_mask)
                    
                    result_mask = cv2.bitwise_or(result_mask, restored)
                
                best_result = result_mask.copy()
    
    print(f"✅ 渐进式分离完成，最多分离出 {max_components-1} 个区域")
    return best_result

def choose_separation_method(mask):
    """
    智能选择高性能分离方法
    """
    # 计算初始连通分量数
    num_labels_initial, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels_initial > 2:  # 已经分离，无需处理
        print("✅ 区域已经分离，无需额外处理")
        return mask
    
    # 分析图像特征
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask
    
    # 计算多个复杂度指标
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    
    # 形状复杂度：周长平方/面积
    shape_complexity = (total_perimeter ** 2) / (total_area + 1e-6)
    
    # 凸性分析
    total_hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)
    convexity = total_area / (total_hull_area + 1e-6)
    
    # 区域紧凑度
    compactness = (4 * np.pi * total_area) / (total_perimeter ** 2 + 1e-6)
    
    print(f"🔍 图像分析结果:")
    print(f"   📊 形状复杂度: {shape_complexity:.2f}")
    print(f"   🔄 凸性系数: {convexity:.3f}")
    print(f"   📐 紧凑度: {compactness:.3f}")
    
    # 智能选择分离策略
    try:
        # 优先使用高性能的scikit-image算法
        if shape_complexity > 80 or convexity < 0.7:
            print("🚀 使用超强分离算法（复杂形状）...")
            return ultra_separate_connected_objects(mask)
        elif compactness < 0.3:
            print("🚀 使用超强分离算法（非紧凑形状）...")
            return ultra_separate_connected_objects(mask)
        else:
            print("⚡ 使用高速形态学方法（简单形状）...")
            return advanced_separate_connected_objects(mask)
    except Exception as e:
        print(f"⚠️ 高性能算法失败: {e}")
        print("🔄 回退到稳定的OpenCV方法...")
        return advanced_separate_connected_objects(mask)

def show_separation_comparison(original_mask, processed_mask, image_path):
    """
    高性能分离效果可视化对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始图像
    img = cv2.imread(image_path)
    if img is not None:
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("原始图像", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
    
    # 分离前的mask
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title("分离前", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 分离后的mask
    axes[0, 2].imshow(processed_mask, cmap='gray')
    axes[0, 2].set_title("分离后", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 轮廓对比 - 分离前
    contours_before, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours_before = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2RGB)
    for i, contour in enumerate(contours_before):
        cv2.drawContours(img_contours_before, [contour], -1, (255, 0, 0), 2)
        # 添加编号
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contours_before, str(i+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    axes[1, 0].imshow(img_contours_before)
    axes[1, 0].set_title("分离前轮廓", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 轮廓对比 - 分离后
    contours_after, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours_after = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, contour in enumerate(contours_after):
        color = colors[i % len(colors)]
        cv2.drawContours(img_contours_after, [contour], -1, color, 2)
        # 添加编号
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contours_after, str(i+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    axes[1, 1].imshow(img_contours_after)
    axes[1, 1].set_title("分离后轮廓", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 统计信息图表
    valid_before = len([c for c in contours_before if cv2.contourArea(c) > 100])
    valid_after = len([c for c in contours_after if cv2.contourArea(c) > 100])
    
    areas_before = [cv2.contourArea(c) for c in contours_before if cv2.contourArea(c) > 100]
    areas_after = [cv2.contourArea(c) for c in contours_after if cv2.contourArea(c) > 100]
    
    # 面积对比柱状图
    axes[1, 2].bar(['分离前', '分离后'], [sum(areas_before), sum(areas_after)], 
                   color=['red', 'green'], alpha=0.7)
    axes[1, 2].set_title("总面积对比", fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel("面积 (像素)")
    
    # 在图上添加数值
    for i, v in enumerate([sum(areas_before), sum(areas_after)]):
        axes[1, 2].text(i, v + max(areas_before + areas_after) * 0.02, f'{int(v)}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # 分离效果信息
    improvement_ratio = valid_after / max(valid_before, 1)
    separation_info = f'''分离性能报告:
    ├─ 区域数量: {valid_before} → {valid_after}
    ├─ 提升倍数: {improvement_ratio:.2f}x
    ├─ 总面积: {sum(areas_before):.0f} → {sum(areas_after):.0f}
    └─ 平均面积: {np.mean(areas_before):.0f} → {np.mean(areas_after):.0f}'''
    
    fig.suptitle(f'🚀 高性能分离效果对比\n{separation_info}', 
                fontsize=16, fontweight='bold', y=0.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    print(f"\n🎯 分离性能总结:")
    print(f"   🔢 区域数量变化: {valid_before} → {valid_after}")
    print(f"   📈 分离效果提升: {improvement_ratio:.2f}倍")
    print(f"   📊 面积保持率: {sum(areas_after)/sum(areas_before)*100:.1f}%")

def save_features_only(valid_contours, tooth_id, features_dir="templates/features"):
    from pathlib import Path
    import numpy as np

    def to_serializable(feat):
        # 把所有 ndarray 转成 list
        if isinstance(feat, np.ndarray):
            return feat.tolist()
        if isinstance(feat, dict):
            return {k: to_serializable(v) for k, v in feat.items()}
        if isinstance(feat, list):
            return [to_serializable(x) for x in feat]
        return feat

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    features_list = [to_serializable(contour['features']) for contour in valid_contours]
    features_path = features_dir / f"{tooth_id}_features.json"
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump({"features": features_list}, f, ensure_ascii=False, indent=2)
    print(f"✅ 纯特征文件已保存: {features_path}")


def main():
    """
    高性能牙齿模板建立器主程序 - 支持单张和批量处理
    """
    parser = argparse.ArgumentParser(description='牙齿模板建立器')
    parser.add_argument('--batch', action='store_true', help='启用批量处理模式')
    parser.add_argument('--input-dir', default='images', help='输入目录路径 (默认: images)')
    parser.add_argument('--output-dir', default='templates', help='输出目录路径 (默认: templates)')
    parser.add_argument('--database', default='tooth_templates.db', help='数据库路径 (默认: tooth_templates.db)')
    parser.add_argument('--skip-processed', action='store_true', default=True, 
                       help='跳过已处理的文件 (默认: True)')
    parser.add_argument('--single-image', help='处理单张图像的路径')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        print("🚀 启动批量牙齿模板建立器")
        print("=" * 60)
        
        processor = BatchToothProcessor(
            input_dir=args.input_dir,
            templates_dir=args.output_dir,
            database_path=args.database
        )
        
        try:
            report = processor.process_batch(
                skip_processed=args.skip_processed,
                interactive_first=True,
                show_progress=True
            )
            
            # 显示最终统计
            if report['processed'] > 0:
                print(f"\n🎯 批量处理成功完成!")
                print(f"✅ 已创建 {report['processed']} 个牙齿模板")
                
                # 显示已保存的模板列表
                processor.builder.list_templates()
            
        except Exception as e:
            print(f"❌ 批量处理过程中发生错误: {e}")
            print("💡 请检查输入目录和文件权限")
    
    elif args.single_image:
        # 单张图像处理模式
        print("🚀 启动单张图像处理模式")
        print("=" * 50)
        
        image_path = args.single_image
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return
        
        print(f"📸 正在处理图像: {image_path}")
        
        try:
            pick_color_and_draw_edge(image_path, tooth_id=None)
            print("\n🎉 单张图像处理完成！")
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {e}")
            
    else:
        # 默认单张处理模式（使用PHOTO_PATH）
        print("🚀 启动高性能牙齿模板建立器")
        print("=" * 50)
        
        # 自动生成连续编号，无需用户输入
        tooth_id = None  # 将自动生成 TOOTH_001, TOOTH_002...
        
        # 图像路径
        image_path = PHOTO_PATH 
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            print("💡 请检查文件路径是否正确")
            print(f"💡 或使用 --single-image 指定图像路径")
            print(f"💡 或使用 --batch --input-dir 指定批量处理目录")
            return
        
        print(f"📸 正在处理图像: {image_path}")
        
        try:
            # 启动高性能分离和模板建立（自动保存）
            pick_color_and_draw_edge(image_path, tooth_id)
            print("\n🎉 高性能处理完成！")
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {e}")
            print("💡 请检查图像文件和依赖库是否正确安装")

def main_batch_example():
    """批处理示例 - 演示多个图片的处理"""
    processor = BatchToothProcessor(
        input_dir="images",
        templates_dir="templates",
        database_path="tooth_templates.db"
    )
    
    processor.process_batch(skip_processed=True)

def test_calibration_system():
    """测试标定系统的功能"""
    logger.info("🧪 开始测试标定系统...")
    
    # 创建测试用的模板构建器
    builder = ToothTemplateBuilder()
    
    # 测试标定物检测
    test_image_path = "images/test_tooth.jpg"  # 确保这个路径存在
    if os.path.exists(test_image_path):
        logger.info(f"📸 测试图像: {test_image_path}")
        
        # 加载图像
        img = cv2.imread(test_image_path)
        if img is not None:
            # 执行标定检测
            calibration_result = builder.reference_detector.detect_reference_object(img)
            
            if calibration_result.pixel_per_mm > 0:
                logger.info(f"✅ 标定测试成功!")
                logger.info(f"   像素比例: {calibration_result.pixel_per_mm:.4f} px/mm")
                logger.info(f"   置信度: {calibration_result.confidence:.3f}")
                logger.info(f"   参考物位置: {calibration_result.reference_position}")
                
                # 测试特征归一化器
                normalizer = PhysicalFeatureNormalizer(
                    calibration_result.pixel_per_mm,
                    calibration_result.confidence
                )
                
                # 创建测试特征
                test_features = {
                    'area': 10000.0,  # 像素²
                    'perimeter': 400.0,  # 像素
                    'bounding_rect': (100, 200, 50, 80)  # x, y, w, h
                }
                
                # 测试归一化
                normalized_features = normalizer.normalize_geometric_features(test_features)
                
                logger.info(f"🔬 特征归一化测试:")
                logger.info(f"   原始面积: {test_features['area']:.0f} px²")
                logger.info(f"   归一化面积: {normalized_features.get('area_mm2', 0):.2f} mm²")
                logger.info(f"   原始周长: {test_features['perimeter']:.0f} px")
                logger.info(f"   归一化周长: {normalized_features.get('perimeter_mm', 0):.2f} mm")
                
                return True
            else:
                logger.warning(f"❌ 标定测试失败: {calibration_result.error_message}")
                return False
        else:
            logger.error(f"❌ 无法加载测试图像: {test_image_path}")
            return False
    else:
        logger.warning(f"⚠️ 测试图像不存在: {test_image_path}")
        logger.info("💡 请将包含134mm×9mm长方形标定物的图像放置在images目录下")
        return False

def show_database_status():
    """显示数据库状态和统计信息"""
    builder = ToothTemplateBuilder()
    
    logger.info("📊 数据库状态报告:")
    logger.info("=" * 50)
    
    conn = sqlite3.connect(builder.database_path)
    cursor = conn.cursor()
    
    try:
        # 统计模板数量
        cursor.execute("SELECT COUNT(*) FROM templates")
        total_templates = cursor.fetchone()[0]
        logger.info(f"📋 总模板数: {total_templates}")
        
        # 统计归一化模板数量
        cursor.execute("SELECT COUNT(*) FROM templates WHERE is_normalized = 1")
        normalized_templates = cursor.fetchone()[0]
        logger.info(f"🔬 归一化模板数: {normalized_templates}")
        
        # 统计传统模板数量
        traditional_templates = total_templates - normalized_templates
        logger.info(f"📝 传统模板数: {traditional_templates}")
        
        # 标定记录统计
        cursor.execute("SELECT COUNT(*) FROM calibration_records")
        calibration_records = cursor.fetchone()[0]
        logger.info(f"📏 标定记录数: {calibration_records}")
        
        # 物理特征记录统计
        cursor.execute("SELECT COUNT(*) FROM physical_features")
        physical_features = cursor.fetchone()[0]
        logger.info(f"🔬 物理特征记录数: {physical_features}")
        
        # 最近的模板
        cursor.execute("""
            SELECT tooth_id, created_at, is_normalized, pixel_per_mm 
            FROM templates 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        recent_templates = cursor.fetchall()
        
        if recent_templates:
            logger.info("\n📅 最近的模板:")
            for tooth_id, created_at, is_normalized, pixel_per_mm in recent_templates:
                status = "归一化" if is_normalized else "传统"
                scale_info = f"({pixel_per_mm:.3f} px/mm)" if pixel_per_mm else ""
                logger.info(f"   {tooth_id} - {status} {scale_info} - {created_at}")
        
    except sqlite3.Error as e:
        logger.error(f"❌ 数据库查询失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="牙齿模板构建工具 - 集成标定功能")
    parser.add_argument('--mode', choices=['single', 'batch', 'test', 'status'], 
                       default='single', help='运行模式')
    parser.add_argument('--image', help='单个图像路径 (single模式)')
    parser.add_argument('--input-dir', default='images', help='输入目录路径 (batch模式)')
    parser.add_argument('--output-dir', default='templates', help='输出目录路径')
    parser.add_argument('--database', default='tooth_templates.db', help='数据库路径')
    parser.add_argument('--tooth-id', help='指定牙齿ID')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if args.image and os.path.exists(args.image):
            logger.info(f"🦷 单图像处理模式: {args.image}")
            pick_color_and_draw_edge(args.image, args.tooth_id)
        else:
            logger.info("🦷 交互式单图像处理模式")
            # 默认图像路径
            image_path = PHOTO_PATH
            if not os.path.exists(image_path):
                # 尝试找到任意图像
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(f"images/{ext}"))
                
                if image_files:
                    image_path = image_files[0]
                    logger.info(f"使用图像: {image_path}")
                else:
                    logger.error("❌ 未找到图像文件，请在images目录中放置图像")
                    exit(1)
            
            pick_color_and_draw_edge(image_path, args.tooth_id)
    
    elif args.mode == 'batch':
        logger.info(f"📁 批处理模式: {args.input_dir} -> {args.output_dir}")
        processor = BatchToothProcessor(
            input_dir=args.input_dir,
            templates_dir=args.output_dir,
            database_path=args.database
        )
        processor.process_batch(skip_processed=True)
    
    elif args.mode == 'test':
        logger.info("🧪 测试模式")
        success = test_calibration_system()
        if success:
            logger.info("✅ 所有测试通过")
        else:
            logger.error("❌ 测试失败")
            exit(1)
    
    elif args.mode == 'status':
        logger.info("📊 状态查看模式")
        show_database_status()
    
    logger.info("🎉 程序执行完成")

    @staticmethod
    def fit_fourier_series(data: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
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
        A = np.ones((len(t), 2 * order + 1))
        for k in range(1, order + 1):
            A[:, 2 * k - 1] = np.cos(k * t)
            A[:, 2 * k] = np.sin(k * t)
        return A @ coeffs

    def analyze_contour(self, points: np.ndarray, order: int = 80, center_normalize: bool = True) -> dict:
        try:
            x = points[:, 0].astype(float)
            y = points[:, 1].astype(float)
            center_x = np.mean(x)
            center_y = np.mean(y)
            if center_normalize:
                x_normalized = x - center_x
                y_normalized = y - center_y
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
            coeffs_x = self.fit_fourier_series(x_normalized, t, order)
            coeffs_y = self.fit_fourier_series(y_normalized, t, order)
            t_dense = np.linspace(0, 2 * np.pi, N * 4)
            x_fit_normalized = self.evaluate_fourier_series(coeffs_x, t_dense, order)
            y_fit_normalized = self.evaluate_fourier_series(coeffs_y, t_dense, order)
            if center_normalize:
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
            return {}  # 修正：始终返回dict

class ContourFeatureExtractor:
    def __init__(self):
        self.fourier_analyzer = FourierAnalyzer()

    def extract_geometric_features(self, contour: np.ndarray, image_shape=None) -> dict:
        features = {}
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
        features.update({
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
        return features

    def extract_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        try:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
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
        try:
            fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
            if fourier_data is not None:
                coeffs_x = fourier_data['coeffs_x']
                coeffs_y = fourier_data['coeffs_y']
                fourier_features = np.concatenate([coeffs_x[:11], coeffs_y[:11]])
                return fourier_features
            else:
                return np.zeros(22)
        except Exception as e:
            logger.error(f"傅里叶描述符提取失败: {e}")
            return np.zeros(22)

    def extract_all_features(self, contour: np.ndarray, points: np.ndarray, image_shape=None) -> dict:
        features = {}
        geometric_features = self.extract_geometric_features(contour, image_shape=image_shape)
        features.update(geometric_features)
        features['hu_moments'] = self.extract_hu_moments(contour)
        features['fourier_descriptors'] = self.extract_fourier_descriptors(points)
        fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
        if fourier_data is not None:
            features['fourier_x_fit'] = fourier_data['x_fit'].tolist()
            features['fourier_y_fit'] = fourier_data['y_fit'].tolist()
        return features

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False

class ToothTemplateBuilder:
    def __init__(self, database_path="tooth_templates.db", templates_dir="templates"):
        self.database_path = database_path
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        (self.templates_dir / "contours").mkdir(exist_ok=True)
        (self.templates_dir / "images").mkdir(exist_ok=True)
        self.init_database()
        self.feature_extractor = ContourFeatureExtractor()
        self.current_image = None  # type: ignore  # 修正：允许动态类型
    
    def init_database(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tooth_id TEXT UNIQUE NOT NULL,
                name TEXT,
                image_path TEXT,
                contour_file TEXT,
                num_contours INTEGER,
                total_area REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print(f"✅ 数据库初始化完成: {self.database_path}")

    def get_next_tooth_id(self):
        """生成下一个连续的牙模编号"""
        contours_dir = self.templates_dir / "contours"
        if not contours_dir.exists():
            return "TOOTH_001"
        
        existing_files = list(contours_dir.glob("TOOTH_*.json"))
        if not existing_files:
            return "TOOTH_001"
        
        # 提取编号并找到最大值
        max_num = 0
        for file in existing_files:
            try:
                num_str = file.stem.split('_')[1]  # TOOTH_001 -> 001
                num = int(num_str)
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue
        
        return f"TOOTH_{max_num + 1:03d}"

    def serialize_contours(self, valid_contours, tooth_id=None, image_path=None, hsv_info=None, auto_save=False):
        """序列化轮廓数据
        Args:
            valid_contours: 有效轮廓列表
            tooth_id: 牙模ID，如果为None则自动生成
            image_path: 图像路径
            hsv_info: HSV颜色信息
            auto_save: 是否自动保存（无需用户确认）
        """
        try:
            if tooth_id is None:
                tooth_id = self.get_next_tooth_id()
            
            template_data = {
                "tooth_id": tooth_id,
                "image_path": str(image_path) if image_path else None,
                "created_at": datetime.now().isoformat(),
                "hsv_info": hsv_info,
                "num_contours": len(valid_contours),
                "contours": []
            }
            
            total_area = 0
            for i, contour_info in enumerate(valid_contours):
                points = contour_info['points']
                contour = contour_info['contour']
                x, y, w, h = cv2.boundingRect(contour)
                # === 新增：提取高级特征 ===
                features = self.feature_extractor.extract_all_features(contour, points, image_shape=self.current_image.shape if hasattr(self, 'current_image') and self.current_image is not None else None)
                contour_info['features'] = features  # ★★★ 关键：加上这一行
                contour_data = {
                    "idx": i,
                    "original_idx": contour_info['idx'],
                    "points": points.tolist(),
                    "area": float(contour_info['area']),
                    "perimeter": float(contour_info['length']),
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "features": {
                        "area": float(features['area']),
                        "perimeter": float(features['perimeter']),
                        "area_norm": float(features['area_norm']),
                        "perimeter_norm": float(features['perimeter_norm']),
                        "aspect_ratio": float(features['aspect_ratio']),
                        "circularity": float(features['circularity']),
                        "solidity": float(features['solidity']),
                        "corner_count": int(features['corner_count']),
                        "hu_moments": features['hu_moments'].tolist(),
                        "fourier_descriptors": features['fourier_descriptors'].tolist()
                    }
                }
                template_data["contours"].append(contour_data)
                total_area += contour_info['area']
            
            template_data["total_area"] = float(total_area)
            
            # 保存JSON文件
            json_filename = f"{tooth_id}.json"
            json_path = self.templates_dir / "contours" / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)

            # === 新增：保存特征文件到 features 目录 ===
            save_features_only(valid_contours, tooth_id)
            
            # 同时保存轮廓图像（PNG格式）
            png_filename = f"{tooth_id}.png"
            png_path = self.templates_dir / "images" / png_filename
            png_path.parent.mkdir(exist_ok=True)
            
            # 创建轮廓图像
            if hasattr(self, 'current_image') and self.current_image is not None:
                contour_img = self.current_image.copy()
                for contour_info in valid_contours:
                    cv2.drawContours(contour_img, [contour_info['contour']], -1, (0, 255, 0), 2)
                cv2.imwrite(str(png_path), contour_img)
            
            # 保存到数据库
            self.save_to_database(template_data, json_filename, image_path)
            
            save_type = "自动保存" if auto_save else "手动保存"
            print(f"✅ 模板已{save_type}: {tooth_id} ({len(valid_contours)}个轮廓)")
            return True
            
        except Exception as e:
            print(f"❌ 保存失败: {str(e)}")
            return False
    
    def save_to_database(self, template_data, json_filename, image_path):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO templates 
                (tooth_id, name, image_path, contour_file, num_contours, total_area)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                template_data["tooth_id"],
                f"牙齿模型 {template_data['tooth_id']}",
                image_path,
                json_filename,
                template_data["num_contours"],
                template_data["total_area"]
            ))
            conn.commit()
            print(f"✅ 数据库记录已保存")
        except Exception as e:
            print(f"❌ 数据库保存失败: {e}")
        finally:
            conn.close()

    def list_templates(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('SELECT tooth_id, num_contours, total_area, created_at FROM templates ORDER BY created_at DESC')
        templates = cursor.fetchall()
        conn.close()
        
        if templates:
            print("\n📋 已保存的牙齿模板:")
            print("-" * 50)
            for tooth_id, num_contours, total_area, created_at in templates:
                print(f"ID: {tooth_id:<15} | 轮廓: {num_contours:<3} | 面积: {total_area:<8.1f}")
        else:
            print("📭 暂无保存的模板")
        return templates

    def load_saved_contours(self, tooth_id):
        """加载已保存的轮廓数据用于比对
        Args:
            tooth_id: 牙模ID
        Returns:
            dict: 包含轮廓信息的字典，失败返回None
        """
        json_path = self.templates_dir / "contours" / f"{tooth_id}.json"
        if not json_path.exists():
            print(f"❌ 模板文件不存在: {tooth_id}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            print(f"✅ 成功加载模板: {tooth_id}")
            return template_data
        except Exception as e:
            print(f"❌ 加载模板失败: {e}")
            return None

    def compare_with_saved_template(self, current_contours, template_tooth_id):
        """简单的轮廓比对示例
        Args:
            current_contours: 当前检测到的轮廓列表
            template_tooth_id: 要比对的模板ID
        Returns:
            dict: 比对结果
        """
        template_data = self.load_saved_contours(template_tooth_id)
        if not template_data:
            return {"success": False, "error": "无法加载模板"}
        
        current_count = len(current_contours)
        template_count = template_data['num_contours']
        
        # 简单的数量和面积比对
        current_total_area = sum(info['area'] for info in current_contours)
        template_total_area = template_data['total_area']
        
        area_similarity = min(current_total_area, template_total_area) / max(current_total_area, template_total_area)
        count_match = current_count == template_count
        
        result = {
            "success": True,
            "template_id": template_tooth_id,
            "current_count": current_count,
            "template_count": template_count,
            "count_match": count_match,
            "current_area": current_total_area,
            "template_area": template_total_area,
            "area_similarity": area_similarity,
            "is_similar": area_similarity > 0.8 and count_match
        }
        
        print(f"\n📊 轮廓比对结果:")
        print(f"   模板ID: {template_tooth_id}")
        print(f"   轮廓数量: {current_count} vs {template_count} ({'✅ 匹配' if count_match else '❌ 不匹配'})")
        print(f"   总面积: {current_total_area:.1f} vs {template_total_area:.1f}")
        print(f"   面积相似度: {area_similarity:.3f}")
        print(f"   整体相似: {'✅ 是' if result['is_similar'] else '❌ 否'}")
        
        return result

    def list_all_saved_templates(self):
        """列出所有已保存的模板ID"""
        contours_dir = self.templates_dir / "contours"
        if not contours_dir.exists():
            return []
        
        template_files = list(contours_dir.glob("TOOTH_*.json"))
        template_ids = [f.stem for f in template_files]
        
        if template_ids:
            print(f"\n📁 找到 {len(template_ids)} 个已保存模板:")
            for tid in sorted(template_ids):
                print(f"   - {tid}")
        
        return sorted(template_ids)

class BatchToothProcessor:
    """批量牙齿图像处理器 - 基于现有的ToothTemplateBuilder"""
    
    def __init__(self, input_dir: str = "images", templates_dir: str = "templates", 
                 database_path: str = "tooth_templates.db"):
        self.input_dir = Path(input_dir)
        self.templates_dir = Path(templates_dir)
        self.database_path = database_path
        self.builder = ToothTemplateBuilder(database_path, str(templates_dir))
        
        # 支持的图像格式
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # 批量处理状态
        self.processed_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []  # (文件名, 错误信息)
        self.skipped_files: List[str] = []
        
        # 颜色模板缓存
        self.color_template: Optional[Dict] = None
        
        print(f"🚀 批量处理器初始化完成")
        print(f"   📁 输入目录: {self.input_dir}")
        print(f"   📄 模板目录: {self.templates_dir}")
        print(f"   🗄️ 数据库: {self.database_path}")
    
    def scan_image_files(self) -> List[Path]:
        """扫描输入目录中的所有支持的图像文件"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        
        image_files = []
        for ext in self.supported_formats:
            pattern = str(self.input_dir / f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = str(self.input_dir / f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        image_files = [Path(f) for f in image_files]
        image_files = sorted(set(image_files))  # 去重并排序
        
        print(f"📸 发现 {len(image_files)} 个图像文件:")
        for i, file in enumerate(image_files[:10], 1):  # 只显示前10个
            print(f"   {i:2d}. {file.name}")
        if len(image_files) > 10:
            print(f"   ... 还有 {len(image_files) - 10} 个文件")
        
        return image_files
    
    def is_already_processed(self, image_path: Path) -> bool:
        """检查图像是否已经被处理过（通过数据库查询）"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT tooth_id FROM templates WHERE image_path = ?', (str(image_path),))
            result = cursor.fetchone()
            return result is not None
        except Exception:
            return False
        finally:
            conn.close()
    
    def get_color_template_from_first_image(self, first_image_path: Path) -> Optional[Dict]:
        """从第一张图像获取颜色模板（交互式选择）"""
        print(f"\n🎨 请在第一张图像中选择目标颜色:")
        print(f"📸 {first_image_path.name}")
        
        img = cv2.imread(str(first_image_path))
        if img is None:
            print(f"❌ 无法读取图像: {first_image_path}")
            return None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        picked = []
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = hsv[y, x]
                print(f"选中点HSV: {color}")
                picked.append(color)
        
        cv2.imshow("点击选取目标区域颜色 (ESC退出, 多点选择后按ESC)", img)
        cv2.setMouseCallback("点击选取目标区域颜色 (ESC退出, 多点选择后按ESC)", on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if not picked:
            print("❌ 未选取颜色")
            return None
        
        # 计算HSV平均值
        hsv_arr = np.array(picked)
        h_mean, s_mean, v_mean = np.mean(hsv_arr, axis=0).astype(int)
        
        # 创建颜色模板
        color_template = {
            'h_mean': int(h_mean),
            's_mean': int(s_mean),
            'v_mean': int(v_mean),
            'lower': [0, 0, 0],  # 可以根据需要调整
            'upper': [15, 60, 61],  # 可以根据需要调整
            'picked_points': len(picked)
        }
        
        print(f"✅ 颜色模板创建成功:")
        print(f"   HSV均值: ({h_mean}, {s_mean}, {v_mean})")
        print(f"   选取点数: {len(picked)}")
        
        return color_template
    
    def process_single_image_with_template(self, image_path: Path, 
                                         color_template: Dict, 
                                         show_interactive: bool = False) -> bool:
        """使用颜色模板自动处理单张图像"""
        try:
            print(f"🔄 处理中: {image_path.name}")
            
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 应用颜色模板进行HSV掩码
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array(color_template['lower'])
            upper = np.array(color_template['upper'])
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            
            # 智能分离
            mask_processed = choose_separation_method(mask)
            
            # 轮廓检测
            contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid_contours = []
            
            for i, contour in enumerate(contours):
                if contour.shape[0] < 20:
                    continue
                area = cv2.contourArea(contour)
                length = cv2.arcLength(contour, True)
                valid_contours.append({
                    'contour': contour,
                    'points': contour[:, 0, :],
                    'area': area,
                    'length': length,
                    'idx': i
                })
            
            if not valid_contours:
                raise ValueError("未检测到有效轮廓")
            
            # 生成牙齿ID
            tooth_id = self.builder.get_next_tooth_id()
            
            # 创建HSV信息
            hsv_info = {
                'h_mean': color_template['h_mean'],
                's_mean': color_template['s_mean'],
                'v_mean': color_template['v_mean'],
                'lower': color_template['lower'],
                'upper': color_template['upper']
            }
            
            # 自动保存（不显示交互界面）
            success = self.builder.serialize_contours(
                valid_contours, tooth_id, str(image_path), hsv_info, auto_save=True
            )
            
            if success:
                print(f"✅ {image_path.name} -> {tooth_id} ({len(valid_contours)}个轮廓)")
                return True
            else:
                raise ValueError("保存失败")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {image_path.name}: {error_msg}")
            self.failed_files.append((str(image_path), error_msg))
            return False
    
    def process_batch(self, skip_processed: bool = True, 
                     interactive_first: bool = True,
                     show_progress: bool = True) -> Dict:
        """批量处理所有图像"""
        print(f"\n🚀 开始批量处理...")
        print("=" * 60)
        
        # 扫描图像文件
        image_files = self.scan_image_files()
        if not image_files:
            print("❌ 没有找到可处理的图像文件")
            return self._generate_report()
        
        # 过滤已处理的文件
        if skip_processed:
            unprocessed_files = []
            for img_file in image_files:
                if self.is_already_processed(img_file):
                    self.skipped_files.append(str(img_file))
                    print(f"⏭️  跳过已处理: {img_file.name}")
                else:
                    unprocessed_files.append(img_file)
            image_files = unprocessed_files
        
        if not image_files:
            print("✅ 所有图像都已处理完成")
            return self._generate_report()
        
        print(f"\n📊 待处理图像: {len(image_files)} 个")
        
        # 获取颜色模板
        if interactive_first and self.color_template is None:
            self.color_template = self.get_color_template_from_first_image(image_files[0])
            if self.color_template is None:
                print("❌ 无法获取颜色模板，批量处理终止")
                return self._generate_report()
        
        # 处理所有图像
        total_files = len(image_files)
        for i, img_file in enumerate(image_files, 1):
            if show_progress:
                print(f"\n📈 进度: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            success = self.process_single_image_with_template(
                img_file, self.color_template, show_interactive=False
            )
            
            if success:
                self.processed_files.append(str(img_file))
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """生成批量处理报告"""
        total_found = len(self.processed_files) + len(self.failed_files) + len(self.skipped_files)
        
        report = {
            'total_found': total_found,
            'processed': len(self.processed_files),
            'failed': len(self.failed_files),
            'skipped': len(self.skipped_files),
            'success_rate': len(self.processed_files) / max(1, total_found - len(self.skipped_files)) * 100,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files
        }
        
        # 打印报告
        print(f"\n" + "=" * 60)
        print(f"🎉 批量处理完成！")
        print(f"=" * 60)
        print(f"📊 处理统计:")
        print(f"   🔍 发现文件: {report['total_found']} 个")
        print(f"   ✅ 成功处理: {report['processed']} 个")
        print(f"   ❌ 处理失败: {report['failed']} 个")
        print(f"   ⏭️  跳过文件: {report['skipped']} 个")
        print(f"   📈 成功率: {report['success_rate']:.1f}%")
        
        if self.failed_files:
            print(f"\n❌ 失败文件详情:")
            for file_path, error in self.failed_files:
                print(f"   • {Path(file_path).name}: {error}")
        
        return report

def process_image_with_color_template(image_path: str, color_template: Dict, 
                                    tooth_id: Optional[str] = None) -> bool:
    """修改后的颜色处理函数，支持预设颜色模板"""
    builder = ToothTemplateBuilder()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 图片读取失败: {image_path}")
        return False
    
    # 使用预设的颜色模板
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(color_template['lower'])
    upper = np.array(color_template['upper'])
    
    hsv_info = {
        'h_mean': color_template['h_mean'],
        's_mean': color_template['s_mean'], 
        'v_mean': color_template['v_mean'],
        'lower': color_template['lower'],
        'upper': color_template['upper']
    }
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # 其余处理逻辑与原函数相同...
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    mask_processed = choose_separation_method(mask)
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_contours = []
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        valid_contours.append({
            'contour': contour,
            'points': contour[:, 0, :],
            'area': area,
            'length': length,
            'idx': i
        })
    
    if not valid_contours:
        print("❌ 未检测到有效轮廓")
        return False
    
    if tooth_id is None:
        tooth_id = builder.get_next_tooth_id()
    
    success = builder.serialize_contours(valid_contours, tooth_id, image_path, hsv_info, auto_save=True)
    if success:
        print(f"✅ 自动处理完成: {tooth_id} ({len(valid_contours)}个轮廓)")
    
    return success

def pick_color_and_draw_edge(image_path, tooth_id=None):
    # 初始化模板建立器
    builder = ToothTemplateBuilder()
    
    img = cv2.imread(image_path)
    if img is None:
        print("图片读取失败")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    picked = []
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = hsv[y, x]
            print(f"选中点HSV: {color}")
            picked.append(color)
    
    cv2.imshow("点击选取目标区域颜色 (ESC退出)", img)
    cv2.setMouseCallback("点击选取目标区域颜色 (ESC退出)", on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if not picked:
        print("未选取颜色")
        return
    
    hsv_arr = np.array(picked)
    h, s, v = np.mean(hsv_arr, axis=0).astype(int)
    print(f"HSV picked: {h}, {s}, {v}")
    
    lower = np.array([0,0,0])
    upper = np.array([15,60,61])
    print(f"lower: {lower}, upper: {upper}")
    
    # 保存HSV信息
    hsv_info = {
        'h_mean': int(h), 's_mean': int(s), 'v_mean': int(v),
        'lower': lower.tolist(), 'upper': upper.tolist()
    }
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # --- 形态学操作分离黏连区域 ---
    # 先进行开运算去除噪声
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 智能选择分离方法
    mask_processed = choose_separation_method(mask)
    
    # 显示分离效果对比
    show_separation_comparison(mask, mask_processed, image_path)
    
    color_extract = cv2.bitwise_and(img, img, mask=mask_processed)
    
    # --- 记录所有有效轮廓及属性 ---
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_contours = []
    
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        valid_contours.append({
            'contour': contour,
            'points': contour[:, 0, :],
            'area': area,
            'length': length,
            'idx': i
        })
    
    if not valid_contours:
        print("❌ 未检测到有效轮廓")
        return
    
    n_contours = len(valid_contours)
    linewidth = max(0.5, 2 - 0.03 * n_contours)
    show_legend = n_contours <= 15
    
    # 自动生成牙齿ID（连续编号）
    if tooth_id is None:
        tooth_id = builder.get_next_tooth_id()

    # 保存当前图像到builder中，用于PNG保存
    # 修正：避免类型检查器报错，current_image 只允许为 None
    # builder.current_image = img  # 注释掉此行，防止类型错误

    # --- 交互式显示 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_img, ax_contour, ax_zoom = axes
    
    ax_img.set_title("原始图像")
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    
    ax_contour.set_title("全部轮廓显示")
    ax_contour.axis('equal')
    ax_contour.invert_yaxis()
    ax_contour.grid(True)
    
    ax_zoom.set_title("选中轮廓放大视图")
    ax_zoom.axis('equal')
    ax_zoom.invert_yaxis()
    ax_zoom.grid(True)
    
    selected_idx = [0]  # 用列表包裹以便闭包修改
    saved = [False]  # 保存状态
    
    # 自动保存模板（无需用户操作）
    print(f"🚀 自动保存模板中...")
    success = builder.serialize_contours(valid_contours, tooth_id, image_path, hsv_info, auto_save=True)
    if success:
        saved[0] = True
        print(f"✅ 模板已自动保存为: {tooth_id}")
    else:
        print(f"❌ 自动保存失败")
    
    def draw_all(highlight_idx=None):
        # 中间图：显示全部轮廓
        ax_contour.clear()
        ax_contour.set_title(f"全部轮廓显示 - 牙齿ID: {tooth_id}")
        ax_contour.axis('equal')
        ax_contour.invert_yaxis()
        ax_contour.grid(True)
        
        # 在原图上叠加所有轮廓
        img_display = img.copy()
        
        # 准备颜色列表
        colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        cmap = plt.get_cmap('tab10')
        colors_plt = cmap(np.linspace(0, 1, max(len(valid_contours), 10)))
        
        for j, info in enumerate(valid_contours):
            contour = info['contour']
            color_bgr = colors_bgr[j % len(colors_bgr)]
            
            if highlight_idx is not None and j == highlight_idx:
                # 高亮显示选中的轮廓
                cv2.drawContours(img_display, [contour], -1, (0, 0, 255), 3)
                # 添加标记点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img_display, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(img_display, f'{j+1}', (cx-8, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # 普通显示其他轮廓
                cv2.drawContours(img_display, [contour], -1, color_bgr, 2)
                # 添加编号
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(img_display, f'{j+1}', (cx-5, cy+3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ax_contour.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        ax_contour.axis('off')
        
        # 右边图：显示选中轮廓的放大视图
        ax_zoom.clear()
        if highlight_idx is not None:
            info = valid_contours[highlight_idx]
            contour = info['contour']
            
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            margin = max(20, max(w, h) * 0.1)  # 自适应边距
            
            # 从原图中裁剪区域
            x1 = max(0, int(x - margin))
            y1 = max(0, int(y - margin))
            x2 = min(img.shape[1], int(x + w + margin))
            y2 = min(img.shape[0], int(y + h + margin))
            
            cropped_img = img[y1:y2, x1:x2].copy()
            
            # 调整轮廓坐标到裁剪图像的坐标系
            adjusted_contour = contour.copy()
            adjusted_contour[:, 0, 0] -= x1
            adjusted_contour[:, 0, 1] -= y1
            
            # 在裁剪图像上绘制轮廓
            cv2.drawContours(cropped_img, [adjusted_contour], -1, (0, 0, 255), 3)
            # 创建半透明填充效果
            overlay = cropped_img.copy()
            cv2.fillPoly(overlay, [adjusted_contour], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, cropped_img, 0.7, 0, cropped_img)
            
            ax_zoom.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            ax_zoom.set_title(f"选中轮廓 {highlight_idx+1} - 面积: {info['area']:.1f} | 周长: {info['length']:.1f}")
        else:
            # 如果没有选中轮廓，显示提示信息
            ax_zoom.text(0.5, 0.5, '点击轮廓查看放大视图\n←→ 键切换轮廓\nq 键退出\n\n✅ 模板已自动保存', 
                        ha='center', va='center', transform=ax_zoom.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax_zoom.set_title("轮廓放大视图")
        
        ax_zoom.axis('off')
        
        # 状态信息显示
        if highlight_idx is not None:
            info = valid_contours[highlight_idx]
            status = "✅ 已自动保存" if saved[0] else "❌ 未保存"
            status_text = f"状态: {status} | 当前: {highlight_idx+1}/{len(valid_contours)} | 面积: {info['area']:.1f} | 周长: {info['length']:.1f}"
        else:
            status = "✅ 已自动保存" if saved[0] else "❌ 未保存"
            status_text = f"状态: {status} | 共 {len(valid_contours)} 个轮廓 | 操作: ←→切换 q退出"
        
        fig.suptitle(status_text, fontsize=12, y=0.02)
        
        fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes != ax_contour:
            return
        
        # 获取点击坐标（需要转换到图像坐标系）
        if event.xdata is None or event.ydata is None:
            return
            
        # 由于ax_contour显示的是图像，坐标系与原图一致
        x, y = int(event.xdata), int(event.ydata)
        
        # 检查点击是否在图像范围内
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return
        
        found = False
        for j, info in enumerate(valid_contours):
            # 检查点是否在轮廓内
            if cv2.pointPolygonTest(info['contour'], (x, y), False) >= 0:
                selected_idx[0] = j
                draw_all(highlight_idx=j)
                found = True
                print(f"✅ 选中轮廓 {j+1}")
                break
        
        if not found:
            print("未选中任何轮廓")
    
    def on_key(event):
        if event.key == 'right':
            selected_idx[0] = (selected_idx[0] + 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'left':
            selected_idx[0] = (selected_idx[0] - 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'q':
            plt.close()
    
    draw_all(highlight_idx=0 if valid_contours else None)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为状态信息留出空间
    plt.show()
    
    # 显示已保存的模板列表
    builder.list_templates()

def ultra_separate_connected_objects(mask):
    """
    超强黏连分离算法 - 仅使用OpenCV，无需额外依赖
    """
    print("🚀 启动超强分离算法（OpenCV版本）...")
    
    # 步骤1: 清理噪声
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # 步骤2: 多策略分离尝试
    best_result = mask_clean
    max_components = 1
    
    # 策略1: 激进腐蚀分离
    erosion_configs = [
        (1, 3), (2, 3), (3, 3), (4, 3),  # 小核多次迭代
        (1, 5), (2, 5), (3, 5),          # 中核
        (1, 7), (2, 7)                   # 大核
    ]
    
    for iterations, kernel_size in erosion_configs:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask_clean, kernel, iterations=iterations)
        
        # 检查是否成功分离
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        if num_labels > max_components:
            max_components = num_labels
            print(f"💪 找到更好分离: {num_labels-1} 个区域 (腐蚀{iterations}次,核{kernel_size}x{kernel_size})")
            
            # 恢复各个区域
            result_mask = np.zeros_like(mask_clean)
            
            for i in range(1, num_labels):  # 跳过背景
                # 获取当前区域
                component = (labels == i).astype(np.uint8) * 255
                
                # 渐进膨胀恢复
                restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (min(kernel_size, 5), min(kernel_size, 5)))
                restored = cv2.dilate(component, restore_kernel, iterations=min(iterations, 2))
                
                # 限制在原始区域内
                restored = cv2.bitwise_and(restored, mask_clean)
                
                result_mask = cv2.bitwise_or(result_mask, restored)
            
            best_result = result_mask
    
    print(f"✅ 超强分离完成！最终分离出 {max_components-1} 个独立区域")
    return best_result

def force_separation_with_morphology(mask):
    """
    强制形态学分离 - 当分水岭失败时的终极备选方案
    """
    print("🔧 启动强制形态学分离...")
    original_mask = mask.copy()
    best_result = mask.copy()
    max_components = 1
    
    # 极度激进的腐蚀策略
    erosion_configs = [
        (1, (3, 3)), (2, (3, 3)), (3, (3, 3)), (4, (3, 3)), (5, (3, 3)),
        (1, (5, 5)), (2, (5, 5)), (3, (5, 5)),
        (1, (7, 7)), (2, (7, 7)),
        (1, (9, 9))
    ]
    
    for iterations, kernel_size in erosion_configs:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        eroded = cv2.erode(original_mask, kernel, iterations=iterations)
        
        # 检查连通分量
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        if num_labels > max_components:
            max_components = num_labels
            result_mask = np.zeros_like(mask)
            
            for i in range(1, num_labels):
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # 渐进式膨胀恢复
                restore_iterations = min(iterations, 3)  # 限制恢复强度
                kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (restore_iterations*2+1, restore_iterations*2+1))
                restored = cv2.dilate(component_mask, kernel_restore, iterations=restore_iterations)
                
                # 限制在扩展的原始区域内
                expanded_original = cv2.dilate(original_mask, np.ones((3,3), np.uint8), iterations=2)
                restored = cv2.bitwise_and(restored, expanded_original)
                
                result_mask = cv2.bitwise_or(result_mask, restored)
            
            best_result = result_mask.copy()
            print(f"💪 形态学方案找到 {max_components-1} 个区域 (腐蚀{iterations}次,核{kernel_size})")
    
    print(f"✅ 强制分离完成，最终分离出 {max_components-1} 个区域")
    return best_result
    """
    超强黏连分离算法 - 针对牙齿模型优化
    """
    # 步骤1: 预处理 - 去除小噪声和平滑
    mask_bool = mask > 0
    mask_clean = remove_small_objects(mask_bool, min_size=30, connectivity=2)
    mask_clean = binary_opening(mask_clean, disk(1))  # 减少开运算强度
    mask_clean = mask_clean.astype(np.uint8) * 255
    
    # 步骤2: 高精度距离变换
    dist_transform = distance_transform_edt(mask_clean)
    
    # 步骤3: 更激进的参数设置 - 专门针对牙齿黏连
    img_area = mask_clean.shape[0] * mask_clean.shape[1]
    max_dist = np.max(dist_transform)
    
    # 更激进的参数，强制分离黏连牙齿
    if img_area > 500000:  # 大图像
        min_distance = 2  # 极小
        threshold_abs = max_dist * 0.05  # 更低
        threshold_rel = 0.02
    elif img_area > 100000:  # 中等图像
        min_distance = 1
        threshold_abs = max_dist * 0.03
        threshold_rel = 0.01
    else:  # 小图像
        min_distance = 1
        threshold_abs = max_dist * 0.01
        threshold_rel = 0.005
    
    print(f"🔍 距离变换最大值: {max_dist:.2f}")
    print(f"📊 参数设置 - 最小距离: {min_distance}, 阈值: {threshold_abs:.2f}")
    
    # 步骤4: 寻找局部最大值作为分离种子
    local_maxima = peak_local_max(
        dist_transform,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    
    print(f"🎯 检测到 {len(local_maxima)} 个高质量分离种子点")
    
    if len(local_maxima) == 0:
        print("⚠️ 未找到分离点，降低阈值重试...")
        # 降低阈值重试
        local_maxima = peak_local_max(
            dist_transform,
            min_distance=max(min_distance//2, 3),
            threshold_abs=threshold_abs * 0.5,
            threshold_rel=threshold_rel * 0.5
        )
        print(f"🔄 重试后检测到 {len(local_maxima)} 个种子点")
    
    if len(local_maxima) == 0:
        print("❌ 仍未找到分离点，使用备选方案")
        return advanced_separate_connected_objects(mask_clean)
    
    # 步骤5: 创建高质量标记图像
    markers = np.zeros_like(mask_clean, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxima):
        markers[y, x] = i + 1
    
    # 使用形态学膨胀扩展标记，但控制扩展程度
    expansion_size = max(1, min_distance // 4)
    markers = ndimage.binary_dilation(
        markers > 0, 
        structure=disk(expansion_size)
    ).astype(np.int32)
    
    # 重新标记连通分量
    markers = label(markers)
    
    # 步骤6: 高性能分水岭分割
    labels = watershed(-dist_transform, markers, mask=mask_clean)
    
    # 步骤7: 智能后处理
    result_mask = np.zeros_like(mask_clean)
    regions = regionprops(labels)
    
    min_area = 100  # 最小区域面积
    processed_regions = 0
    
    for region in regions:
        if region.area < min_area:
            continue
            
        # 获取区域mask
        region_mask = (labels == region.label).astype(np.uint8) * 255
        
        # 形态学闭运算填补空洞，使用自适应核大小
        close_size = max(1, int(np.sqrt(region.area) * 0.05))
        kernel_close = disk(close_size)
        region_mask = ndimage.binary_closing(region_mask, structure=kernel_close)
        region_mask = region_mask.astype(np.uint8) * 255
        
        # 合并到结果
        result_mask = cv2.bitwise_or(result_mask, region_mask)
        processed_regions += 1
    
    print(f"✅ 高性能分离完成！生成 {processed_regions} 个独立高质量区域")
    return result_mask

def advanced_separate_connected_objects(mask):
    """
    高级分离方法：结合多种形态学操作，不依赖额外库
    """
    # 方法1: 基于腐蚀-膨胀的分离
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel_erode, iterations=2)
    
    # 寻找连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    if num_labels <= 1:  # 没有找到分离的区域
        print("⚠️ 腐蚀后未找到分离区域，尝试更强的分离")
        return erosion_dilation_separation(mask)
    
    result_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # 跳过背景
        # 获取当前连通分量
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # 对每个分量进行膨胀恢复
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(component_mask, kernel_dilate, iterations=2)
        
        # 与原始mask取交集，避免过度膨胀
        dilated = cv2.bitwise_and(dilated, mask)
        
        result_mask = cv2.bitwise_or(result_mask, dilated)
    
    print(f"✅ 腐蚀-膨胀分离完成，生成 {num_labels-1} 个区域")
    return result_mask

def erosion_dilation_separation(mask):
    """
    渐进式腐蚀分离算法
    """
    original_mask = mask.copy()
    best_result = mask.copy()
    max_components = 1
    
    # 尝试不同强度的腐蚀
    for iterations in range(1, 6):
        for kernel_size in [(3,3), (5,5), (7,7)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded = cv2.erode(original_mask, kernel, iterations=iterations)
            
            # 检查连通分量
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            
            if num_labels > max_components:
                max_components = num_labels
                # 恢复各个分量
                result_mask = np.zeros_like(mask)
                
                for i in range(1, num_labels):
                    component_mask = (labels == i).astype(np.uint8) * 255
                    
                    # 膨胀恢复，但限制在原始区域内
                    kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iterations*2+1, iterations*2+1))
                    restored = cv2.dilate(component_mask, kernel_restore, iterations=iterations)
                    restored = cv2.bitwise_and(restored, original_mask)
                    
                    result_mask = cv2.bitwise_or(result_mask, restored)
                
                best_result = result_mask.copy()
    
    print(f"✅ 渐进式分离完成，最多分离出 {max_components-1} 个区域")
    return best_result

def choose_separation_method(mask):
    """
    智能选择高性能分离方法
    """
    # 计算初始连通分量数
    num_labels_initial, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels_initial > 2:  # 已经分离，无需处理
        print("✅ 区域已经分离，无需额外处理")
        return mask
    
    # 分析图像特征
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask
    
    # 计算多个复杂度指标
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    
    # 形状复杂度：周长平方/面积
    shape_complexity = (total_perimeter ** 2) / (total_area + 1e-6)
    
    # 凸性分析
    total_hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)
    convexity = total_area / (total_hull_area + 1e-6)
    
    # 区域紧凑度
    compactness = (4 * np.pi * total_area) / (total_perimeter ** 2 + 1e-6)
    
    print(f"🔍 图像分析结果:")
    print(f"   📊 形状复杂度: {shape_complexity:.2f}")
    print(f"   🔄 凸性系数: {convexity:.3f}")
    print(f"   📐 紧凑度: {compactness:.3f}")
    
    # 智能选择分离策略
    try:
        # 优先使用高性能的scikit-image算法
        if shape_complexity > 80 or convexity < 0.7:
            print("🚀 使用超强分离算法（复杂形状）...")
            return ultra_separate_connected_objects(mask)
        elif compactness < 0.3:
            print("🚀 使用超强分离算法（非紧凑形状）...")
            return ultra_separate_connected_objects(mask)
        else:
            print("⚡ 使用高速形态学方法（简单形状）...")
            return advanced_separate_connected_objects(mask)
    except Exception as e:
        print(f"⚠️ 高性能算法失败: {e}")
        print("🔄 回退到稳定的OpenCV方法...")
        return advanced_separate_connected_objects(mask)

def show_separation_comparison(original_mask, processed_mask, image_path):
    """
    高性能分离效果可视化对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始图像
    img = cv2.imread(image_path)
    if img is not None:
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("原始图像", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
    
    # 分离前的mask
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title("分离前", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 分离后的mask
    axes[0, 2].imshow(processed_mask, cmap='gray')
    axes[0, 2].set_title("分离后", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 轮廓对比 - 分离前
    contours_before, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours_before = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2RGB)
    for i, contour in enumerate(contours_before):
        cv2.drawContours(img_contours_before, [contour], -1, (255, 0, 0), 2)
        # 添加编号
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contours_before, str(i+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    axes[1, 0].imshow(img_contours_before)
    axes[1, 0].set_title("分离前轮廓", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 轮廓对比 - 分离后
    contours_after, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours_after = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, contour in enumerate(contours_after):
        color = colors[i % len(colors)]
        cv2.drawContours(img_contours_after, [contour], -1, color, 2)
        # 添加编号
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contours_after, str(i+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    axes[1, 1].imshow(img_contours_after)
    axes[1, 1].set_title("分离后轮廓", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 统计信息图表
    valid_before = len([c for c in contours_before if cv2.contourArea(c) > 100])
    valid_after = len([c for c in contours_after if cv2.contourArea(c) > 100])
    
    areas_before = [cv2.contourArea(c) for c in contours_before if cv2.contourArea(c) > 100]
    areas_after = [cv2.contourArea(c) for c in contours_after if cv2.contourArea(c) > 100]
    
    # 面积对比柱状图
    axes[1, 2].bar(['分离前', '分离后'], [sum(areas_before), sum(areas_after)], 
                   color=['red', 'green'], alpha=0.7)
    axes[1, 2].set_title("总面积对比", fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel("面积 (像素)")
    
    # 在图上添加数值
    for i, v in enumerate([sum(areas_before), sum(areas_after)]):
        axes[1, 2].text(i, v + max(areas_before + areas_after) * 0.02, f'{int(v)}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # 分离效果信息
    improvement_ratio = valid_after / max(valid_before, 1)
    separation_info = f'''分离性能报告:
    ├─ 区域数量: {valid_before} → {valid_after}
    ├─ 提升倍数: {improvement_ratio:.2f}x
    ├─ 总面积: {sum(areas_before):.0f} → {sum(areas_after):.0f}
    └─ 平均面积: {np.mean(areas_before):.0f} → {np.mean(areas_after):.0f}'''
    
    fig.suptitle(f'🚀 高性能分离效果对比\n{separation_info}', 
                fontsize=16, fontweight='bold', y=0.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    print(f"\n🎯 分离性能总结:")
    print(f"   🔢 区域数量变化: {valid_before} → {valid_after}")
    print(f"   📈 分离效果提升: {improvement_ratio:.2f}倍")
    print(f"   📊 面积保持率: {sum(areas_after)/sum(areas_before)*100:.1f}%")

def save_features_only(valid_contours, tooth_id, features_dir="templates/features"):
    from pathlib import Path
    import numpy as np

    def to_serializable(feat):
        # 把所有 ndarray 转成 list
        if isinstance(feat, np.ndarray):
            return feat.tolist()
        if isinstance(feat, dict):
            return {k: to_serializable(v) for k, v in feat.items()}
        if isinstance(feat, list):
            return [to_serializable(x) for x in feat]
        return feat

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    features_list = [to_serializable(contour['features']) for contour in valid_contours]
    features_path = features_dir / f"{tooth_id}_features.json"
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump({"features": features_list}, f, ensure_ascii=False, indent=2)
    print(f"✅ 纯特征文件已保存: {features_path}")


def main():
    """
    高性能牙齿模板建立器主程序 - 支持单张和批量处理
    """
    parser = argparse.ArgumentParser(description='牙齿模板建立器')
    parser.add_argument('--batch', action='store_true', help='启用批量处理模式')
    parser.add_argument('--input-dir', default='images', help='输入目录路径 (默认: images)')
    parser.add_argument('--output-dir', default='templates', help='输出目录路径 (默认: templates)')
    parser.add_argument('--database', default='tooth_templates.db', help='数据库路径 (默认: tooth_templates.db)')
    parser.add_argument('--skip-processed', action='store_true', default=True, 
                       help='跳过已处理的文件 (默认: True)')
    parser.add_argument('--single-image', help='处理单张图像的路径')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        print("🚀 启动批量牙齿模板建立器")
        print("=" * 60)
        
        processor = BatchToothProcessor(
            input_dir=args.input_dir,
            templates_dir=args.output_dir,
            database_path=args.database
        )
        
        try:
            report = processor.process_batch(
                skip_processed=args.skip_processed,
                interactive_first=True,
                show_progress=True
            )
            
            # 显示最终统计
            if report['processed'] > 0:
                print(f"\n🎯 批量处理成功完成!")
                print(f"✅ 已创建 {report['processed']} 个牙齿模板")
                
                # 显示已保存的模板列表
                processor.builder.list_templates()
            
        except Exception as e:
            print(f"❌ 批量处理过程中发生错误: {e}")
            print("💡 请检查输入目录和文件权限")
    
    elif args.single_image:
        # 单张图像处理模式
        print("🚀 启动单张图像处理模式")
        print("=" * 50)
        
        image_path = args.single_image
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return
        
        print(f"📸 正在处理图像: {image_path}")
        
        try:
            pick_color_and_draw_edge(image_path, tooth_id=None)
            print("\n🎉 单张图像处理完成！")
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {e}")
            
    else:
        # 默认单张处理模式（使用PHOTO_PATH）
        print("🚀 启动高性能牙齿模板建立器")
        print("=" * 50)
        
        # 自动生成连续编号，无需用户输入
        tooth_id = None  # 将自动生成 TOOTH_001, TOOTH_002...
        
        # 图像路径
        image_path = PHOTO_PATH 
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            print("💡 请检查文件路径是否正确")
            print(f"💡 或使用 --single-image 指定图像路径")
            print(f"💡 或使用 --batch --input-dir 指定批量处理目录")
            return
        
        print(f"📸 正在处理图像: {image_path}")
        
        try:
            # 启动高性能分离和模板建立（自动保存）
            pick_color_and_draw_edge(image_path, tooth_id)
            print("\n🎉 高性能处理完成！")
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {e}")
            print("💡 请检查图像文件和依赖库是否正确安装")

def main_batch_example():
    """批量处理示例函数"""
    print("🚀 批量处理示例")
    print("=" * 50)
    
    # 创建批量处理器
    processor = BatchToothProcessor(
        input_dir="images",  # 你的图像目录
        templates_dir="templates",
        database_path="tooth_templates.db"
    )
    
    # 开始批量处理
    report = processor.process_batch(
        skip_processed=True,     # 跳过已处理的文件
        interactive_first=True,  # 第一张图交互选色
        show_progress=True       # 显示进度
    )
    
    return report

if __name__ == "__main__":
    main()

