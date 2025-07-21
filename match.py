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
from datetime import datetime
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先黑体、雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['font.size'] = 10

#TODO 修改图片路径
PHOTO_PATH = r'c:\Users\Jason\Desktop\tooth\Tooth_5.png'
# 配置常量
class Config:
    DEFAULT_HSV_TOLERANCE = {'h': 15, 's': 60, 'v': 60}
    FOURIER_ORDER = 80
    MIN_CONTOUR_POINTS = 20
    SIMILARITY_THRESHOLD = 0.99  # 改为1.0作为临界值
    SIZE_TOLERANCE = 0.3
    DATABASE_PATH = "tooth_templates.db"
    TEMPLATES_DIR = "templates"
  
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

import os
import json

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
    """牙齿匹配器主类 - 增强版"""
    
    def __init__(self):
        self.feature_extractor = ContourFeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.fourier_analyzer = FourierAnalyzer()
        self.db_interface = DatabaseInterface()
        self.templates = load_features_templates()
        self.current_image_path = None
    
    def load_templates(self):
        """加载模板库"""
        self.templates = load_features_templates()
        return len(self.templates) > 0
    
    def match_against_database(self, query_features_list, threshold=Config.SIMILARITY_THRESHOLD):
        """与数据库模板进行匹配"""
        if not self.templates:
            logger.warning("❌ 未加载模板数据，请先使用 BuildTheLab 创建模板")
            return {}
        all_matches = {}
        for query_idx, query_features in enumerate(query_features_list):
            query_matches = []
            for template_id, template_features_list in self.templates.items():
                for template_idx, template_features in enumerate(template_features_list):
                    similarities = self.similarity_calculator.compare_contours_approx(
                        query_features, template_features, rel_tol=0.01, abs_tol=0.1)
                    # 删除详细调试输出
                    if similarities['overall'] >= threshold:
                        match_info = {
                            'template_id': template_id,
                            'template_contour_idx': template_idx,
                            'similarity': similarities['overall'],
                            'details': similarities,
                            'query_contour_idx': query_idx
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
        """处理图像的主函数"""
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
        
        # 修正 query_features_list 的生成方式
        query_features_list = [c['contours'] for c in valid_contours]
        matches = self.match_against_database(query_features_list)
        
        # 显示交互式界面
        self._show_interactive_display(color_extract, valid_contours, all_contours, matches)
    
    def _pick_colors(self, img: np.ndarray, hsv: np.ndarray) -> list:
        """颜色选择"""
        picked = []
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = hsv[y, x]
                logger.info(f"选中点HSV: {color}")
                picked.append(color)
        
        cv2.imshow("点击选取目标区域颜色", img)
        cv2.setMouseCallback("点击选取目标区域颜色", on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return picked
    
    def _create_mask(self, hsv: np.ndarray, picked_colors: list) -> np.ndarray:
        """创建颜色掩码"""
        hsv_arr = np.array(picked_colors)
        h, s, v = np.mean(hsv_arr, axis=0).astype(int)
        logger.info(f"HSV picked: {h}, {s}, {v}")
        
        tolerance = Config.DEFAULT_HSV_TOLERANCE
        lower = np.array([0,0,0])
        upper = np.array([15,60,61])
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
                'contours': contours
            })
            all_contours.append(contours)
        return valid_contours, all_contours
    
    def _show_interactive_display(self, color_extract: np.ndarray, 
                                 valid_contours: list, all_contours: list, matches):
        """显示交互式界面 - 增强版"""
        n_contours = len(valid_contours)
        linewidth = max(0.5, 2 - 0.03 * n_contours)
        show_legend = n_contours <= 15
        
        # 如果有模板库，使用更大的布局
        if self.templates:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            ax_img, ax_fit, ax_zoom = axes[0]
            ax_db_matches, ax_stats, ax_history = axes[1]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            ax_img, ax_fit, ax_zoom = axes
            ax_db_matches = ax_stats = ax_history = None
        
        ax_img.set_title("颜色提取结果", fontproperties=myfont)
        ax_img.imshow(cv2.cvtColor(color_extract, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
        
        ax_fit.set_title("轮廓显示", fontproperties=myfont)
        ax_fit.axis('equal')
        ax_fit.invert_yaxis()
        ax_fit.grid(True)
        
        ax_zoom.set_title("色块放大视图", fontproperties=myfont)
        ax_zoom.axis('equal')
        ax_zoom.invert_yaxis()
        ax_zoom.grid(True)
        
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
            # 显示模板库统计
            total_templates = len(self.templates)
            total_contours = sum(len(t) for t in self.templates.values()) # 这里需要调整，因为 templates 现在是 dict
            stats_text = f"模板库统计:\n总模板数: {total_templates}\n总轮廓数: {total_contours}\n\n"
            stats_text += "模板列表:\n"
            for i, (template_id, data) in enumerate(list(self.templates.items())[:10]):
                stats_text += f"{i+1}. {template_id} ({len(data)}个轮廓)\n" # 这里需要调整，因为 data 是 list
            if total_templates > 10:
                stats_text += f"... 还有 {total_templates-10} 个模板"
            if ax_stats is not None:
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                             fontsize=10, verticalalignment='top', fontproperties=myfont)
        
        selected_idx = [0]
        
        def draw_all(highlight_idx=None):
            self._draw_contours_enhanced(ax_fit, ax_zoom, valid_contours, all_contours, 
                                        highlight_idx, linewidth, show_legend, fig,
                                        ax_db_matches if self.templates else None, matches)
        
        def on_click(event):
            if self.templates and event.inaxes not in [ax_img, ax_fit, ax_zoom, ax_db_matches]:
                return
            elif not self.templates and event.inaxes not in [ax_img, ax_fit, ax_zoom]:
                return
                
            x, y = int(event.xdata), int(event.ydata)
            for j, info in enumerate(valid_contours):
                if cv2.pointPolygonTest(info['contour'], (x, y), False) >= 0:
                    selected_idx[0] = j
                    draw_all(highlight_idx=j)
                    break
        
        def on_key(event):
            if event.key == 'right':
                selected_idx[0] = (selected_idx[0] + 1) % n_contours
                draw_all(highlight_idx=selected_idx[0])
            elif event.key == 'left':
                selected_idx[0] = (selected_idx[0] - 1) % n_contours
                draw_all(highlight_idx=selected_idx[0])
        
        draw_all(highlight_idx=selected_idx[0])
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()
    
    def _draw_contours_enhanced(self, ax_fit, ax_zoom, valid_contours, all_contours, 
                               highlight_idx, linewidth, show_legend, fig, ax_db_matches=None, matches=None):
        """绘制轮廓 - 增强版"""
        ax_fit.clear()
        ax_fit.set_title(f"轮廓显示 (模板库: {'已加载' if self.templates else '未加载'})", fontproperties=myfont)
        ax_fit.axis('equal')
        ax_fit.invert_yaxis()
        ax_fit.grid(True)
        
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, max(len(valid_contours), 10)))
        
        # 找到相似轮廓（当前图像内部）
        similar_contours = []
        database_matches = {}
        
        if highlight_idx is not None:
            target_contours = valid_contours[highlight_idx]['contours']
            
            # 当前图像内相似轮廓
            similar_contours = self.find_similar_contours(target_contours, all_contours)
            
            # 数据库匹配
            if self.templates:
                query_features_list = [valid_contours[highlight_idx]['contours']]
                database_matches = self.match_against_database(query_features_list)
        
        # 绘制所有轮廓
        for j, info in enumerate(valid_contours):
            points = info['points']
            label = f"色块{info['idx']+1}"
            
            # 检查是否为相似轮廓
            is_similar = any(sim['index'] == j for sim in similar_contours)
            
            # 设置颜色和样式
            if highlight_idx is not None and j == highlight_idx:
                fill_color, edge_color = 'red', 'darkred'
                lw, alpha, zorder = linewidth * 2, 0.7, 10
                text_color = 'white'
            elif is_similar:
                fill_color, edge_color = 'orange', 'darkorange'
                lw, alpha, zorder = linewidth * 1.5, 0.6, 5
                text_color = 'black'
                sim_info = next(sim for sim in similar_contours if sim['index'] == j)
                label += f" (相似度:{sim_info['similarity']:.2f})"
            else:
                fill_color, edge_color = colors[j % len(colors)], 'black'
                lw, alpha, zorder = linewidth, 0.5, 1
                text_color = 'black'
            
            x, y = points[:, 0], points[:, 1]
            
            # 绘制填充和边界
            ax_fit.fill(x, y, color=fill_color, alpha=alpha, zorder=zorder, 
                       label=label if show_legend else None)
            ax_fit.plot(x, y, '-', color=edge_color, linewidth=lw, zorder=zorder+1)
            
            # 标注编号
            center_x, center_y = np.mean(x), np.mean(y)
            fontsize = max(8, min(14, int(np.sqrt(info['area']) / 10)))
            
            ax_fit.text(center_x, center_y, str(info['idx']+1), 
                       fontsize=fontsize, fontweight='bold', 
                       color=text_color, ha='center', va='center', zorder=zorder+2)
        
        if show_legend:
            ax_fit.legend(prop=myfont)
        
        # 显示特征信息和数据库匹配结果
        self._update_info_display_enhanced(ax_fit, ax_zoom, valid_contours, all_contours, 
                                          highlight_idx, similar_contours, database_matches, 
                                          fig, ax_db_matches, matches)
    
    def _update_info_display_enhanced(self, ax_fit, ax_zoom, valid_contours, all_contours, 
                                     highlight_idx, similar_contours, database_matches, 
                                     fig, ax_db_matches=None, matches=None):
        """更新信息显示 - 增强版"""
        info = valid_contours[highlight_idx if highlight_idx is not None else 0]
        contours = info['contours']
        
        # 构建特征信息
        feature_info = self._build_feature_info_enhanced(info, contours, similar_contours, 
                                                        valid_contours, database_matches, highlight_idx)
        
        # 显示特征信息
        ax_fit.text(0.02, -0.25, feature_info, transform=ax_fit.transAxes, 
                   fontsize=8, color='red', va='top', ha='left', fontproperties=myfont)
        
        # 更新放大
        self._update_zoom_view(ax_zoom, info, highlight_idx)
        
        # 更新数据库匹配视图
        if ax_db_matches and database_matches:
            print("当前all_matches keys:", database_matches.keys(), "当前highlight_idx:", highlight_idx)
            self._update_database_matches_view(ax_db_matches, database_matches, highlight_idx)
        
        fig.canvas.draw_idle()
    
    def _build_feature_info_enhanced(self, info, contours, similar_contours, 
                                    valid_contours, database_matches, highlight_idx):
        """构建增强的特征信息字符串"""
        feature_info = f"色块编号: {info['idx']+1}\n"
        feature_info += f"面积: {contours['area']:.2f}\n"
        feature_info += f"周长: {contours['perimeter']:.2f}\n"
        feature_info += f"长宽比: {contours['aspect_ratio']:.3f}\n"
        feature_info += f"圆形度: {contours['circularity']:.3f}\n"
        feature_info += f"凸度: {contours['solidity']:.3f}\n"
        feature_info += f"角点数: {contours['corner_count']}\n"
        
        # 当前图像相似轮廓信息
        if highlight_idx is not None and similar_contours:
            feature_info += f"\n当前图像相似轮廓:\n"
            for sim in similar_contours[:2]:
                details = sim['details']
                feature_info += f"  色块{valid_contours[sim['index']]['idx']+1}: {sim['similarity']:.3f}\n"
        
        # 数据库匹配信息
        if database_matches and f'query_{highlight_idx}' in database_matches:
            matches = database_matches[f'query_{highlight_idx}']
            if matches:
                feature_info += f"\n🏆 数据库匹配 (前2名):\n"
                for match in matches[:2]:
                    feature_info += f"  {match['template_id']}: {match['similarity']:.3f}\n"
            else:
                feature_info += f"\n❌ 无数据库匹配"
        
        return feature_info
    
    def _update_database_matches_view(self, ax_db_matches, database_matches, highlight_idx):
        """更新数据库匹配视图"""
        ax_db_matches.clear()
        ax_db_matches.set_title("数据库匹配结果", fontproperties=myfont)
        ax_db_matches.axis('off')

        key = f'query_{highlight_idx}'
        if key in database_matches:
            matches = database_matches[key]
            if matches:
                match_text = f"🎯 色块 {highlight_idx+1} 的数据库匹配:\n\n"
                match_text += f"{'排名':<4} {'模板ID':<15} {'相似度':<8} {'详细分数'}\n"
                match_text += "-" * 50 + "\n"
                for i, match in enumerate(matches[:8]):
                    details = match['details']
                    match_text += f"{i+1:<4} {match['template_id']:<15} {match['similarity']:<8.3f} "
                    match_text += f"几何:{details['geometric']:.2f} Hu:{details['hu_moments']:.2f}\n"
                if len(matches) > 8:
                    match_text += f"\n... 还有 {len(matches)-8} 个匹配"
            else:
                match_text = f"❌ 色块 {highlight_idx+1} 无数据库匹配\n\n"
                match_text += "可能原因:\n"
                match_text += "• 相似度低于阈值 (0.99)\n"
                match_text += "• 模板库中无相似轮廓\n"
                match_text += "• 特征提取失败"
        else:
            match_text = f"❌ 色块 {highlight_idx+1} 无数据库匹配\n\n"
            match_text += "可能原因:\n"
            match_text += "• 相似度低于阈值 (0.99)\n"
            match_text += "• 模板库中无相似轮廓\n"
            match_text += "• 特征提取失败"

        ax_db_matches.text(0.05, 0.95, match_text, transform=ax_db_matches.transAxes, 
                          fontsize=9, verticalalignment='top', fontproperties=myfont)

    def _update_zoom_view(self, ax_zoom, info, highlight_idx):
        contours = info['contours']  # ← 这行是关键
        ax_zoom.clear()
        contour = info['contour']
        x, y, w, h = cv2.boundingRect(contour)
        margin = max(20, int(0.1 * max(w, h)))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin

        # 假设info里有原始图像（如info['image']），否则只能画轮廓
        if 'image' in info:
            img = info['image']
            crop = img[y1:y2, x1:x2].copy()
            adjusted_contour = contour.copy()
            adjusted_contour[:, 0, 0] -= x1
            adjusted_contour[:, 0, 1] -= y1
            cv2.drawContours(crop, [adjusted_contour], -1, (0, 0, 255), 2)
            ax_zoom.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            # 只画轮廓
            points = info['points']
            ax_zoom.plot(points[:, 0], points[:, 1], 'r-')
            ax_zoom.fill(points[:, 0], points[:, 1], alpha=0.3)
            ax_zoom.set_aspect('equal')
        ax_zoom.set_title(f'色块放大视图 {info["idx"]+1}', fontproperties=myfont)
        ax_zoom.axis('off')

        if 'fourier_x_fit' in contours and 'fourier_y_fit' in contours:
            ax_zoom.plot(contours['fourier_x_fit'], contours['fourier_y_fit'], 'g--', linewidth=2, label='傅里叶平滑')



def main():
    """主函数"""
    print("🦷 牙齿匹配系统")
    print("1. 分析当前图像")
    print("2. 与模板库匹配")
    
    choice = input("请选择模式 (1/2, 默认2): ").strip()
    
    image_path = input(f"请输入图片路径 (默认: {PHOTO_PATH}): ").strip()
    if not image_path:
        image_path = PHOTO_PATH
    
    try:
        matcher = ToothMatcher()
        matcher.process_image(image_path)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()
