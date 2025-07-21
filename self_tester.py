import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import logging
import time

from match import ToothMatcher, Config
from BulidTheLab import ToothTemplateBuilder
from cloud_downloader import CloudImageDownloader
from config import TEST_CONFIG, UI_CONFIG, CLOUD_CONFIG

logger = logging.getLogger(__name__)

class SelfTester(ToothMatcher):
    """自我测试器 - 基于云盘图片的模板匹配测试"""
    
    def __init__(self, test_results_db: str = None):
        try:
            super().__init__()
        except FileNotFoundError as e:
            logger.warning(f"模板目录不存在，将在首次使用时创建: {e}")
            from pathlib import Path
            templates_dir = Path("templates")
            templates_dir.mkdir(exist_ok=True)
            (templates_dir / "features").mkdir(exist_ok=True)
            (templates_dir / "contours").mkdir(exist_ok=True)
            (templates_dir / "images").mkdir(exist_ok=True)
            super().__init__()
        
        self.test_results_db = test_results_db or TEST_CONFIG['test_results_db']
        self.cloud_downloader = CloudImageDownloader(CLOUD_CONFIG['download_dir'])
        self.init_test_database()
        self.current_session_id = None
        
        logger.info("自我测试器初始化完成")
    
    def init_test_database(self):
        """初始化测试结果数据库"""
        conn = sqlite3.connect(self.test_results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_images INTEGER DEFAULT 0,
                confirmed_matches INTEGER DEFAULT 0,
                correct_matches INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                completed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                image_path TEXT NOT NULL,
                image_name TEXT,
                expected_template_id TEXT,
                matched_template_id TEXT,
                similarity_score REAL,
                user_confirmed BOOLEAN DEFAULT FALSE,
                user_marked_correct BOOLEAN DEFAULT FALSE,
                processing_time REAL,
                test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (session_id) REFERENCES test_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"测试数据库初始化完成: {self.test_results_db}")
    
    def start_test_session(self, session_name: str, cloud_urls: List[str]) -> int:
        """开始新的测试会话"""
        conn = sqlite3.connect(self.test_results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_sessions (session_name, total_images)
            VALUES (?, ?)
        ''', (session_name, len(cloud_urls)))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.current_session_id = session_id
        logger.info(f"创建测试会话: {session_name} (ID: {session_id})")
        return session_id
    
    def process_test_image_with_confirmation(self, image_path: Path, 
                                           expected_template: Optional[str] = None) -> Dict:
        """处理测试图片并要求手动确认"""
        start_time = time.time()
        
        try:
            logger.info(f"开始处理测试图片: {image_path}")
            
            if not self.cloud_downloader.validate_image(image_path):
                logger.error(f"图片验证失败: {image_path}")
                return {'success': False, 'error': '图片验证失败'}
            
            # 加载模板库
            if not self.load_templates():
                logger.warning("⚠️ 未找到模板库，无法进行匹配测试")
                return {'success': False, 'error': '模板库加载失败'}
            
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return {'success': False, 'error': '图片读取失败'}
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            picked_colors = self._auto_pick_colors(hsv)
            if not picked_colors:
                logger.warning("自动颜色选择失败，使用默认颜色范围")
                picked_colors = [np.array([0, 0, 0])]  # 默认颜色
            
            # 创建掩码并提取轮廓
            mask = self._create_mask(hsv, picked_colors)
            valid_contours, all_contours = self._process_contours(mask, img.shape[:2])
            
            if not valid_contours:
                logger.warning("未检测到有效轮廓")
                return {'success': False, 'error': '未检测到有效轮廓'}
            
            query_features_list = [c['contours'] for c in valid_contours]
            matches = self.match_against_database(query_features_list)
            
            best_match = None
            best_similarity = 0.0
            if matches:
                for template_matches in matches.values():
                    if template_matches:
                        for match in template_matches:
                            if match['similarity'] > best_similarity:
                                best_similarity = match['similarity']
                                best_match = match
            
            processing_time = time.time() - start_time
            
            confirmation_result = self._show_test_confirmation_display(
                image_path, img, valid_contours, matches, expected_template, best_match
            )
            
            result = {
                'success': True,
                'image_path': str(image_path),
                'expected_template': expected_template,
                'matched_template': best_match['template_id'] if best_match else None,
                'similarity_score': best_similarity,
                'user_confirmed': confirmation_result.get('confirmed', False),
                'user_marked_correct': confirmation_result.get('correct', False),
                'processing_time': processing_time,
                'contours_count': len(valid_contours)
            }
            
            # 保存到数据库
            if self.current_session_id:
                self._save_test_result(result)
            
            logger.info(f"图片处理完成: {image_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"处理图片时发生错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _auto_pick_colors(self, hsv: np.ndarray) -> List[np.ndarray]:
        """自动选择颜色（简化版，避免用户交互）"""
        try:
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            h_peaks = []
            for i in range(10, 170):  # 排除接近黑白的区域
                if h_hist[i] > np.mean(h_hist) * 2:
                    h_peaks.append(i)
            
            if h_peaks:
                main_h = max(h_peaks, key=lambda x: h_hist[x])
                main_s = 128
                main_v = 128
                return [np.array([main_h, main_s, main_v])]
            else:
                return [np.array([0, 100, 100])]
                
        except Exception as e:
            logger.warning(f"自动颜色选择失败: {e}")
            return [np.array([0, 100, 100])]
    
    def _show_test_confirmation_display(self, image_path: Path, img: np.ndarray,
                                      valid_contours: List, matches: Dict, 
                                      expected_template: Optional[str] = None,
                                      best_match: Optional[Dict] = None) -> Dict:
        """显示测试确认界面"""
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=UI_CONFIG['figure_size'])
        ax_img, ax_contours, ax_matches = axes[0]
        ax_expected, ax_results, ax_controls = axes[1]
        
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"测试图片: {image_path.name}", fontsize=UI_CONFIG['font_size'])
        ax_img.axis('off')
        
        img_with_contours = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, contour_info in enumerate(valid_contours[:5]):  # 最多显示5个轮廓
            color = colors[i % len(colors)]
            cv2.drawContours(img_with_contours, [contour_info['contour']], -1, color, 2)
        
        ax_contours.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        ax_contours.set_title(f"检测到的轮廓 ({len(valid_contours)}个)", fontsize=UI_CONFIG['font_size'])
        ax_contours.axis('off')
        
        ax_matches.axis('off')
        if best_match:
            match_text = f"最佳匹配结果:\n\n"
            match_text += f"模板ID: {best_match['template_id']}\n"
            match_text += f"相似度: {best_match['similarity']:.3f}\n"
            match_text += f"匹配类型: {best_match.get('match_type', 'N/A')}\n"
            
            if best_match['similarity'] >= TEST_CONFIG['auto_confirm_threshold']:
                match_text += f"\n匹配质量: 优秀 ✅"
            elif best_match['similarity'] >= TEST_CONFIG['similarity_threshold']:
                match_text += f"\n匹配质量: 良好 ⚠️"
            else:
                match_text += f"\n匹配质量: 较差 ❌"
        else:
            match_text = "未找到匹配的模板\n\n请检查:\n- 模板库是否完整\n- 图片质量是否良好\n- 轮廓提取是否正确"
        
        ax_matches.text(0.05, 0.95, match_text, transform=ax_matches.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        ax_matches.set_title("匹配分析", fontsize=UI_CONFIG['font_size'])
        
        ax_expected.axis('off')
        if expected_template:
            expected_text = f"预期模板: {expected_template}\n\n"
            if best_match:
                is_correct = (best_match['template_id'] == expected_template)
                expected_text += f"实际匹配: {best_match['template_id']}\n"
                expected_text += f"匹配正确: {'✅ 是' if is_correct else '❌ 否'}\n"
                expected_text += f"相似度: {best_match['similarity']:.3f}"
            else:
                expected_text += "实际匹配: 无匹配结果\n匹配正确: ❌ 否"
        else:
            expected_text = "未提供预期模板\n\n请根据实际情况\n判断匹配结果是否正确"
        
        ax_expected.text(0.05, 0.95, expected_text, transform=ax_expected.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax_expected.set_title("预期对比", fontsize=UI_CONFIG['font_size'])
        
        ax_results.axis('off')
        stats_text = f"处理统计:\n\n"
        stats_text += f"轮廓数量: {len(valid_contours)}\n"
        stats_text += f"模板库大小: {len(self.templates) if self.templates else 0}\n"
        if matches:
            total_matches = sum(len(m) for m in matches.values())
            stats_text += f"候选匹配: {total_matches}\n"
        stats_text += f"图片尺寸: {img.shape[1]}x{img.shape[0]}"
        
        ax_results.text(0.05, 0.95, stats_text, transform=ax_results.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        ax_results.set_title("处理统计", fontsize=UI_CONFIG['font_size'])
        
        ax_controls.axis('off')
        control_text = "请确认匹配结果:\n\n"
        control_text += "点击下方按钮进行确认:\n"
        control_text += "• 确认正确: 匹配结果正确\n"
        control_text += "• 确认错误: 匹配结果错误\n"
        control_text += "• 跳过: 跳过此图片"
        
        ax_controls.text(0.05, 0.95, control_text, transform=ax_controls.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        ax_controls.set_title("操作指南", fontsize=UI_CONFIG['font_size'])
        
        confirmation_result = {'confirmed': False, 'correct': None}
        
        def on_confirm_correct(event):
            confirmation_result['confirmed'] = True
            confirmation_result['correct'] = True
            plt.close()
        
        def on_confirm_incorrect(event):
            confirmation_result['confirmed'] = True
            confirmation_result['correct'] = False
            plt.close()
        
        def on_skip(event):
            confirmation_result['confirmed'] = False
            confirmation_result['correct'] = None
            plt.close()
        
        from matplotlib.widgets import Button
        
        button_width, button_height = UI_CONFIG['button_size']
        ax_btn_correct = plt.axes([0.3, 0.02, button_width, button_height])
        ax_btn_incorrect = plt.axes([0.45, 0.02, button_width, button_height])
        ax_btn_skip = plt.axes([0.6, 0.02, button_width, button_height])
        
        btn_correct = Button(ax_btn_correct, '确认正确', color='lightgreen')
        btn_incorrect = Button(ax_btn_incorrect, '确认错误', color='lightcoral')
        btn_skip = Button(ax_btn_skip, '跳过', color='lightgray')
        
        btn_correct.on_clicked(on_confirm_correct)
        btn_incorrect.on_clicked(on_confirm_incorrect)
        btn_skip.on_clicked(on_skip)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 为按钮留出空间
        
        plt.show()
        
        return confirmation_result
    
    def _save_test_result(self, result: Dict):
        """保存测试结果到数据库"""
        if not self.current_session_id:
            logger.warning("没有活动的测试会话，跳过结果保存")
            return
        
        conn = sqlite3.connect(self.test_results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_results 
            (session_id, image_path, image_name, expected_template_id, 
             matched_template_id, similarity_score, user_confirmed, 
             user_marked_correct, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session_id,
            result['image_path'],
            Path(result['image_path']).name,
            result.get('expected_template'),
            result.get('matched_template'),
            result.get('similarity_score', 0.0),
            result.get('user_confirmed', False),
            result.get('user_marked_correct', False),
            result.get('processing_time', 0.0)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"测试结果已保存到数据库")
    
    def generate_test_report(self, session_id: int) -> Dict:
        """生成测试报告"""
        conn = sqlite3.connect(self.test_results_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM test_sessions WHERE id = ?', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            logger.error(f"未找到测试会话: {session_id}")
            return {}
        
        cursor.execute('''
            SELECT * FROM test_results 
            WHERE session_id = ? 
            ORDER BY test_timestamp
        ''', (session_id,))
        results = cursor.fetchall()
        
        total_images = len(results)
        confirmed_results = [r for r in results if r[7]]  # user_confirmed
        correct_results = [r for r in results if r[8]]    # user_marked_correct
        
        confirmed_count = len(confirmed_results)
        correct_count = len(correct_results)
        accuracy_rate = correct_count / confirmed_count if confirmed_count > 0 else 0.0
        
        cursor.execute('''
            UPDATE test_sessions 
            SET confirmed_matches = ?, correct_matches = ?, accuracy_rate = ?, completed = TRUE
            WHERE id = ?
        ''', (confirmed_count, correct_count, accuracy_rate, session_id))
        
        conn.commit()
        conn.close()
        
        report = {
            'session_id': session_id,
            'session_name': session[1],
            'created_at': session[2],
            'total_images': total_images,
            'confirmed_count': confirmed_count,
            'correct_count': correct_count,
            'accuracy_rate': accuracy_rate,
            'results': []
        }
        
        for result in results:
            report['results'].append({
                'image_name': result[3],
                'expected_template': result[4],
                'matched_template': result[5],
                'similarity_score': result[6],
                'user_confirmed': bool(result[7]),
                'user_marked_correct': bool(result[8]),
                'processing_time': result[9]
            })
        
        report_file = Path(f"test_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试报告已生成: {report_file}")
        
        print(f"\n{'='*50}")
        print(f"测试报告摘要")
        print(f"{'='*50}")
        print(f"会话名称: {report['session_name']}")
        print(f"测试时间: {report['created_at']}")
        print(f"总图片数: {total_images}")
        print(f"已确认数: {confirmed_count}")
        print(f"正确匹配: {correct_count}")
        print(f"准确率: {accuracy_rate:.2%}")
        print(f"报告文件: {report_file}")
        print(f"{'='*50}\n")
        
        return report
