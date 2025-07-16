import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path

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

    def serialize_contours(self, valid_contours, tooth_id, image_path, hsv_info=None):
        try:
            template_data = {
                "tooth_id": tooth_id,
                "image_path": image_path,
                "created_at": datetime.now().isoformat(),
                "hsv_info": hsv_info,
                "num_contours": len(valid_contours),
                "contours": []
            }
            
            total_area = 0
            for i, contour_info in enumerate(valid_contours):
                points = contour_info['points'].tolist()
                x, y, w, h = cv2.boundingRect(contour_info['contour'])
                
                contour_data = {
                    "idx": i,
                    "original_idx": contour_info['idx'],
                    "points": points,
                    "area": float(contour_info['area']),
                    "perimeter": float(contour_info['length']),
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                }
                template_data["contours"].append(contour_data)
                total_area += contour_info['area']
            
            template_data["total_area"] = float(total_area)
            
            # 保存JSON文件
            json_filename = f"{tooth_id}.json"
            json_path = self.templates_dir / "contours" / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            # 保存到数据库
            self.save_to_database(template_data, json_filename, image_path)
            
            print(f"✅ 模板已保存: {tooth_id} ({len(valid_contours)}个轮廓)")
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
    
    lower = np.array([max(h-15,0), max(s-60,0), max(v-60,0)])
    upper = np.array([min(h+15,179), min(s+60,255), min(v+60,255)])
    print(f"lower: {lower}, upper: {upper}")
    
    # 保存HSV信息
    hsv_info = {
        'h_mean': int(h), 's_mean': int(s), 'v_mean': int(v),
        'lower': lower.tolist(), 'upper': upper.tolist()
    }
    
    mask = cv2.inRange(hsv, lower, upper)
    color_extract = cv2.bitwise_and(img, img, mask=mask)
    
    # --- 记录所有有效轮廓及属性 ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    
    # 自动生成牙齿ID
    if tooth_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tooth_id = f"TOOTH_{timestamp}"
    
    # --- 交互式显示 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_img, ax_contour, ax_zoom = axes
    
    ax_img.set_title("颜色提取结果")
    ax_img.imshow(cv2.cvtColor(color_extract, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    
    ax_contour.set_title("轮廓显示")
    ax_contour.axis('equal')
    ax_contour.invert_yaxis()
    ax_contour.grid(True)
    
    ax_zoom.set_title("色块放大视图")
    ax_zoom.axis('equal')
    ax_zoom.grid(True)
    
    selected_idx = [0]  # 用列表包裹以便闭包修改
    saved = [False]  # 保存状态
    
    def draw_all(highlight_idx=None):
        ax_contour.clear()
        ax_contour.set_title(f"轮廓显示 - 牙齿ID: {tooth_id}")
        ax_contour.axis('equal')
        ax_contour.invert_yaxis()
        ax_contour.grid(True)
        
        # 准备颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(valid_contours), 10)))
        
        for j, info in enumerate(valid_contours):
            points = info['points']
            label = f"区域{info['idx']+1}"
            
            if highlight_idx is not None and j == highlight_idx:
                fill_color = 'red'
                edge_color = 'darkred'
                lw = linewidth * 2
                alpha = 0.7
                zorder = 10
            else:
                fill_color = colors[j % len(colors)]
                edge_color = 'black'
                lw = linewidth
                alpha = 0.5
                zorder = 1
            
            x = points[:, 0]
            y = points[:, 1]
            
            # 填充色块区域
            ax_contour.fill(x, y, color=fill_color, alpha=alpha, zorder=zorder, label=label if show_legend else None)
            # 绘制轮廓边界
            ax_contour.plot(x, y, '-', color=edge_color, linewidth=lw, zorder=zorder+1)
            
        if show_legend:
            ax_contour.legend()
        
        # --- 属性信息显示 ---
        info = valid_contours[highlight_idx if highlight_idx is not None else 0]
        status = "✅ 已保存" if saved[0] else "❌ 未保存"
        ax_contour.text(0.02, -0.08, f"区域: {info['idx']+1} | 面积: {info['area']:.1f} | 周长: {info['length']:.1f}",
                        transform=ax_contour.transAxes, fontsize=10, color='blue')
        ax_contour.text(0.02, -0.12, f"状态: {status} | 按 's' 保存模板 | 按 'q' 退出",
                        transform=ax_contour.transAxes, fontsize=10, color='red')
        
        # --- 放大视图 ---
        ax_zoom.clear()
        ax_zoom.set_title("区域放大视图")
        ax_zoom.axis('equal')
        ax_zoom.invert_yaxis()
        ax_zoom.grid(True)
        
        points = info['points']
        x = points[:, 0]
        y = points[:, 1]
        
        # 在放大视图中显示填充
        if highlight_idx is not None:
            fill_color = 'red'
            alpha = 0.7
        else:
            fill_color = colors[0 % len(colors)]
            alpha = 0.5
            
        # 填充区域
        ax_zoom.fill(x, y, color=fill_color, alpha=alpha)
        # 绘制轮廓点和连线
        ax_zoom.plot(x, y, 'k.', markersize=1.5)
        ax_zoom.plot(x, y, 'black', linewidth=1.5)
        
        # 自适应缩放
        margin = 20
        ax_zoom.set_xlim(x.min()-margin, x.max()+margin)
        ax_zoom.set_ylim(y.min()-margin, y.max()+margin)
        
        fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes not in [ax_img, ax_contour, ax_zoom]:
            return
        x, y = int(event.xdata), int(event.ydata)
        found = False
        for j, info in enumerate(valid_contours):
            if cv2.pointPolygonTest(info['contour'], (x, y), False) >= 0:
                selected_idx[0] = j
                draw_all(highlight_idx=j)
                found = True
                break
        if not found:
            print("未选中任何色块")
    
    def on_key(event):
        if event.key == 'right':
            selected_idx[0] = (selected_idx[0] + 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'left':
            selected_idx[0] = (selected_idx[0] - 1) % n_contours
            draw_all(highlight_idx=selected_idx[0])
        elif event.key == 's':
            # 保存模板
            success = builder.serialize_contours(valid_contours, tooth_id, image_path, hsv_info)
            if success:
                saved[0] = True
                draw_all(highlight_idx=selected_idx[0])
        elif event.key == 'q':
            plt.close()
    
    draw_all(highlight_idx=selected_idx[0])
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
    
    # 显示已保存的模板列表
    builder.list_templates()

def main():
    # 可以指定牙齿ID
    tooth_id = input("🦷 请输入牙齿ID (直接回车自动生成): ").strip()
    if not tooth_id:
        tooth_id = None
    
    pick_color_and_draw_edge('c:\\Users\\Jason\\Desktop\\ya.jpg', tooth_id)

if __name__ == "__main__":
    main()