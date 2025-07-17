import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

# 修改字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

INPUT_VIDEO = 'c:\\Users\\Jason\\Desktop\\0717_1.DNG'


def fit_fourier_series(data, t, order):
    A = np.ones((len(t), 2 * order + 1))
    for k in range(1, order + 1):
        A[:, 2 * k - 1] = np.cos(k * t)
        A[:, 2 * k] = np.sin(k * t)
    coeffs, _, _, _ = lstsq(A, data, rcond=None)
    return coeffs

def evaluate_fourier_series(coeffs, t, order):
    A = np.ones((len(t), 2 * order + 1))
    for k in range(1, order + 1):
        A[:, 2 * k - 1] = np.cos(k * t)
        A[:, 2 * k] = np.sin(k * t)
    return A @ coeffs

def fourier_fit_and_plot(points, order, label='傅里叶拟合轮廓', linewidth=1.2, ax=None, center_normalize=True):
    try:
        x = points[:, 0].astype(float)
        y = points[:, 1].astype(float)
        
        # 计算几何中心
        center_x = np.mean(x)
        center_y = np.mean(y)
        
        if center_normalize:
            # 以几何中心为原点进行归一化
            x_normalized = x - center_x
            y_normalized = y - center_y
            
            # 计算缩放因子（使用最大距离进行归一化）
            max_dist = np.max(np.sqrt(x_normalized**2 + y_normalized**2))
            if max_dist > 0:
                x_normalized /= max_dist
                y_normalized /= max_dist
        else:
            x_normalized = x
            y_normalized = y
        
        N = len(points)
        t = np.linspace(0, 2 * np.pi, N)
        
        # 对归一化后的坐标进行傅里叶拟合
        coeffs_x = fit_fourier_series(x_normalized, t, order)
        coeffs_y = fit_fourier_series(y_normalized, t, order)
        
        # 生成更密集的参数点用于平滑显示
        t_dense = np.linspace(0, 2 * np.pi, N * 4)
        x_fit_normalized = evaluate_fourier_series(coeffs_x, t_dense, order)
        y_fit_normalized = evaluate_fourier_series(coeffs_y, t_dense, order)
        
        if center_normalize:
            # 将拟合结果还原到原始坐标系
            x_fit = x_fit_normalized * max_dist + center_x
            y_fit = y_fit_normalized * max_dist + center_y
        else:
            x_fit = x_fit_normalized
            y_fit = y_fit_normalized
        
        if ax is not None:
            ax.plot(x, y, 'k.', markersize=1, alpha=0.3, label='原始轮廓点')
            ax.plot(x_fit, y_fit, '-', linewidth=linewidth, label=label, color='blue')
            
        # 返回拟合结果和系数
        fourier_data = {
            'coeffs_x': coeffs_x,
            'coeffs_y': coeffs_y,
            'center_x': center_x,
            'center_y': center_y,
            'max_dist': max_dist if center_normalize else 1.0,
            'order': order
        }
        return x_fit, y_fit, fourier_data
            
    except Exception as e:
        print(f"[傅里叶拟合失败]：{e}")
        return None, None, None

def extract_contour_features(contour, points):
    """提取轮廓的多种特征"""
    features = {}
    
    # 1. 几何特征
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # 边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    
    # 圆形度
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
    
    # 凸包
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0
    
    features['area'] = area
    features['perimeter'] = perimeter
    features['aspect_ratio'] = aspect_ratio
    features['circularity'] = circularity
    features['solidity'] = solidity
    
    # 2. Hu矩特征
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # 对数变换使其更稳定
    for i in range(len(hu_moments)):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
        else:
            hu_moments[i] = 0
    features['hu_moments'] = hu_moments
    
    # 3. 傅里叶描述符特征
    try:
        x_fit, y_fit, fourier_data = fourier_fit_and_plot(points, order=80, ax=None, center_normalize=True)
        if fourier_data is not None:
            # 提取低频系数作为形状特征
            coeffs_x = fourier_data['coeffs_x']
            coeffs_y = fourier_data['coeffs_y']
            # 组合前5阶系数
            fourier_features = np.concatenate([coeffs_x[:11], coeffs_y[:11]])  # 0阶+10阶*2
            features['fourier_descriptors'] = fourier_features
        else:
            features['fourier_descriptors'] = np.zeros(22)
    except:
        features['fourier_descriptors'] = np.zeros(22)
    
    # 4. 轮廓角点特征
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features['corner_count'] = len(approx)
    
    return features

def compare_contours(features1, features2, size_tolerance=0.9):
    """比较两个轮廓的相似度，先进行尺寸筛选"""
    similarities = {}
    
    # 0. 一级筛选：周长和面积相似度
    area1, area2 = features1['area'], features2['area']
    perimeter1, perimeter2 = features1['perimeter'], features2['perimeter']
    
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
    
    # 尺寸综合相似度
    size_similarity = (area_sim + perimeter_sim) / 2
    similarities['size'] = size_similarity
    
    # 一级筛选：如果尺寸差异过大，直接返回低相似度
    if size_similarity < size_tolerance:
        similarities['geometric'] = 0.0
        similarities['hu_moments'] = 0.0
        similarities['fourier'] = 0.0
        similarities['overall'] = size_similarity  # 直接使用尺寸相似度，不重复加权
        return similarities
    
    # TODO 1. 几何特征相似度（通过一级筛选后才计算）
    geometric_features = ['circularity', 'aspect_ratio', 'solidity']
    geometric_weights = [0.2, 0.1, 0.7]
    '''TODO: 
       circularity圆角度
       aspect_ratio 长宽比 
       solidity 凸包占比'''
    
    geometric_sim = []
    for feat in geometric_features:
        v1, v2 = features1[feat], features2[feat]
        if v1 == 0 and v2 == 0:
            sim = 1.0
        elif v1 == 0 or v2 == 0:
            sim = 0.0
        else:
            diff = abs(v1 - v2) / max(v1, v2)
            sim = max(0, 1 - diff * 1.5)
        geometric_sim.append(sim)
    
    weighted_geometric_sim = sum(w * s for w, s in zip(geometric_weights, geometric_sim))
    similarities['geometric'] = weighted_geometric_sim
    
    # 2. Hu矩相似度
    hu1 = features1['hu_moments']
    hu2 = features2['hu_moments']
    hu_sim = cosine_similarity([hu1], [hu2])[0][0]
    similarities['hu_moments'] = max(0, hu_sim)
    
    # 3. 傅里叶描述符相似度
    fourier1 = features1['fourier_descriptors']
    fourier2 = features2['fourier_descriptors']
    fourier_sim = cosine_similarity([fourier1], [fourier2])[0][0]
    similarities['fourier'] = max(0, fourier_sim)
    
  #TODO: 4. 轮廓角点相似度（可选）调节最终权重
    weights = {
        'geometric': 0.55,    # 几何特征权重
        'hu_moments': 0.05,   # Hu矩权重  
        'fourier': 0.4       # 傅里叶描述符权重
    }
    
    # 只对通过尺寸筛选的轮廓计算形状相似度
    shape_similarity = sum(weights[k] * similarities[k] for k in weights)
    
    # TODO 最终相似度 = 尺寸相似度 × 形状相似度
    w1, w2 = 0.1, 0.9  # 尺寸和形状的权重
    similarities['overall'] = size_similarity *w1+ shape_similarity*w2
    
    return similarities

def find_similar_contours(target_features, all_features, threshold=0.5, size_tolerance=0.3):
    """找到与目标轮廓相似的所有轮廓"""
    similar_contours = []
    
    for i, features in enumerate(all_features):
        if features == target_features:  # 跳过自己
            continue
        
        similarities = compare_contours(target_features, features, size_tolerance)
        if similarities['overall'] >= threshold:  # 使用1.0作为临界值
            similar_contours.append({
                'index': i,
                'similarity': similarities['overall'],
                'details': similarities
            })
    
    # 按相似度排序
    similar_contours.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_contours

def pick_color_and_draw_edge(image_path):
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
    cv2.imshow("点击选取目标区域颜色", img)
    cv2.setMouseCallback("点击选取目标区域颜色", on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not picked:
        print("未选取颜色")
        return
    hsv_arr = np.array(picked)
    h, s, v = np.mean(hsv_arr, axis=0).astype(int)
    print(f"HSV picked: {h}, {s}, {v}")
    #TODO: 这里可以根据选取的颜色调整阈值                                                                            
    lower = np.array([max(h-15,0), max(s-60,0), max(v-60,0)])
    upper = np.array([min(h+15,179), min(s+60,255), min(v+60,255)])
    print(f"lower: {lower}, upper: {upper}")
    mask = cv2.inRange(hsv, lower, upper)
    color_extract = cv2.bitwise_and(img, img, mask=mask)
    
    # --- 记录所有有效轮廓及属性 ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_contours = []
    all_features = []
    
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        points = contour[:, 0, :]
        
        # 提取特征
        features = extract_contour_features(contour, points)
        
        valid_contours.append({
            'contour': contour,
            'points': points,
            'area': area,
            'length': length,
            'idx': i,
            'features': features
        })
        all_features.append(features)
    
    n_contours = len(valid_contours)
    linewidth = max(0.5, 2 - 0.03 * n_contours)
    show_legend = n_contours <= 15
    
    # --- 交互式显示 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_img, ax_fit, ax_zoom = axes
    ax_img.set_title("颜色提取结果")
    ax_img.imshow(cv2.cvtColor(color_extract, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    ax_fit.set_title("轮廓显示")
    ax_fit.axis('equal')
    ax_fit.invert_yaxis()
    ax_fit.grid(True)
    ax_zoom.set_title("色块放大视图")
    ax_zoom.axis('equal')
    ax_zoom.invert_yaxis()
    ax_zoom.grid(True)
    selected_idx = [0]
    
    def draw_all(highlight_idx=None):
        ax_fit.clear()
        ax_fit.set_title("轮廓显示")
        ax_fit.axis('equal')
        ax_fit.invert_yaxis()
        ax_fit.grid(True)
        
        # 准备颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(valid_contours), 10)))
        
        # 找到相似轮廓 - 使用1.0作为临界值
        similar_contours = []
        if highlight_idx is not None:
            target_features = valid_contours[highlight_idx]['features']
            similar_contours = find_similar_contours(target_features, all_features, 
                                                   threshold=0.5, size_tolerance=0.3)  # ← 改为1.0
        
        for j, info in enumerate(valid_contours):
            points = info['points']
            label = f"色块{info['idx']+1}"
            
            # 检查是否为相似轮廓
            is_similar = any(sim['index'] == j for sim in similar_contours)
            
            if highlight_idx is not None and j == highlight_idx:
                fill_color = 'red'
                edge_color = 'darkred'
                lw = linewidth * 2
                alpha = 0.7
                zorder = 10
                text_color = 'white'  # 红色背景用白色文字
            elif is_similar:
                fill_color = 'orange'  # 相似轮廓用橙色
                edge_color = 'darkorange'
                lw = linewidth * 1.5
                alpha = 0.6
                zorder = 5
                text_color = 'black'  # 橙色背景用黑色文字
                # 添加相似度信息到标签
                sim_info = next(sim for sim in similar_contours if sim['index'] == j)
                label += f" (相似度:{sim_info['similarity']:.2f})"
            else:
                fill_color = colors[j % len(colors)]
                edge_color = 'black'
                lw = linewidth
                alpha = 0.5
                zorder = 1
                text_color = 'black'  # 其他颜色用黑色文字
            
            x = points[:, 0]
            y = points[:, 1]
            
            # 填充色块区域
            ax_fit.fill(x, y, color=fill_color, alpha=alpha, zorder=zorder, label=label if show_legend else None)
            # 绘制轮廓边界
            ax_fit.plot(x, y, '-', color=edge_color, linewidth=lw, zorder=zorder+1)
            
            # 在色块中心标注编号
            center_x = np.mean(x)
            center_y = np.mean(y)
            
            # 根据色块大小调整字体大小
            contour_area = info['area']
            if contour_area > 10000:
                fontsize = 14
            elif contour_area > 5000:
                fontsize = 12
            elif contour_area > 1000:
                fontsize = 10
            else:
                fontsize = 8
            
            # 标注色块编号
            ax_fit.text(center_x, center_y, str(info['idx']+1), 
                       fontsize=fontsize, fontweight='bold', 
                       color=text_color, ha='center', va='center',
                       zorder=zorder+2)
            
            # 如果是相似轮廓，额外标注相似度
            if is_similar:
                sim_info = next(sim for sim in similar_contours if sim['index'] == j)
                ax_fit.text(center_x, center_y - 15, f"{sim_info['similarity']:.2f}", 
                           fontsize=max(6, fontsize-2), fontweight='normal', 
                           color=text_color, ha='center', va='center',
                           zorder=zorder+2)
            
        if show_legend:
            ax_fit.legend()
        
        # --- 显示特征信息 ---
        info = valid_contours[highlight_idx if highlight_idx is not None else 0]
        features = info['features']
        
        # 构建特征信息字符串
        feature_info = f"色块编号: {info['idx']+1}\n"
        feature_info += f"面积: {features['area']:.2f}\n"
        feature_info += f"周长: {features['perimeter']:.2f}\n"
        feature_info += f"长宽比: {features['aspect_ratio']:.3f}\n"
        feature_info += f"圆形度: {features['circularity']:.3f}\n"
        feature_info += f"凸度: {features['solidity']:.3f}\n"
        feature_info += f"角点数: {features['corner_count']}\n"
        
        # 显示相似轮廓信息
        if highlight_idx is not None:
            target_features = valid_contours[highlight_idx]['features']
            similar_list = find_similar_contours(target_features, all_features, 
                                               threshold=0.5, size_tolerance=0.3)  # ← 改为1.0
            if similar_list:
                feature_info += f"\n相似轮廓 (相似度>1.0):\n"  # ← 更新提示文字
                for sim in similar_list[:3]:  # 只显示前3个
                    details = sim['details']
                    feature_info += f"  色块{valid_contours[sim['index']]['idx']+1}: {sim['similarity']:.3f}\n"
                    feature_info += f"    (尺寸:{details['size']:.2f} 几何:{details['geometric']:.2f} Hu:{details['hu_moments']:.2f})\n"
            else:
                feature_info += f"\n未找到相似轮廓 (相似度>1.0)"  # ← 更新提示文字
        
        # 显示特征信息
        ax_fit.text(0.02, -0.25, feature_info,
                    transform=ax_fit.transAxes, fontsize=8, color='red', 
                    va='top', ha='left', fontfamily='sans-serif')

        # --- 放大视图 ---
        ax_zoom.clear()
        ax_zoom.set_title("色块放大视图")
        ax_zoom.axis('equal')
        ax_zoom.invert_yaxis()
        ax_zoom.grid(True)
        points = info['points']
        x = points[:, 0]
        y = points[:, 1]
        
        # 在放大视图中也显示填充
        if highlight_idx is not None:
            fill_color = 'red'
            alpha = 0.7
        else:
            fill_color = colors[0 % len(colors)]
            alpha = 0.5
            
        # 填充区域
        ax_zoom.fill(x, y, color=fill_color, alpha=alpha, label='填充区域')
        # 绘制轮廓点和连线
        ax_zoom.plot(x, y, 'k.', markersize=2, alpha=0.8, label='轮廓点')
        ax_zoom.plot(x, y, 'black', linewidth=2, label='轮廓线')
        
        # 在放大视图中也标注编号
        center_x = np.mean(x)
        center_y = np.mean(y)
        ax_zoom.text(center_x, center_y, str(info['idx']+1), 
                    fontsize=16, fontweight='bold', 
                    color='white' if highlight_idx is not None else 'black', 
                    ha='center', va='center', zorder=10)
        
        # 添加傅里叶拟合（归一化）
        x_fit, y_fit, _ = fourier_fit_and_plot(points, order=80, 
                                           label='傅里叶拟合(归一化)', 
                                           linewidth=2, ax=ax_zoom, 
                                           center_normalize=True)
        
        ax_zoom.legend()
        # 自适应缩放
        margin = 20
        ax_zoom.set_xlim(x.min()-margin, x.max()+margin)
        ax_zoom.set_ylim(y.min()-margin, y.max()+margin)
        fig.canvas.draw_idle()

    def show_info(idx):
        # 已集成到draw_all中，无需单独显示
        pass
    def on_click(event):
        if event.inaxes not in [ax_img, ax_fit, ax_zoom]:
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
    draw_all(highlight_idx=selected_idx[0])
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

def main():
    #TODO: 替换为实际图片路径
    pick_color_and_draw_edge(INPUT_VIDEO) 
if __name__ == "__main__":
    main()
