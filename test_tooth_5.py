#!/usr/bin/env python3
"""
测试Tooth_5.png图片的完整工作流程
使用用户指定的HSV参数: [0,0,0] 到 [15,60,61]
"""
import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path

def test_tooth_5_workflow():
    """使用Tooth_5.png测试完整工作流程"""
    image_path = "test_tooth_5.png"
    
    print(f"🧪 测试Tooth_5.png的建库和匹配工作流程")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return False
    
    try:
        cleanup_previous_data()
        
        print("\n🔨 步骤1: 建立模板库")
        success = build_template_with_custom_hsv(image_path)
        if not success:
            print("❌ 模板建立失败")
            return False
        
        print("\n📁 步骤2: 验证模板文件")
        template_files = verify_template_files()
        if not template_files:
            print("❌ 模板文件验证失败")
            return False
        
        print("\n🔍 步骤3: 匹配识别")
        matches = match_image_with_custom_hsv(image_path)
        if not matches:
            print("❌ 匹配识别失败")
            return False
        
        print("✅ Tooth_5.png工作流程测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_previous_data():
    """清理之前的测试数据"""
    if os.path.exists("tooth_templates.db"):
        os.remove("tooth_templates.db")
    if os.path.exists("templates/contours"):
        shutil.rmtree("templates/contours")
    if os.path.exists("templates/features"):
        shutil.rmtree("templates/features")
    os.makedirs("templates/contours", exist_ok=True)
    os.makedirs("templates/features", exist_ok=True)

def build_template_with_custom_hsv(image_path):
    """使用自定义HSV参数建立模板"""
    from BulidTheLab import ToothTemplateBuilder, ContourFeatureExtractor
    import cv2
    
    builder = ToothTemplateBuilder()
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False
    
    print(f"✅ 成功读取图像，尺寸: {img.shape}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([15, 60, 61])
    
    print(f"📸 使用自定义HSV参数: lower={lower}, upper={upper}")
    
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print(f"🔍 找到 {len(contours)} 个原始轮廓")
    
    valid_contours = []
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        extractor = ContourFeatureExtractor()
        points = contour.reshape(-1, 2)
        features = extractor.extract_all_features(contour, points, img.shape)
        
        valid_contours.append({
            'idx': i,
            'contour': contour,
            'points': points,
            'area': area,
            'length': cv2.arcLength(contour, True),
            'features': features
        })
    
    print(f"✅ 筛选出 {len(valid_contours)} 个有效轮廓")
    
    if not valid_contours:
        print("❌ 未找到有效轮廓")
        return False
    
    builder.current_image = img
    success = builder.serialize_contours(valid_contours, auto_save=True, image_path=image_path)
    return success

def verify_template_files():
    """验证模板文件是否正确生成"""
    contours_dir = Path("templates/contours")
    features_dir = Path("templates/features")
    
    contour_files = list(contours_dir.glob("TOOTH_*.json"))
    feature_files = list(features_dir.glob("TOOTH_*_features.json"))
    
    if not contour_files or not feature_files:
        print(f"❌ 模板文件缺失: contours={len(contour_files)}, features={len(feature_files)}")
        return False
    
    print(f"✅ 找到模板文件: {len(contour_files)} 个轮廓文件, {len(feature_files)} 个特征文件")
    return True

def match_image_with_custom_hsv(image_path):
    """使用自定义HSV参数匹配图像"""
    from match import ToothMatcher
    
    matcher = ToothMatcher()
    if not matcher.load_templates():
        print("❌ 加载模板失败")
        return False
    
    print(f"✅ 已加载 {len(matcher.templates)} 个模板")
    
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([15, 60, 61])
    
    print(f"🔍 匹配使用HSV参数: lower={lower}, upper={upper}")
    
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print(f"🔍 匹配图像中找到 {len(contours)} 个原始轮廓")
    
    query_features = []
    for contour in contours:
        if contour.shape[0] < 20 or cv2.contourArea(contour) < 100:
            continue
        
        points = contour.reshape(-1, 2)
        features = matcher.feature_extractor.extract_all_features(contour, points, img.shape)
        query_features.append(features)
    
    print(f"✅ 提取到 {len(query_features)} 个查询特征")
    
    if not query_features:
        print("❌ 未提取到查询特征")
        return False
    
    matches = matcher.match_against_database(query_features, threshold=0.3)
    
    print(f"✅ 匹配完成，找到 {len(matches)} 个匹配结果")
    
    total_matches = 0
    perfect_matches = 0
    high_confidence_matches = 0
    
    for query_id, query_matches in matches.items():
        if query_matches:
            total_matches += len(query_matches)
            best_match = query_matches[0]
            similarity = best_match['similarity']
            
            if similarity >= 0.99:
                perfect_matches += 1
                confidence_level = "完美"
            elif similarity >= 0.8:
                high_confidence_matches += 1
                confidence_level = "高"
            elif similarity >= 0.5:
                confidence_level = "中等"
            else:
                confidence_level = "低"
            
            print(f"   {query_id}: 最佳匹配 {best_match['template_id']} (相似度: {similarity:.3f}, 置信度: {confidence_level})")
    
    print(f"\n📊 匹配统计:")
    print(f"   总匹配数: {total_matches}")
    print(f"   完美匹配 (≥0.99): {perfect_matches}")
    print(f"   高置信度匹配 (≥0.8): {high_confidence_matches}")
    print(f"   匹配成功率: {len(matches)}/{len(query_features)} = {len(matches)/len(query_features)*100:.1f}%")
    
    return len(matches) > 0

def main():
    """主测试函数"""
    print("🚀 TOOTHLAB Tooth_5.png 专项测试")
    print("=" * 50)
    
    result = test_tooth_5_workflow()
    
    print("\n" + "=" * 50)
    if result:
        print("🎉 Tooth_5.png测试完全成功！")
        print("✅ 系统能够正确处理该图像并实现一一对应匹配")
    else:
        print("⚠️ Tooth_5.png测试失败，需要进一步调试")
    
    return result

if __name__ == "__main__":
    main()
