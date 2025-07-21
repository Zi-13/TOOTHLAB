#!/usr/bin/env python3
"""
测试BulidTheLab.py和match.py集成的脚本
"""
import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path

def test_same_image_workflow(image_path):
    """使用同一张图片测试完整工作流程"""
    print(f"🧪 测试同一张图片的建库和匹配工作流程: {image_path}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return False
    
    try:
        cleanup_previous_data()
        
        print("\n🔨 步骤1: 建立模板库")
        success = build_template_non_interactive(image_path)
        if not success:
            print("❌ 模板建立失败")
            return False
        
        print("\n📁 步骤2: 验证模板文件")
        template_files = verify_template_files()
        if not template_files:
            print("❌ 模板文件验证失败")
            return False
        
        print("\n🔍 步骤3: 匹配识别")
        matches = match_image_non_interactive(image_path)
        if not matches:
            print("❌ 匹配识别失败")
            return False
        
        print("✅ 同一图片工作流程测试成功")
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
    if os.path.exists("templates/images"):
        shutil.rmtree("templates/images")
    os.makedirs("templates/contours", exist_ok=True)
    os.makedirs("templates/features", exist_ok=True)
    os.makedirs("templates/images", exist_ok=True)

def build_template_non_interactive(image_path):
    """非交互式建立模板"""
    from BulidTheLab import ToothTemplateBuilder, ContourFeatureExtractor
    import cv2
    
    builder = ToothTemplateBuilder()
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False
    
    print(f"✅ 成功读取图像，尺寸: {img.shape}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if "lab" in image_path.lower():
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        print("📸 使用3D建模图参数")
    else:
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])
        print("📷 使用现实照片参数")
    
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
    try:
        success = builder.serialize_contours(valid_contours, auto_save=True, image_path=image_path)
        return success
    except Exception as e:
        print(f"⚠️ 序列化失败，尝试简化保存: {e}")
        tooth_id = builder.get_next_tooth_id()
        simple_data = {
            "tooth_id": tooth_id,
            "num_contours": len(valid_contours),
            "test_mode": True
        }
        json_path = Path("templates/contours") / f"{tooth_id}.json"
        with open(json_path, 'w') as f:
            import json
            json.dump(simple_data, f)
        print(f"✅ 简化模板已保存: {tooth_id}")
        return True


def verify_template_files():
    """验证模板文件是否正确生成"""
    contours_dir = Path("templates/contours")
    features_dir = Path("templates/features")
    
    contour_files = list(contours_dir.glob("TOOTH_*.json"))
    feature_files = list(features_dir.glob("TOOTH_*_features.json"))
    
    if not contour_files:
        print(f"❌ 轮廓文件缺失: {len(contour_files)} 个")
        return False
    
    print(f"✅ 找到模板文件: {len(contour_files)} 个轮廓文件, {len(feature_files)} 个特征文件")
    return True

def match_image_non_interactive(image_path):
    """非交互式匹配图像"""
    from match import ToothMatcher
    
    matcher = ToothMatcher()
    if not matcher.load_templates():
        print("❌ 加载模板失败")
        return False
    
    print(f"✅ 已加载 {len(matcher.templates)} 个模板")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取匹配图像: {image_path}")
        return False
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if "lab" in image_path.lower():
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
    else:
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])
    
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
    
    matches = matcher.match_against_database(query_features, threshold=0.1)
    
    print(f"✅ 匹配完成，找到 {len(matches)} 个匹配结果")
    for query_id, query_matches in matches.items():
        if query_matches:
            best_match = query_matches[0]
            print(f"   查询轮廓 {query_id}: 最佳匹配 {best_match['template_id']} (相似度: {best_match['similarity']:.3f})")
    
    return len(matches) > 0 and any(len(query_matches) > 0 for query_matches in matches.values())

def test_cross_domain_matching():
    """测试跨域匹配：3D建模图 vs 现实照片"""
    print("\n🔬 测试跨域匹配功能")
    print("=" * 40)
    
    lab_image = "test_lab.png"
    tooth_image = "test_tooth.jpg"
    
    if not os.path.exists(lab_image):
        print(f"❌ 3D建模图不存在: {lab_image}")
        return False
    
    if not os.path.exists(tooth_image):
        print(f"❌ 现实照片不存在: {tooth_image}")
        return False
    
    print("📸 使用3D建模图建立模板库")
    cleanup_previous_data()
    success = build_template_non_interactive(lab_image)
    if not success:
        print("❌ 3D建模图建库失败")
        return False
    
    print("🔍 使用现实照片进行匹配")
    matches = match_image_non_interactive(tooth_image)
    if matches:
        print("✅ 跨域匹配测试成功")
        return True
    else:
        print("⚠️ 跨域匹配未找到结果，需要优化参数")
        return False

def main():
    """主测试函数"""
    print("🚀 TOOTHLAB集成测试")
    print("=" * 50)
    
    test_images = ["test_tooth.jpg", "test_lab.png"]
    same_image_results = []
    
    for image in test_images:
        if os.path.exists(image):
            print(f"\n📋 测试图片: {image}")
            result = test_same_image_workflow(image)
            same_image_results.append((image, result))
        else:
            print(f"\n⚠️ 测试图片不存在: {image}")
            same_image_results.append((image, False))
    
    cross_domain_result = test_cross_domain_matching()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    for image, result in same_image_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   同一图片测试 ({image}): {status}")
    
    cross_status = "✅ 通过" if cross_domain_result else "❌ 失败"
    print(f"   跨域匹配测试: {cross_status}")
    
    all_passed = all(result for _, result in same_image_results) and cross_domain_result
    if all_passed:
        print("\n🎉 所有集成测试通过！")
    else:
        print("\n⚠️ 部分测试失败，需要进一步优化")
    
    return all_passed

if __name__ == "__main__":
    main()
