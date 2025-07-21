#!/usr/bin/env python3
"""
专门测试match性能的脚本 - 使用同一张照片验证一一对应匹配
"""
import os
import cv2
import numpy as np
import json
from pathlib import Path
from BulidTheLab import ContourFeatureExtractor
from match import ToothMatcher, SimilarityCalculator

def extract_contours_and_features(image_path, image_type="real"):
    """提取轮廓和特征，不保存到文件"""
    print(f"🔍 处理图像: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None, None
    
    print(f"✅ 图像尺寸: {img.shape}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if image_type == "lab" or "lab" in image_path.lower():
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
    
    extractor = ContourFeatureExtractor()
    valid_contours = []
    
    for i, contour in enumerate(contours):
        if contour.shape[0] < 20:
            continue
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        points = contour.reshape(-1, 2)
        features = extractor.extract_all_features(contour, points, img.shape)
        
        valid_contours.append({
            'id': f"contour_{i}",
            'contour': contour,
            'points': points,
            'area': area,
            'features': features
        })
    
    print(f"✅ 筛选出 {len(valid_contours)} 个有效轮廓")
    return img, valid_contours

def test_self_matching(image_path):
    """测试同一张图片的自匹配性能"""
    print(f"\n🧪 测试自匹配性能: {image_path}")
    print("=" * 60)
    
    img, template_contours = extract_contours_and_features(image_path)
    if not template_contours:
        print("❌ 模板轮廓提取失败")
        return False
    
    img2, query_contours = extract_contours_and_features(image_path)
    if not query_contours:
        print("❌ 查询轮廓提取失败")
        return False
    
    calculator = SimilarityCalculator()
    
    print(f"\n🔄 开始匹配测试:")
    print(f"   模板轮廓数: {len(template_contours)}")
    print(f"   查询轮廓数: {len(query_contours)}")
    
    similarity_matrix = []
    match_results = []
    
    for i, query in enumerate(query_contours):
        query_similarities = []
        best_match = None
        best_similarity = 0
        
        for j, template in enumerate(template_contours):
            geo_sim = calculator.calculate_geometric_similarity(query['features'], template['features'])
            hu_sim = calculator.calculate_hu_similarity(query['features'], template['features'])
            fourier_sim = calculator.calculate_fourier_similarity(query['features'], template['features'])
            
            total_sim = (geo_sim * 0.4 + hu_sim * 0.3 + fourier_sim * 0.3)
            query_similarities.append(total_sim)
            
            if total_sim > best_similarity:
                best_similarity = total_sim
                best_match = j
        
        similarity_matrix.append(query_similarities)
        match_results.append({
            'query_id': i,
            'best_match': best_match,
            'similarity': best_similarity,
            'is_correct': i == best_match  # 理想情况下应该匹配自己
        })
    
    correct_matches = sum(1 for result in match_results if result['is_correct'])
    total_matches = len(match_results)
    accuracy = correct_matches / total_matches if total_matches > 0 else 0
    
    print(f"\n📊 匹配结果分析:")
    print(f"   总轮廓数: {total_matches}")
    print(f"   正确匹配: {correct_matches}")
    print(f"   匹配准确率: {accuracy:.2%}")
    
    print(f"\n📋 详细匹配结果:")
    for result in match_results[:10]:  # 只显示前10个
        status = "✅" if result['is_correct'] else "❌"
        print(f"   {status} 查询{result['query_id']} -> 模板{result['best_match']} (相似度: {result['similarity']:.3f})")
    
    if len(match_results) > 10:
        print(f"   ... 还有 {len(match_results) - 10} 个结果")
    
    similarities = [result['similarity'] for result in match_results]
    if similarities:
        print(f"\n📈 相似度统计:")
        print(f"   平均相似度: {np.mean(similarities):.3f}")
        print(f"   最高相似度: {np.max(similarities):.3f}")
        print(f"   最低相似度: {np.min(similarities):.3f}")
        print(f"   标准差: {np.std(similarities):.3f}")
    
    return accuracy >= 0.8  # 80%以上准确率认为通过

def test_cross_domain_matching():
    """测试跨域匹配：3D建模图 vs 现实照片"""
    print(f"\n🔬 测试跨域匹配性能")
    print("=" * 50)
    
    lab_image = "test_lab.png"
    tooth_image = "test_tooth.jpg"
    
    if not os.path.exists(lab_image) or not os.path.exists(tooth_image):
        print("❌ 测试图片不存在，跳过跨域测试")
        return False
    
    img1, template_contours = extract_contours_and_features(lab_image, "lab")
    if not template_contours:
        print("❌ 3D建模图轮廓提取失败")
        return False
    
    img2, query_contours = extract_contours_and_features(tooth_image, "real")
    if not query_contours:
        print("❌ 现实照片轮廓提取失败")
        return False
    
    calculator = SimilarityCalculator()
    
    print(f"\n🔄 开始跨域匹配:")
    print(f"   3D建模图轮廓数: {len(template_contours)}")
    print(f"   现实照片轮廓数: {len(query_contours)}")
    
    matches_found = 0
    high_confidence_matches = 0
    
    for i, query in enumerate(query_contours[:5]):  # 只测试前5个查询轮廓
        best_similarity = 0
        best_match = None
        
        for j, template in enumerate(template_contours):
            geo_sim = calculator.calculate_geometric_similarity(query['features'], template['features'])
            hu_sim = calculator.calculate_hu_similarity(query['features'], template['features'])
            fourier_sim = calculator.calculate_fourier_similarity(query['features'], template['features'])
            
            total_sim = (geo_sim * 0.5 + hu_sim * 0.3 + fourier_sim * 0.2)
            
            if total_sim > best_similarity:
                best_similarity = total_sim
                best_match = j
        
        if best_similarity > 0.1:  # 降低阈值
            matches_found += 1
            if best_similarity > 0.3:
                high_confidence_matches += 1
            
            confidence = "高" if best_similarity > 0.3 else "中等" if best_similarity > 0.2 else "低"
            print(f"   🔍 查询轮廓{i} -> 模板轮廓{best_match} (相似度: {best_similarity:.3f}, 置信度: {confidence})")
    
    print(f"\n📊 跨域匹配结果:")
    print(f"   找到匹配: {matches_found}/5")
    print(f"   高置信度匹配: {high_confidence_matches}/5")
    
    return matches_found >= 2  # 至少找到2个匹配认为通过

def main():
    """主测试函数"""
    print("🚀 TOOTHLAB 匹配性能测试")
    print("=" * 50)
    
    test_images = ["test_tooth.jpg", "test_lab.png"]
    self_match_results = []
    
    for image in test_images:
        if os.path.exists(image):
            result = test_self_matching(image)
            self_match_results.append((image, result))
        else:
            print(f"\n⚠️ 测试图片不存在: {image}")
            self_match_results.append((image, False))
    
    cross_domain_result = test_cross_domain_matching()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    for image, result in self_match_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   自匹配测试 ({image}): {status}")
    
    cross_status = "✅ 通过" if cross_domain_result else "❌ 失败"
    print(f"   跨域匹配测试: {cross_status}")
    
    all_passed = all(result for _, result in self_match_results) and cross_domain_result
    if all_passed:
        print("\n🎉 所有匹配性能测试通过！")
    else:
        print("\n⚠️ 部分测试失败，需要优化匹配算法")
    
    return all_passed

if __name__ == "__main__":
    main()
