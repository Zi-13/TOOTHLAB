#!/usr/bin/env python3
"""
测试四阶段升级功能的脚本
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_batch_processor():
    """测试批量处理器"""
    print("🧪 测试批量处理器...")
    try:
        from batch_processor import BatchTemplateProcessor
        processor = BatchTemplateProcessor()
        print("✅ 批量处理器导入成功")
        return True
    except Exception as e:
        print(f"❌ 批量处理器测试失败: {e}")
        return False

def test_web_api():
    """测试Web API"""
    print("🧪 测试Web API...")
    try:
        from web_api import app
        print("✅ Web API导入成功")
        return True
    except Exception as e:
        print(f"❌ Web API测试失败: {e}")
        return False

def test_enhanced_matching():
    """测试增强的匹配功能"""
    print("🧪 测试增强匹配功能...")
    try:
        from match import SimilarityCalculator
        calc = SimilarityCalculator()
        
        if hasattr(calc, 'coarse_match') and hasattr(calc, 'fine_match'):
            print("✅ 分层匹配功能可用")
            return True
        else:
            print("❌ 分层匹配功能缺失")
            return False
    except Exception as e:
        print(f"❌ 匹配功能测试失败: {e}")
        return False

def test_template_management():
    """测试模板管理功能"""
    print("🧪 测试模板管理功能...")
    try:
        from BulidTheLab import ToothTemplateBuilder
        builder = ToothTemplateBuilder()
        
        if hasattr(builder, 'delete_template'):
            print("✅ 模板删除功能可用")
            return True
        else:
            print("❌ 模板删除功能缺失")
            return False
    except Exception as e:
        print(f"❌ 模板管理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始TOOTHLAB四阶段升级测试")
    print("=" * 50)
    
    tests = [
        ("阶段1: 批量处理", test_batch_processor),
        ("阶段2: 增强匹配", test_enhanced_matching),
        ("阶段3: 模板管理", test_template_management),
        ("阶段4: Web API", test_web_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！四阶段升级成功完成。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
