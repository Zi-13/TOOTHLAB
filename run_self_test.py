#!/usr/bin/env python3
"""
云盘图片自我测试主程序
使用方法：python run_self_test.py --urls url1,url2,url3 --session-name "测试会话1"
"""

import argparse
import sys
from pathlib import Path
from typing import List
import logging

from self_tester import SelfTester
from config import LOGGING_CONFIG

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGGING_CONFIG['file'], encoding='utf-8')
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='云盘图片自我测试')
    parser.add_argument('--urls', required=True, help='云盘图片URL列表，用逗号分隔')
    parser.add_argument('--session-name', required=True, help='测试会话名称')
    parser.add_argument('--expected-templates', help='预期模板ID列表，用逗号分隔（可选）')
    parser.add_argument('--clear-cache', action='store_true', help='清空下载缓存')
    
    args = parser.parse_args()
    
    urls = [url.strip() for url in args.urls.split(',') if url.strip()]
    expected_templates = []
    if args.expected_templates:
        expected_templates = [t.strip() for t in args.expected_templates.split(',') if t.strip()]
    
    if not urls:
        logger.error("未提供有效的URL")
        sys.exit(1)
    
    try:
        tester = SelfTester()
    except Exception as e:
        logger.error(f"初始化测试器失败: {e}")
        sys.exit(1)
    
    if args.clear_cache:
        tester.cloud_downloader.clear_download_dir()
        logger.info("已清空下载缓存")
    
    print(f"🚀 开始自我测试会话: {args.session_name}")
    print(f"📥 待测试图片数量: {len(urls)}")
    if expected_templates:
        print(f"📋 预期模板数量: {len(expected_templates)}")
    
    try:
        session_id = tester.start_test_session(args.session_name, urls)
        
        successful_tests = 0
        failed_tests = 0
        
        for i, url in enumerate(urls):
            expected = expected_templates[i] if i < len(expected_templates) else None
            print(f"\n📸 处理图片 {i+1}/{len(urls)}: {url}")
            
            try:
                print("  ⬇️ 正在下载...")
                image_path = tester.cloud_downloader.download_from_url(url, f"test_image_{i+1:03d}.jpg")
                if not image_path:
                    logger.error(f"下载失败: {url}")
                    failed_tests += 1
                    continue
                
                print(f"  ✅ 下载完成: {image_path.name}")
                
                print("  🔍 正在处理和匹配...")
                result = tester.process_test_image_with_confirmation(image_path, expected)
                
                if result['success']:
                    print(f"  ✅ 图片 {i+1} 处理完成")
                    if result.get('user_confirmed'):
                        print(f"  👤 用户确认: {'正确' if result.get('user_marked_correct') else '错误'}")
                    successful_tests += 1
                else:
                    print(f"  ❌ 图片 {i+1} 处理失败: {result.get('error', '未知错误')}")
                    failed_tests += 1
                    
            except Exception as e:
                logger.error(f"处理图片 {i+1} 时发生错误: {e}")
                failed_tests += 1
                continue
        
        print(f"\n📊 正在生成测试报告...")
        report = tester.generate_test_report(session_id)
        
        print(f"\n🎉 测试会话完成: {args.session_name}")
        print(f"✅ 成功处理: {successful_tests} 张图片")
        print(f"❌ 处理失败: {failed_tests} 张图片")
        
        if report:
            print(f"📈 最终准确率: {report['accuracy_rate']:.2%}")
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        print("\n⚠️ 测试被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
