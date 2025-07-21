#!/usr/bin/env python3
"""
启动TOOTHLAB Web服务器的便捷脚本
"""
import os
import sys
import uvicorn
from pathlib import Path

def main():
    """启动Web服务器"""
    print("🚀 启动TOOTHLAB Web服务器")
    print("=" * 50)
    
    try:
        from web_api import app
        print("✅ Web API模块加载成功")
    except ImportError as e:
        print(f"❌ 无法导入Web API模块: {e}")
        print("💡 请确保已安装所有依赖: pip install -r requirements.txt")
        return 1
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🌐 服务器将在 http://{host}:{port} 启动")
    print("📱 Web界面功能:")
    print("   - 模板管理: 批量上传和管理牙齿模板")
    print("   - 识别系统: 上传照片进行匹配识别")
    print("   - 系统统计: 查看运行状态和数据")
    print("\n🔧 API端点:")
    print("   - POST /api/batch_upload - 批量上传模板")
    print("   - POST /api/recognize - 识别单个图像")
    print("   - GET /api/templates - 获取模板列表")
    print("   - DELETE /api/templates/{id} - 删除模板")
    
    print(f"\n🎯 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        return 0
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
