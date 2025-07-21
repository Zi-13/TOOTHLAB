import os
from pathlib import Path

class Config:
    """系统配置类"""
    
    BASE_DIR = Path(__file__).parent
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"
    DATABASE_PATH = BASE_DIR / "tooth_templates.db"
    
    UPLOAD_MAX_SIZE = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    FOURIER_ORDER = 10
    
    SIMILARITY_WEIGHTS = {
        'geometric': 0.3,
        'hu_moments': 0.3,
        'fourier': 0.4
    }
    
    MATCHING_THRESHOLDS = {
        'coarse': 0.3,
        'fine': 0.7
    }
    
    @classmethod
    def get_photo_path(cls):
        """获取默认图片路径"""
        return os.getenv('PHOTO_PATH', str(cls.BASE_DIR / "test_images" / "sample.jpg"))
