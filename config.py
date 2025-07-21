"""
自我测试配置文件
"""

CLOUD_CONFIG = {
    'download_dir': 'cloud_images',
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'timeout': 30,
    'retry_times': 3,
    'supported_formats': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
}

TEST_CONFIG = {
    'similarity_threshold': 0.8,
    'auto_confirm_threshold': 0.95,  # 高于此阈值自动确认为正确
    'batch_size': 10,
    'results_export_format': 'json',
    'min_contour_points': 20,
    'test_results_db': 'self_test_results.db'
}

UI_CONFIG = {
    'figure_size': (20, 14),
    'font_size': 12,
    'button_size': (0.1, 0.04),
    'confirmation_timeout': 300,  # 5分钟超时
    'show_debug_info': True
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'self_test.log'
}
