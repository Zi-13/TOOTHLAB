"""
跨域匹配配置：3D建模图 vs 现实照片
"""

class CrossDomainConfig:
    MODELING_IMAGE_CONFIG = {
        'hsv_range': {
            'lower': [0, 0, 0],
            'upper': [180, 255, 50]
        },
        'similarity_weights': {
            'geometric': 0.4,
            'hu_moments': 0.3,
            'fourier': 0.3
        },
        'thresholds': {
            'coarse_match': 0.2,
            'fine_match': 0.5
        }
    }
    
    REAL_PHOTO_CONFIG = {
        'hsv_range': {
            'lower': [0, 0, 0],
            'upper': [180, 255, 100]
        },
        'similarity_weights': {
            'geometric': 0.3,
            'hu_moments': 0.4,
            'fourier': 0.3
        },
        'thresholds': {
            'coarse_match': 0.15,
            'fine_match': 0.4
        }
    }
    
    CROSS_DOMAIN_CONFIG = {
        'similarity_weights': {
            'geometric': 0.5,
            'hu_moments': 0.3,
            'fourier': 0.2
        },
        'thresholds': {
            'coarse_match': 0.1,
            'fine_match': 0.3
        }
    }
