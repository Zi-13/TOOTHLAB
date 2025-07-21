import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time
from urllib.parse import urlparse
import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CloudImageDownloader:
    """云盘图片下载器"""
    
    def __init__(self, download_dir: str = "cloud_images"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.timeout = 30
        self.retry_times = 3
        
        logger.info(f"云盘下载器初始化完成，下载目录: {self.download_dir}")
    
    def download_from_url(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """从URL下载图片"""
        try:
            logger.info(f"开始下载图片: {url}")
            
            if filename is None:
                parsed_url = urlparse(url)
                filename = Path(parsed_url.path).name
                if not filename or '.' not in filename:
                    filename = f"image_{int(time.time())}.jpg"
            
            file_path = self.download_dir / filename
            if file_path.suffix.lower() not in self.supported_formats:
                file_path = file_path.with_suffix('.jpg')
            
            for attempt in range(self.retry_times):
                try:
                    response = requests.get(url, timeout=self.timeout, stream=True)
                    response.raise_for_status()
                    
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        logger.error(f"文件过大: {content_length} bytes > {self.max_file_size} bytes")
                        return None
                    
                    total_size = int(content_length) if content_length else 0
                    with open(file_path, 'wb') as f:
                        if total_size > 0:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                                        pbar.update(len(chunk))
                        else:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    
                    if self.validate_image(file_path):
                        logger.info(f"图片下载成功: {file_path}")
                        return file_path
                    else:
                        logger.error(f"下载的文件不是有效图片: {file_path}")
                        file_path.unlink(missing_ok=True)
                        return None
                        
                except requests.RequestException as e:
                    logger.warning(f"下载尝试 {attempt + 1} 失败: {e}")
                    if attempt == self.retry_times - 1:
                        logger.error(f"下载失败，已重试 {self.retry_times} 次")
                        return None
                    time.sleep(2 ** attempt)  # 指数退避
                    
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            return None
    
    def download_batch(self, urls: List[str]) -> List[Path]:
        """批量下载图片"""
        logger.info(f"开始批量下载 {len(urls)} 张图片")
        downloaded_files = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"下载进度: {i}/{len(urls)}")
            file_path = self.download_from_url(url, f"batch_image_{i:03d}.jpg")
            if file_path:
                downloaded_files.append(file_path)
            else:
                logger.warning(f"跳过无效URL: {url}")
        
        logger.info(f"批量下载完成，成功下载 {len(downloaded_files)} 张图片")
        return downloaded_files
    
    def validate_image(self, image_path: Path) -> bool:
        """验证图片格式和完整性"""
        try:
            if not image_path.exists():
                return False
            
            if image_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"不支持的图片格式: {image_path.suffix}")
                return False
            
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"OpenCV无法读取图片: {image_path}")
                return False
            
            height, width = img.shape[:2]
            if height < 50 or width < 50:
                logger.warning(f"图片尺寸过小: {width}x{height}")
                return False
            
            if height > 10000 or width > 10000:
                logger.warning(f"图片尺寸过大: {width}x{height}")
                return False
            
            logger.debug(f"图片验证通过: {image_path} ({width}x{height})")
            return True
            
        except Exception as e:
            logger.error(f"图片验证失败: {e}")
            return False
    
    def get_downloaded_images(self) -> List[Path]:
        """获取已下载的所有图片"""
        images = []
        for ext in self.supported_formats:
            images.extend(self.download_dir.glob(f"*{ext}"))
            images.extend(self.download_dir.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def clear_download_dir(self):
        """清空下载目录"""
        for file_path in self.download_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
        logger.info(f"已清空下载目录: {self.download_dir}")
