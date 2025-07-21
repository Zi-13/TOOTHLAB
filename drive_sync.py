import subprocess
import os
import sys
from pathlib import Path
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriveSync:
    def __init__(self):
        self.config = Config()
        self.mount_point = self.config.drive_mount_point
        self.remote_name = "gdrive"
        
    def is_mounted(self):
        """Check if Google Drive is mounted"""
        return self.mount_point.exists() and any(self.mount_point.iterdir())
        
    def mount_drive(self):
        """Mount Google Drive using rclone"""
        try:
            if self.is_mounted():
                logger.info("Google Drive already mounted")
                return True
                
            self.mount_point.mkdir(parents=True, exist_ok=True)
            cmd = f"rclone mount {self.remote_name}: {self.mount_point} --daemon"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Google Drive mounted at {self.mount_point}")
                return True
            else:
                logger.error(f"Failed to mount: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Mount error: {e}")
            return False
            
    def unmount_drive(self):
        """Unmount Google Drive"""
        try:
            if not self.is_mounted():
                logger.info("Google Drive not mounted")
                return True
                
            cmd = f"fusermount -u {self.mount_point}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Google Drive unmounted")
                return True
            else:
                logger.error(f"Failed to unmount: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Unmount error: {e}")
            return False
            
    def sync_to_drive(self, local_path, drive_path=None):
        """Upload local files to Google Drive"""
        try:
            if drive_path is None:
                drive_path = f"{self.remote_name}:TOOTHLAB/"
                
            cmd = f"rclone sync {local_path} {drive_path} -v"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Synced {local_path} to Drive")
                return True
            else:
                logger.error(f"Sync to Drive failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Sync to Drive error: {e}")
            return False
            
    def sync_from_drive(self, drive_path=None, local_path="."):
        """Download files from Google Drive to local"""
        try:
            if drive_path is None:
                drive_path = f"{self.remote_name}:TOOTHLAB/"
                
            cmd = f"rclone sync {drive_path} {local_path} -v"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Synced from Drive to {local_path}")
                return True
            else:
                logger.error(f"Sync from Drive failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Sync from Drive error: {e}")
            return False
            
    def setup_drive_structure(self):
        """Create necessary directory structure on Google Drive"""
        try:
            directories = [
                f"{self.remote_name}:TOOTHLAB/",
                f"{self.remote_name}:TOOTHLAB/templates/",
                f"{self.remote_name}:TOOTHLAB/templates/contours/",
                f"{self.remote_name}:TOOTHLAB/templates/images/",
                f"{self.remote_name}:TOOTHLAB/templates/features/",
                f"{self.remote_name}:TOOTHLAB/images/"
            ]
            
            for directory in directories:
                cmd = f"rclone mkdir {directory}"
                subprocess.run(cmd, shell=True, capture_output=True)
                
            logger.info("Drive directory structure created")
            return True
        except Exception as e:
            logger.error(f"Setup Drive structure error: {e}")
            return False
