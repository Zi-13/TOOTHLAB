import os
from pathlib import Path

class Config:
    def __init__(self):
        self.drive_mount_point = Path.home() / "gdrive"
        self.local_templates_dir = Path("templates")
        self.drive_templates_dir = self.drive_mount_point / "TOOTHLAB" / "templates"
        self.use_drive = os.getenv("TOOTHLAB_USE_DRIVE", "false").lower() == "true"
        
    def get_photo_path(self):
        """Get the current photo path based on configuration"""
        if self.use_drive:
            return str(self.drive_mount_point / "TOOTHLAB" / "images" / "current_image.png")
        else:
            return "current_image.png"  # Use local test image
            
    def get_templates_dir(self):
        """Get the templates directory path"""
        if self.use_drive:
            return str(self.drive_templates_dir)
        else:
            return str(self.local_templates_dir)
            
    def get_database_path(self):
        """Get the database path"""
        if self.use_drive:
            return str(self.drive_mount_point / "TOOTHLAB" / "tooth_templates.db")
        else:
            return "tooth_templates.db"
