#!/usr/bin/env python3
import argparse
import sys
from drive_sync import DriveSync
from config import Config
import os

def main():
    parser = argparse.ArgumentParser(description='TOOTHLAB Google Drive Sync Manager')
    parser.add_argument('action', choices=['mount', 'unmount', 'upload', 'download', 'setup', 'status'],
                       help='Action to perform')
    parser.add_argument('--local-path', default='.', help='Local path for sync operations')
    parser.add_argument('--drive-path', help='Drive path for sync operations')
    parser.add_argument('--use-drive', action='store_true', help='Switch to using Drive for operations')
    
    args = parser.parse_args()
    
    sync = DriveSync()
    config = Config()
    
    if args.action == 'mount':
        if sync.mount_drive():
            print("✅ Google Drive mounted successfully")
            if args.use_drive:
                os.environ['TOOTHLAB_USE_DRIVE'] = 'true'
                print("✅ Switched to Drive mode")
        else:
            print("❌ Failed to mount Google Drive")
            sys.exit(1)
            
    elif args.action == 'unmount':
        if sync.unmount_drive():
            print("✅ Google Drive unmounted successfully")
            os.environ['TOOTHLAB_USE_DRIVE'] = 'false'
            print("✅ Switched to local mode")
        else:
            print("❌ Failed to unmount Google Drive")
            sys.exit(1)
            
    elif args.action == 'upload':
        local_path = args.local_path
        drive_path = args.drive_path
        if sync.sync_to_drive(local_path, drive_path):
            print(f"✅ Uploaded {local_path} to Drive")
        else:
            print(f"❌ Failed to upload {local_path}")
            sys.exit(1)
            
    elif args.action == 'download':
        drive_path = args.drive_path
        local_path = args.local_path
        if sync.sync_from_drive(drive_path, local_path):
            print(f"✅ Downloaded from Drive to {local_path}")
        else:
            print(f"❌ Failed to download from Drive")
            sys.exit(1)
            
    elif args.action == 'setup':
        if sync.setup_drive_structure():
            print("✅ Drive directory structure created")
        else:
            print("❌ Failed to setup Drive structure")
            sys.exit(1)
            
    elif args.action == 'status':
        print(f"Drive mounted: {sync.is_mounted()}")
        print(f"Mount point: {sync.mount_point}")
        print(f"Using Drive: {config.use_drive}")
        print(f"Current photo path: {config.get_photo_path()}")
        print(f"Templates directory: {config.get_templates_dir()}")

if __name__ == '__main__':
    main()
