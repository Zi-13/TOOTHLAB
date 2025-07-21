# Google Drive Integration Setup

## Initial Setup

1. Configure rclone for Google Drive:
```bash
rclone config
```
- Choose "New remote"
- Name: `gdrive`
- Type: `drive` (option 22)
- Leave client_id and client_secret blank (use defaults)
- Choose scope 1 (Full access all files)
- Leave service_account_file blank
- Choose "n" for advanced config
- Choose "n" for web browser authentication
- **IMPORTANT**: You'll need to run the authorization command on a machine with a web browser:
  ```bash
  rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlIn0"
  ```
- Copy the resulting token and paste it when prompted

2. Setup Drive directory structure:
```bash
python sync_manager.py setup
```

## Usage

### Mount Google Drive
```bash
python sync_manager.py mount --use-drive
```

### Upload local data to Drive
```bash
python sync_manager.py upload --local-path templates
python sync_manager.py upload --local-path tooth_templates.db --drive-path gdrive:TOOTHLAB/
```

### Download data from Drive
```bash
python sync_manager.py download --drive-path gdrive:TOOTHLAB/templates --local-path templates
```

### Check status
```bash
python sync_manager.py status
```

### Unmount Drive
```bash
python sync_manager.py unmount
```

## Environment Variables

- `TOOTHLAB_USE_DRIVE=true`: Use Google Drive for operations
- `TOOTHLAB_USE_DRIVE=false`: Use local files (default)

## Workflow

1. Mount Drive and switch to Drive mode
2. Run BulidTheLab.py or match.py - they will use Drive paths
3. Sync changes back to Drive as needed
4. Unmount when done

## Bidirectional Sync

The system supports bidirectional sync:
- **Upload**: Local changes can be synced to Google Drive
- **Download**: Drive changes can be synced to local system
- **Manual triggers**: All sync operations are manually triggered via sync_manager.py

## Directory Structure

```
Google Drive/TOOTHLAB/
├── images/           # Input images for processing
├── templates/        # Generated templates
│   ├── contours/    # JSON contour data
│   ├── images/      # PNG visualization files
│   └── features/    # Feature-only JSON files
└── tooth_templates.db # SQLite database
```
