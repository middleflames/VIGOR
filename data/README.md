# Data

All datasets used by VIGOR live under this single `data/` directory. Paths in
`configs/vigor_talk2car.yaml` are resolved relative to the repository root, so
place the files exactly as shown below (the actual data is not tracked in git).

## Expected layout
Refer to [Talk2car](https://github.com/talk2car/Talk2Car) for images.
Other files are uploaded in Google Drive: [data](https://drive.google.com/drive/folders/1ks2U6RTq7CGKr8DGUJW-OzcTGSsPaXIs?usp=drive_link)
```
data/
└── talk2car/
    ├── images/                 # RGB camera frames referenced by ref["file_name"]
    ├── lidar_features/
    │   └── train/              # per-sample LiDAR feature tensors: <name>.pt
    ├── talk2car_key.pt         # dict: sample_token -> LiDAR feature file name
    ├── talk2car_coco.p         # REFER referring expressions (pickle)
    └── talk2car_coco.json      # REFER annotations (COCO-style instances)
```