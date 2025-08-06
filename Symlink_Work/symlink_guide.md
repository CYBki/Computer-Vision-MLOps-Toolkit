# YOLO Dataset Preparation Guide

This guide explains how to prepare a dataset for YOLO object detection step by step. Using two Python scripts, you can split your images into train/val/test sets and generate a final dataset.

## Requirements

- Python 3.x
- PyYAML library (`pip install pyyaml`)
- Linux/Unix system (for symlink support)

## Folder Structure

Initially, your dataset should be structured like this:

```
kendi_datasetiniz/
├── task_1/
│   ├── images/
│   │   └── Train/
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
│   └── labels/
│       └── Train/
│           ├── image1.txt
│           ├── image2.txt
│           └── ...
├── task_2/
│   ├── images/Train/
│   └── labels/Train/
└── ...
```

## Step 1: Splitting Dataset and Creating Symlinks

### `prepare_task_symlinks.py` Script

This script performs the following steps for each task folder:

1. **Reads configuration**: `config.yaml` defines split ratios, dataset path and a random seed.
2. **Finds image files**: Scans for `.jpg` and `.png` files in the `Train` folder.
3. **Shuffles deterministically**: Uses a stable RNG seeded from the config for reproducible splits.
4. **Splits according to ratios**: Defaults are Test %10, Validation %20 and Train %70.
5. **Creates symlinks**: Uses symbolic links instead of copying files (saves disk space).
6. **Generates data.yaml and metadata**: Creates required YOLO config and `split_meta.yaml` with counts and seed.

### Usage

1. Edit the values in `config.yaml`:
   - `dataset_root`: Path to the dataset containing `task_*` folders.
   - `seed`: Seed value for reproducible shuffling.
   - `splits`: Train/val/test percentages.

2. Update class information inside `prepare_task_symlinks.py` if needed:
```python
"nc": 2,  # Number of classes
"names": ["car", "truck"]  # Your actual class names
```

3. Run the script:
```bash
python prepare_task_symlinks.py
```

### Result

Each task folder will have the following structure:

```
task_1/
├── images/
│   ├── Train/           # Original files
│   ├── train/           # Symlinks (%70)
│   ├── val/             # Symlinks (%20)
│   └── test/            # Symlinks (%10)
├── labels/
│   ├── Train/           # Original labels
│   ├── train/           # Symlinks
│   ├── val/             # Symlinks
│   └── test/            # Symlinks
├── data.yaml            # YOLO config file
└── split_meta.yaml      # Metadata containing seed and file counts
```

## Step 2: Merging the Final Dataset

### `merge_final_dataset.py` Script

This script collects split data from all tasks into a single dataset.

### Usage

1. Set the paths:
```python
task_root = Path("/home/user/your_rout")        # Folder where tasks are located
merged_root = Path("/home/user/your_routed_yolo") # Final dataset folder
```

2. Verify the class info in the final YAML:
```python
"nc": 2,  # Total number of classes
"names": ["car", "truck"]  # All class names
```

3. Run the script:
```bash
python merge_final_dataset.py
```

### Result

Final dataset structure:

```
merged_yolo/
├── images/
│   ├── train/           # Train images from all tasks
│   ├── val/             # Validation images from all tasks
│   └── test/            # Test images from all tasks
├── labels/
│   ├── train/           # Corresponding label files
│   ├── val/             # Corresponding label files
│   └── test/            # Corresponding label files
└── data.yaml            # Final YOLO config
```

## Important Notes

### Benefits of Symlinks

- **Disk efficient**: Files are not duplicated—only references (symlinks) are created  
- **Faster**: Much quicker when working with large datasets  
- **Automatically updated**: If the original file changes, the symlink reflects the update immediately


### Filename Conflicts
The script catches `FileExistsError` for files with duplicate names to avoid conflicts.

### Label Format
YOLO expects labels in this format:
```
class_id center_x center_y width height
```
Coordinates must be normalized between 0 and 1.

## Troubleshooting

### Error: "Missing folder"
- Make sure `images/Train` and `labels/Train` directories exist

### Symlink errors
- Ensure you are using a Linux/Unix system
- Check file permissions

### YAML error
- Make sure the PyYAML library is installed: `pip install pyyaml`

## Using with YOLO

Once the dataset is ready, you can use it for YOLO training:

```bash
yolo train data=/path/to/merged_yolo/data.yaml model=yolov8n.pt epochs=100
```

By following this guide, you'll have a dataset ready for YOLO.
