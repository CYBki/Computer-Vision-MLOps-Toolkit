```markdown
# YOLO Dataset Preparation Guide

This guide walks you through preparing a dataset for YOLO object detection. Using two Python scripts, you can split your images into train/val/test sets and generate a final dataset ready for training.

## Requirements

- Python 3.x  
- PyYAML library (`pip install pyyaml`)  
- Linux/Unix system (for symlink support)

## Folder Structure

Initially, your dataset should look like this:

```

your\_dataset/
├── task\_1/
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
├── task\_2/
│   ├── images/Train/
│   └── labels/Train/
└── ...

````

## Step 1: Splitting Dataset and Creating Symlinks

### `prepare_task_symlinks.py` Script

This script performs the following actions for each task folder:

1. **Finds image files**: Scans for `.jpg` and `.png` files in the `Train` folder  
2. **Shuffles the data**: Ensures unbiased splitting  
3. **Splits by ratio**:
   - Test: 10%
   - Validation: 20%
   - Train: 70%
4. **Creates symlinks**: Uses symbolic links instead of copying files (saves disk space)  
5. **Generates `data.yaml`**: The configuration file required by YOLO

### Usage

1. Adjust the dataset path in the script:

    ```python
    root_dir = Path("/home/user/your_dataset")  # Replace with your actual path
    ```

2. Update class info:

    ```python
    "nc": 2,  # Number of classes
    "names": ["car", "truck"]  # Replace with your actual class names
    ```

3. Run the script:

    ```bash
    python prepare_task_symlinks.py
    ```

### Result

Each task folder will now have the following structure:

````

task\_1/
├── images/
│   ├── Train/           # Original images
│   ├── train/           # Symlinks (70%)
│   ├── val/             # Symlinks (20%)
│   └── test/            # Symlinks (10%)
├── labels/
│   ├── Train/           # Original labels
│   ├── train/           # Symlinks
│   ├── val/             # Symlinks
│   └── test/            # Symlinks
└── data.yaml            # YOLO configuration file

````

## Step 2: Merging the Final Dataset

### `merge_final_dataset.py` Script

This script collects split data from all tasks and merges them into a single YOLO-compatible dataset.

### Usage

1. Set the paths in the script:

    ```python
    task_root = Path("/home/user/your_root")           # Folder with all task folders
    merged_root = Path("/home/user/your_merged_yolo")  # Final merged dataset folder
    ```

2. Update class info in the final `data.yaml`:

    ```python
    "nc": 2,  # Total number of classes
    "names": ["car", "truck"]  # All class names
    ```

3. Run the script:

    ```bash
    python merge_final_dataset.py
    ```

### Result

The final dataset structure:

````

merged\_yolo/
├── images/
│   ├── train/           # Train images from all tasks
│   ├── val/             # Validation images from all tasks
│   └── test/            # Test images from all tasks
├── labels/
│   ├── train/           # Corresponding label files
│   ├── val/
│   └── test/
└── data.yaml            # Final YOLO configuration file

```

## Notes

### Symlink Benefits

- **Disk efficient**: No duplication, just references
- **Fast**: Much quicker for large datasets
- **Automatically updated**: If original files change, symlinks reflect changes instantly

### Filename Collisions

The script handles duplicate filenames using `FileExistsError` to avoid conflicts.

### Label Format

YOLO expects labels in this format:

```

class\_id center\_x center\_y width height

````

All values must be normalized between 0 and 1.

## Troubleshooting

### Error: "Missing folder"

Ensure both `images/Train` and `labels/Train` directories exist.

### Symlink issues

- Make sure you’re on a Linux/Unix system  
- Check file permissions if links aren't created

### YAML errors

- Ensure PyYAML is installed:

    ```bash
    pip install pyyaml
    ```

## Using the Dataset with YOLO

Once the dataset is ready, you can train YOLO like this:

```bash
yolo train data=/path/to/merged_yolo/data.yaml model=yolov8n.pt epochs=100
````

By following this guide, you’ll have your dataset fully prepared for YOLO training.

```
