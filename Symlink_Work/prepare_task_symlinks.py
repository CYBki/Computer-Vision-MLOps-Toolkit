import os
import random
import yaml
from pathlib import Path


def split_and_symlink(task_path: Path, cfg: dict, rng: random.Random) -> None:
    """Split dataset files and create symlinks based on a config.

    Parameters
    ----------
    task_path: Path
        Path to the task directory containing ``images/Train`` and ``labels/Train``.
    cfg: dict
        Configuration dictionary with ``splits`` and ``seed``.
    rng: random.Random
        Random number generator initialised with ``cfg['seed']`` to ensure
        reproducible shuffling.
    """

    img_train = task_path / "images/Train"
    lbl_train = task_path / "labels/Train"
    assert img_train.exists() and lbl_train.exists(), "Missing folder"

    # Create target directories
    for split in ["train", "val", "test"]:
        (task_path / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (task_path / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    # Get image files and deterministically shuffle
    image_files = list(img_train.glob("*.jpg")) + list(img_train.glob("*.png"))
    rng.shuffle(image_files)

    total = len(image_files)
    val_ratio = cfg["splits"]["val"] / 100
    test_ratio = cfg["splits"]["test"] / 100
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    val_files = image_files[:val_count]
    test_files = image_files[val_count : val_count + test_count]
    train_files = image_files[val_count + test_count :]

    def link(files, split):
        for img_path in files:
            lbl_path = lbl_train / img_path.with_suffix(".txt").name
            os.symlink(img_path, task_path / f"images/{split}" / img_path.name)
            if lbl_path.exists():
                os.symlink(lbl_path, task_path / f"labels/{split}" / lbl_path.name)

    link(train_files, "train")
    link(val_files, "val")
    link(test_files, "test")

    print(f"Symlinks created for {task_path.name} ✅")

    # Update data.yaml
    new_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,  # You may need to update this with the actual number of classes
        "names": ["car", "truck"],  # Replace with actual class names
    }
    with open(task_path / "data.yaml", "w") as f:
        yaml.dump(new_data, f)

    # Save split metadata for reproducibility
    meta = {
        "seed": cfg["seed"],
        "total_images": total,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }
    with open(task_path / "split_meta.yaml", "w") as f:
        yaml.dump(meta, f)

    print(f"{task_path.name}/data.yaml updated ✅\n")


# Load configuration
with open(Path(__file__).parent / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

stable_random = random.Random(CONFIG.get("seed", 42))
root_dir = Path(CONFIG.get("dataset_root", "/home/user/your_dataset"))

# Apply to all tasks using a stable RNG
for task in root_dir.glob("task_*"):
    split_and_symlink(task, CONFIG, stable_random)
