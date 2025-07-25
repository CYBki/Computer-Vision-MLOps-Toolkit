import os
from pathlib import Path

def create_dirs(base):
    for split in ["train", "val", "test"]:
        (base / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (base / f"labels/{split}").mkdir(parents=True, exist_ok=True)

def symlink_all(task_root: Path, merged_root: Path):
    create_dirs(merged_root)
    for task in task_root.glob("task_*"):
        for split in ["train", "val", "test"]:
            img_dir = task / f"images/{split}"
            lbl_dir = task / f"labels/{split}"
            if not img_dir.exists():
                continue
            for img_file in img_dir.glob("*.[jp][pn]g"):
                target_img = merged_root / f"images/{split}" / img_file.name
                try:
                    os.symlink(img_file.resolve(), target_img)
                except FileExistsError:
                    pass
            for lbl_file in lbl_dir.glob("*.txt"):
                target_lbl = merged_root / f"labels/{split}" / lbl_file.name
                try:
                    os.symlink(lbl_file.resolve(), target_lbl)
                except FileExistsError:
                    pass
    print("✅ All symlinks have been created.")

def create_final_yaml(save_path):
    data = {
        "train": str((save_path / "images/train").resolve()),
        "val": str((save_path / "images/val").resolve()),
        "test": str((save_path / "images/test").resolve()),
        "nc": 2,  # Enter the actual number of classes
        "names": ["car", "truck"]  # Replace with actual class names
    }
    import yaml
    with open(save_path / "data.yaml", "w") as f:
        yaml.dump(data, f)
    print("✅ Final data.yaml has been created.")

# Main process
task_root = Path("/home/user/your_root")
merged_root = Path("/home/user/merged_yolo")

symlink_all(task_root, merged_root)
create_final_yaml(merged_root)
