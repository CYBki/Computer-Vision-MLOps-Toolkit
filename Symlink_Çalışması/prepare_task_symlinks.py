import os
import random
import yaml
from pathlib import Path

def split_and_symlink(task_path: Path, val_ratio=0.2, test_ratio=0.1):
    img_train = task_path / "images/Train"
    lbl_train = task_path / "labels/Train"
    assert img_train.exists() and lbl_train.exists(), "Eksik klasör"

    # Hedef klasörleri oluştur
    for split in ["train", "val", "test"]:
        (task_path / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (task_path / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    # Görselleri al
    image_files = list(img_train.glob("*.jpg")) + list(img_train.glob("*.png"))
    random.shuffle(image_files)

    total = len(image_files)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    val_files = image_files[:val_count]
    test_files = image_files[val_count:val_count + test_count]
    train_files = image_files[val_count + test_count:]

    def link(files, split):
        for img_path in files:
            lbl_path = lbl_train / img_path.with_suffix(".txt").name
            os.symlink(img_path, task_path / f"images/{split}" / img_path.name)
            if lbl_path.exists():
                os.symlink(lbl_path, task_path / f"labels/{split}" / lbl_path.name)

    link(train_files, "train")
    link(val_files, "val")
    link(test_files, "test")

    print(f"{task_path.name} için symlink oluşturuldu ✅")

    # data.yaml güncelle
    new_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,  # Burayı gerçek değere göre alman gerekebilir
        "names": ["car", "truck"]  # Gerçek sınıf isimleri ile değiştir
    }
    with open(task_path / "data.yaml", "w") as f:
        yaml.dump(new_data, f)

    print(f"{task_path.name}/data.yaml güncellendi ✅\n")

# Tüm task'leri uygula
root_dir = Path("/home/seyitaliyorgun/kendi_datasetiniz") # bu orijinal datasetinizin path'ini içermeli
for task in root_dir.glob("task_*"):
    split_and_symlink(task)