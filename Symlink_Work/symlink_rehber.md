# YOLO Dataset Hazırlama Rehberi

Bu rehber, YOLO object detection için dataset hazırlama sürecini adım adım açıklar. İki Python scripti kullanarak görsellerinizi train/val/test setlerine ayırabilir ve final dataset'i oluşturabilirsiniz.

## Gereksinimler

- Python 3.x
- PyYAML kütüphanesi (`pip install pyyaml`)
- Linux/Unix sistemi (symlink desteği için)

## Klasör Yapısı

Başlangıçta dataset'iniz şu yapıda olmalı:

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

## Adım 1: Dataset Bölme ve Symlink Oluşturma

### `prepare_task_symlinks.py` Scripti

Bu script her task klasörü için aşağıdaki işlemleri yapar:

1. **Görsel dosyalarını bulur**: Train klasöründeki .jpg ve .png dosyalarını tarar
2. **Rastgele karıştırır**: Veriyi objektif şekilde böler
3. **Oranlarına göre böler**:
   - Test: %10
   - Validation: %20  
   - Train: %70
4. **Symlink oluşturur**: Dosyaları kopyalamak yerine symbolic link kullanır (disk tasarrufu)
5. **data.yaml oluşturur**: YOLO için gerekli konfigürasyon dosyası

### Kullanım

1. Script'teki path'i düzenleyin:
```python
root_dir = Path("/home/seyitaliyorgun/kendi_datasetiniz")  # Kendi path'inizi yazın
```

2. Class bilgilerini güncelleyin:
```python
"nc": 2,  # Sınıf sayınız
"names": ["car", "truck"]  # Gerçek sınıf isimleriniz
```

3. Script'i çalıştırın:
```bash
python prepare_task_symlinks.py
```

### Sonuç

Her task klasöründe şu yapı oluşur:

```
task_1/
├── images/
│   ├── Train/           # Orijinal dosyalar
│   ├── train/           # Symlink'ler (%70)
│   ├── val/             # Symlink'ler (%20)
│   └── test/            # Symlink'ler (%10)
├── labels/
│   ├── Train/           # Orijinal etiketler
│   ├── train/           # Symlink'ler
│   ├── val/             # Symlink'ler
│   └── test/            # Symlink'ler
└── data.yaml            # YOLO config dosyası
```

## Adım 2: Final Dataset Birleştirme

### `merge_final_dataset.py` Scripti

Bu script tüm task'lerdeki bölünmüş veriyi tek bir dataset'te toplar.

### Kullanım

1. Path'leri düzenleyin:
```python
task_root = Path("/home/seyitaliyorgun/sett")        # Task'lerin bulunduğu klasör
merged_root = Path("/home/seyitaliyorgun/merged_yolo") # Final dataset klasörü
```

2. Final yaml'daki class bilgilerini kontrol edin:
```python
"nc": 2,  # Toplam sınıf sayısı
"names": ["car", "truck"]  # Tüm sınıf isimleri
```

3. Script'i çalıştırın:
```bash
python merge_final_dataset.py
```

### Sonuç

Final dataset yapısı:

```
merged_yolo/
├── images/
│   ├── train/           # Tüm task'lerden train görselleri
│   ├── val/             # Tüm task'lerden val görselleri
│   └── test/            # Tüm task'lerden test görselleri
├── labels/
│   ├── train/           # Karşılık gelen etiketler
│   ├── val/             # Karşılık gelen etiketler
│   └── test/            # Karşılık gelen etiketler
└── data.yaml            # Final YOLO config
```

## Önemli Notlar

### Symlink Avantajları
- **Disk tasarrufu**: Dosyalar kopyalanmaz, sadece referans oluşturulur
- **Hız**: Büyük dataset'lerde çok daha hızlı
- **Güncellenebilirlik**: Orijinal dosya değişirse symlink otomatik güncellenir

### Dosya İsmi Çakışmaları
Script, aynı isimli dosyalar için `FileExistsError` yakalayarak çakışmaları önler.

### Label Formatı
YOLO formatında etiketler beklenir:
```
class_id center_x center_y width height
```
Koordinatlar 0-1 arasında normalize edilmiş olmalı.

## Troubleshooting

### Hata: "Eksik klasör"
- `images/Train` ve `labels/Train` klasörlerinin var olduğundan emin olun

### Symlink hataları
- Linux/Unix sistem kullandığınızdan emin olun
- Dosya izinlerini kontrol edin

### YAML hatası
- PyYAML kütüphanesinin yüklü olduğundan emin olun: `pip install pyyaml`

## YOLO ile Kullanım

Final dataset hazır olduktan sonra YOLO training için kullanabilirsiniz:

```bash
yolo train data=/path/to/merged_yolo/data.yaml model=yolov8n.pt epochs=100
```

Bu rehberi takip ederek dataset'inizi YOLO için hazır hale getirebilirsiniz.