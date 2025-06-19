# ClearML ile YOLOv Modelleri Local Eğitim Rehberi

Bu rehber, ClearML kullanarak YOLOv modellerini tamamen local ortamda eğitmenizi ve dış dünyayla veri paylaşımı yapmadan UI üzerinden izlemenizi sağlar.

## İçindekiler
1. [Sistem Gereksinimleri](#sistem-gereksinimleri)
2. [ClearML Server Kurulumu](#clearml-server-kurulumu)
3. [ClearML Client Kurulumu](#clearml-client-kurulumu)
4. [YOLOv Ortamının Hazırlanması](#yolov-ortamının-hazırlanması)
5. [Model Eğitimi](#model-eğitimi)
6. [UI ile İzleme](#ui-ile-izleme)
7. [Alternatif Kurulum Yöntemleri](#alternatif-kurulum-yöntemleri)

## Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Ubuntu 18.04+, Windows 10+, macOS 10.15+
- **RAM**: 8GB (16GB önerilen)
- **GPU**: NVIDIA GPU (CUDA desteği ile)
- **Disk Alanı**: 50GB boş alan
- **Python**: 3.7-3.10 arası

### Gerekli Yazılımlar
- Docker ve Docker Compose
- Python 3.8+
- Git
- NVIDIA Docker (GPU kullanımı için)

## ClearML Server Kurulumu

### Yöntem 1: Docker Compose ile Kurulum (Önerilen)

#### 1. ClearML Server Repository'sini İndirin
```bash
git clone https://github.com/allegroai/clearml-server.git
cd clearml-server
```

#### 2. Docker Compose ile Başlatın
```bash
# Tüm servisleri başlat
docker-compose -f docker-compose.yml up -d

# Sadece gerekli servisleri başlat (daha az kaynak kullanımı)
docker-compose -f docker-compose-minimal.yml up -d
```

#### 3. Servislerin Durumunu Kontrol Edin
```bash
docker-compose ps
```

### Yöntem 2: Manuel Docker Kurulumu

#### 1. MongoDB Başlatın
```bash
docker run -d --name clearml-mongo \
  -p 27017:27017 \
  -v clearml-mongo-data:/data/db \
  mongo:4.4
```

#### 2. Elasticsearch Başlatın
```bash
docker run -d --name clearml-elastic \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -v clearml-elastic-data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:7.6.2
```

#### 3. Redis Başlatın
```bash
docker run -d --name clearml-redis \
  -p 6379:6379 \
  -v clearml-redis-data:/data \
  redis:6.0-alpine
```

#### 4. ClearML API Server Başlatın
```bash
docker run -d --name clearml-apiserver \
  -p 8008:8008 \
  -e CLEARML_MONGODB_SERVICE_HOST=host.docker.internal \
  -e CLEARML_MONGODB_SERVICE_PORT=27017 \
  -e CLEARML_REDIS_SERVICE_HOST=host.docker.internal \
  -e CLEARML_REDIS_SERVICE_PORT=6379 \
  -e CLEARML_ELASTIC_SERVICE_HOST=host.docker.internal \
  -e CLEARML_ELASTIC_SERVICE_PORT=9200 \
  allegroai/clearml:latest-apiserver
```

#### 5. ClearML Web UI Başlatın
```bash
docker run -d --name clearml-webserver \
  -p 8080:80 \
  -e CLEARML_API_HOST=http://host.docker.internal:8008 \
  allegroai/clearml:latest-webserver
```

#### 6. ClearML File Server Başlatın
```bash
docker run -d --name clearml-fileserver \
  -p 8081:8081 \
  -v clearml-fileserver-data:/mnt/fileserver \
  allegroai/clearml:latest-fileserver
```

## ClearML Client Kurulumu

### 1. Python Sanal Ortamı Oluşturun
```bash
python -m venv clearml-env
source clearml-env/bin/activate  # Linux/Mac
# clearml-env\Scripts\activate  # Windows
```

### 2. ClearML'i Yükleyin
```bash
pip install clearml
pip install clearml-agent  # Agent için (opsiyonel)
```

### 3. ClearML'i Yapılandırın
```bash
clearml-init
```

Karşınıza çıkan sorulara şu şekilde cevap verin:
- **API Server**: `http://localhost:8008`
- **Web Server**: `http://localhost:8080`
- **File Server**: `http://localhost:8081`
- **Credentials**: Web UI'dan alacağınız kimlik bilgileri

### 4. Kimlik Bilgilerini Alın
1. Tarayıcıda `http://localhost:8080` adresine gidin
2. Sağ üstten **Settings > Workspace > Create new credentials** seçin
3. Oluşan `Access Key` ve `Secret Key`'i kopyalayın
4. Terminal'de yapılandırma sırasında bu bilgileri girin

## YOLOv Ortamının Hazırlanması

### 1. YOLOv5 Kurulumu
```bash
# YOLOv5 repository'sini klonlayın
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Gereksinimları yükleyin
pip install -r requirements.txt
pip install clearml
```

### 2. YOLOv8 Kurulumu (Alternatif)
```bash
pip install ultralytics
pip install clearml
```

### 3. Veri Setinizi Hazırlayın
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

#### data.yaml örneği:
```yaml
# Veri seti konfigürasyonu
path: /path/to/dataset  # veri seti root dizini
train: images/train     # eğitim resimlerinin yolu
val: images/val         # validasyon resimlerinin yolu
test: images/test       # test resimlerinin yolu (opsiyonel)

# Sınıflar
nc: 3  # sınıf sayısı
names: ['person', 'car', 'bike']  # sınıf isimleri
```

## Model Eğitimi

### YOLOv5 ile Eğitim

#### 1. ClearML Entegrasyonlu Eğitim Scripti
```python
# train_yolo_clearml.py
from clearml import Task, Dataset
import torch
import sys
import os

# ClearML Task oluştur
task = Task.init(project_name="YOLO_Training", 
                task_name="YOLOv5_Custom_Dataset")

# Parametreler
task.connect({
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'weights': 'yolov5s.pt',
    'data': 'data.yaml',
    'device': '0' if torch.cuda.is_available() else 'cpu'
})

# YOLOv5 eğitimini başlat
os.system(f"""
python train.py \
    --epochs {task.get_parameter('epochs')} \
    --batch-size {task.get_parameter('batch_size')} \
    --img {task.get_parameter('img_size')} \
    --weights {task.get_parameter('weights')} \
    --data {task.get_parameter('data')} \
    --device {task.get_parameter('device')} \
    --project runs/train \
    --name exp
""")

print("Eğitim tamamlandı!")
```

#### 2. Eğitimi Başlatın
```bash
python train_yolo_clearml.py
```

### YOLOv8 ile Eğitim

#### 1. YOLOv8 Eğitim Scripti
```python
# train_yolov8_clearml.py
from clearml import Task
from ultralytics import YOLO

# ClearML Task oluştur
task = Task.init(project_name="YOLO_Training", 
                task_name="YOLOv8_Custom_Dataset")

# Parametreler
params = {
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'data': 'data.yaml',
    'weights': 'yolov8n.pt'
}
task.connect(params)

# Model yükle
model = YOLO(params['weights'])

# Eğitimi başlat
results = model.train(
    data=params['data'],
    epochs=params['epochs'],
    imgsz=params['imgsz'],
    batch=params['batch'],
    project='runs/train',
    name='yolov8_exp'
)

print("YOLOv8 eğitimi tamamlandı!")
```

#### 2. Eğitimi Başlatın
```bash
python train_yolov8_clearml.py
```

## UI ile İzleme

### 1. Web UI'ya Erişim
Tarayıcınızda `http://localhost:8080` adresine gidin.

### 2. İzleyebileceğiniz Metrikler
- **Scalars**: Loss değerleri, mAP, precision, recall
- **Plots**: Confusion matrix, PR curves, F1 curves
- **Debug Samples**: Eğitim sırasında işlenen görüntüler
- **Models**: Eğitilen model dosyaları
- **Logs**: Konsol çıktıları ve hata mesajları

### 3. Experiment Karşılaştırması
- Farklı denemeleri yan yana karşılaştırın
- Hyperparameter optimizasyonu yapın
- En iyi modeli seçin

### 4. Model Registry
- Eğitilen modelleri kaydedin
- Model versiyonlarını takip edin
- Deployment için model seçin

## Alternatif Kurulum Yöntemleri

### Yöntem 1: ClearML Server Offline Kurulumu

#### 1. Docker İmajlarını Önceden İndirin
```bash
# Gerekli imajları çek
docker pull mongo:4.4
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.6.2
docker pull redis:6.0-alpine
docker pull allegroai/clearml:latest-apiserver
docker pull allegroai/clearml:latest-webserver
docker pull allegroai/clearml:latest-fileserver

# İmajları kaydet
docker save -o clearml-images.tar \
  mongo:4.4 \
  docker.elastic.co/elasticsearch/elasticsearch:7.6.2 \
  redis:6.0-alpine \
  allegroai/clearml:latest-apiserver \
  allegroai/clearml:latest-webserver \
  allegroai/clearml:latest-fileserver
```

#### 2. Offline Ortamda Yükle
```bash
# İmajları yükle
docker load -i clearml-images.tar

# Servisleri başlat
docker-compose up -d
```

### Yöntem 2: Conda ile Kurulum
```bash
# Conda ortamı oluştur
conda create -n clearml-yolo python=3.8
conda activate clearml-yolo

# Gerekli paketleri yükle
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install clearml ultralytics
```

### Yöntem 3: Virtual Environment ile Kurulum
```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Paketleri yükle
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install clearml ultralytics opencv-python
```

## Çalıştırma ve Test Aşamaları

### 1. Sistem Kontrolü
```bash
# GPU kontrolü
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# ClearML bağlantı kontrolü
python -c "from clearml import Task; print('ClearML bağlantısı başarılı!')"
```

### 2. Test Eğitimi
```python
# test_training.py
from clearml import Task
from ultralytics import YOLO

# Test task oluştur
task = Task.init(project_name="Test", task_name="YOLO_Test")

# Küçük bir test eğitimi
model = YOLO('yolov8n.pt')
results = model.train(
    data='coco128.yaml',  # Küçük test dataset
    epochs=3,
    imgsz=640,
    batch=8
)

print("Test eğitimi başarılı!")
```

### 3. Model Performans Testi
```python
# test_model.py
from ultralytics import YOLO
import cv2

# Eğitilen modeli yükle
model = YOLO('runs/train/exp/weights/best.pt')

# Test görüntüsü üzerinde tahmin
results = model('test_image.jpg')

# Sonuçları kaydet
results[0].save('prediction_result.jpg')
print("Model testi tamamlandı!")
```

### 4. Batch Tahmin Testi
```python
# batch_inference.py
from ultralytics import YOLO
import os

model = YOLO('runs/train/exp/weights/best.pt')

# Klasördeki tüm görüntüleri işle
test_folder = 'test_images/'
results = model(test_folder)

# Sonuçları kaydet
for i, result in enumerate(results):
    result.save(f'results/prediction_{i}.jpg')

print(f"{len(results)} görüntü işlendi!")
```

## Performans Optimizasyonu

### GPU Bellek Optimizasyonu
```python
# GPU bellek kullanımını optimize et
import torch

# Mixed precision training
torch.backends.cudnn.benchmark = True

# Gradient accumulation için batch size küçült
params = {
    'batch': 8,  # Küçük batch size
    'accumulate': 4  # 4 batch biriktir (etkili batch size = 32)
}
```

### Docker Kaynak Limitleri
```yaml
# docker-compose.override.yml
version: '3.6'
services:
  clearml-apiserver:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
  
  clearml-webserver:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

## Sorun Giderme

### Yaygın Sorunlar ve Çözümleri

#### 1. ClearML Server Bağlantı Hatası
```bash
# Servislerin durumunu kontrol et
docker-compose ps

# Logları kontrol et
docker-compose logs clearml-apiserver

# Port kontrolü
netstat -tulpn | grep :8008
```

#### 2. GPU Tanınmama Sorunu
```bash
# NVIDIA driver kontrolü
nvidia-smi

# CUDA kurulum kontrolü
nvcc --version

# PyTorch CUDA kontrolü
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Disk Alanı Sorunu
```bash
# Docker volume temizliği
docker system prune -a

# Eski model dosyalarını temizle
rm -rf runs/train/exp*/

# Log dosyalarını temizle
docker-compose logs --tail=0 -f > /dev/null
```

#### 4. Bellek Yetersizliği
```python
# Batch size küçült
batch_size = 4  # Varsayılan 16 yerine

# DataLoader workers azalt
workers = 2  # Varsayılan 8 yerine

# Model precision değiştir
model.half()  # FP16 kullan
```

### Log Dosyaları ve Debugging

#### ClearML Logları
```bash
# API Server logları
docker logs clearml-apiserver

# Web Server logları
docker logs clearml-webserver

# File Server logları
docker logs clearml-fileserver
```

#### Python Debug Modu
```python
# Debug modunda çalıştır
import logging
logging.basicConfig(level=logging.DEBUG)

from clearml import Task
task = Task.init(project_name="Debug", task_name="Debug_Run")
```

## Güvenlik ve Backup

### Veri Backup Stratejisi
```bash
# Database backup
docker exec clearml-mongo mongodump --out /backup/mongo-backup

# Model dosyaları backup
tar -czf models_backup.tar.gz runs/

# Configuration backup
cp -r ~/.clearml/ ~/clearml_config_backup/
```

### Güvenlik Önerileri
- ClearML Server'ı sadece local ağda çalıştırın
- Firewall kuralları ile dış erişimi engelleyin
- Düzenli backup alın
- Hassas veri setlerini encrypt edin

## Sonuç

Bu rehber ile ClearML kullanarak YOLOv modellerinizi tamamen local ortamda eğitebilir, dış dünyayla veri paylaşımı yapmadan profesyonel bir ML pipeline kurabilirsiniz. Eğitim süreçlerinizi web UI üzerinden takip edebilir, farklı denemeleri karşılaştırabilir ve en iyi modelleri kaydedebilirsiniz.

### Faydalı Linkler
- [ClearML Documentation](https://clear.ml/docs)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Docker Documentation](https://docs.docker.com)
