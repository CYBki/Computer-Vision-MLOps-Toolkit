# ClearML ile YOLO Dataset Versiyonlama ve Eğitim Rehberi

Bu rehber, ClearML kullanarak YOLO modellerini eğitirken dataset versiyonlama ve otomatik takip sistemini nasıl kuracağınızı gösterir.

## 1. ClearML Data - Built-in Dataset Versiyonlama

Dataset versiyonlama için ClearML'in yerleşik özelliklerini kullanın:

```python
from clearml import Dataset

# Dataset oluştur/güncelle
dataset = Dataset.create(
    dataset_name="yolo_training_data",
    dataset_project="YOLO_Datasets"
)

# Dataset'e dosya ekle
dataset.add_files(path="/path/to/your/dataset")

# Dataset'i finalize et (artık değiştirilemez)
dataset.finalize()

# Dataset ID'sini al (bu ID modelle birlikte kaydedilecek)
dataset_id = dataset.id
print(f"Dataset ID: {dataset_id}")
```

## 2. YOLO Eğitimi - Otomatik ClearML Integration

ClearML kurulumu yeterli - kod değişikliği gerekmez!

```bash
pip install clearml
```

```python
from ultralytics import YOLO
from clearml import Task

# Task otomatik başlatılır (YOLO built-in entegrasyonu)
model = YOLO('yolov8n.pt')

# Eğitim - ClearML otomatik olarak her şeyi takip eder
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 3. Dataset'i Task ile Bağlama - Built-in

```python
from clearml import Task, Dataset

# Task başlat
task = Task.init(project_name="YOLO_Training", task_name="auto_experiment")

# Dataset'i task'a bağla (otomatik versiyon takibi)
dataset = Dataset.get(dataset_id="YOUR_DATASET_ID")
dataset_path = dataset.get_local_copy()  # Otomatik download

# Task'a dataset bilgilerini kaydet
task.connect_configuration({
    'dataset_id': dataset.id,
    'dataset_name': dataset.name,
    'dataset_version': dataset.version
})

# Normal YOLO eğitimi
model = YOLO('yolov8n.pt')
results = model.train(data=f"{dataset_path}/data.yaml", epochs=100)
```

## 4. Otomatik Dataset Sync - Built-in

### Terminal'den tek komut ile dataset sync:

```bash
clearml-data sync --project "YOLO_Datasets" --name "training_data" --folder "/path/to/dataset"
```

### Python'dan:

```python
from clearml import Dataset

# Mevcut dataset'i güncelle
dataset = Dataset.get(dataset_name="yolo_training_data", dataset_project="YOLO_Datasets")
dataset = dataset.create_child_dataset()  # Yeni versiyon oluştur
dataset.sync_folder("/path/to/updated/dataset")  # Otomatik sync
dataset.finalize()
```

## 5. Model-Dataset İlişkisi - Built-in Tracking

```python
from clearml import Task

# Her task otomatik olarak şunları kaydeder:
# - Kullanılan dataset ID'si
# - Model dosyaları
# - Hiperparametreler
# - Git commit hash
# - Çalışma ortamı

task = Task.init(project_name="YOLO_Training")

# Model eğitiminde dataset referansı otomatik kaydedilir
model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=100)

# Task tamamlandığında Web UI'da görünecek:
# - Dataset versiyonu
# - Model performansı
# - Karşılaştırma grafikleri
```

## 6. ClearML Pipelines - Otomatik Workflow

```python
from clearml import PipelineController

@PipelineController.function_decorator(
    return_values=['dataset_id'],
    packages=['clearml']
)
def create_dataset_version(dataset_path: str, version_name: str):
    """Dataset versiyonu oluştur"""
    from clearml import Dataset
    
    dataset = Dataset.create(
        dataset_name="yolo_data",
        dataset_project="YOLO_Datasets",
        dataset_version=version_name
    )
    dataset.add_files(path=dataset_path)
    dataset.finalize()
    return dataset.id

@PipelineController.function_decorator(
    return_values=['model_id'],
    packages=['ultralytics', 'clearml']
)
def train_yolo_model(dataset_id: str, model_config: dict):
    """YOLO modeli eğit"""
    from clearml import Dataset, Task
    from ultralytics import YOLO
    
    # Dataset'i indir
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    
    # Task otomatik başlar
    model = YOLO(model_config['base_model'])
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        **model_config['params']
    )
    
    return Task.current_task().id

# Pipeline oluştur
pipeline = PipelineController(
    name="YOLO_Training_Pipeline",
    project="YOLO_AutoML"
)

# Otomatik workflow
pipeline.add_function_step(
    name="dataset_creation",
    function=create_dataset_version,
    function_kwargs={
        'dataset_path': '/path/to/dataset',
        'version_name': 'v1.0'
    }
)

pipeline.add_function_step(
    name="model_training",
    function=train_yolo_model,
    function_kwargs={
        'dataset_id': '${dataset_creation.dataset_id}',
        'model_config': {
            'base_model': 'yolov8n.pt',
            'params': {'epochs': 100, 'imgsz': 640}
        }
    },
    parents=['dataset_creation']
)

# Pipeline'ı çalıştır
pipeline.start()
```

## 7. Web UI'dan Otomatik Görüntüleme

ClearML Web UI'da otomatik olarak görünür:

1. **Experiments** sayfasında tüm eğitimler
2. Her experiment'te dataset ID ve versiyonu
3. Model performans karşılaştırması
4. Dataset değişiklik geçmişi
5. Model-Dataset dependency grafiği

## 8. Otomatik Model Registry

```python
from clearml import Model

# Model otomatik olarak registry'e kaydedilir
# Web UI'dan:
# - Hangi dataset ile eğitildiği
# - Model performansı 
# - Deployment durumu
# görülebilir

# Model'i programatik olarak al
model = Model(model_id="your_model_id")
print(f"Model dataset: {model.data_artifact}")  # Dataset bilgisi
print(f"Model metrics: {model.get_metrics()}")  # Performans
```

## 9. Otomatik Comparison - Built-in

Web UI'da otomatik olarak:

- Aynı dataset ile eğitilen modelleri karşılaştır
- Farklı dataset versiyonlarının etkisini gör
- Hyperparameter etkilerini analiz et

## 10. Tek Komut Setup

Terminal'de sadece:

```bash
pip install clearml
clearml-init
```

Sonra normal YOLO eğitimi yapmaya devam edin - ClearML otomatik olarak her şeyi takip edecek!

## Önemli Notlar

- ClearML, YOLO ile yerleşik entegrasyona sahiptir
- Kod değişikliği minimum düzeydedir
- Dataset versiyonlama otomatik olarak yapılır
- Web UI üzerinden tüm süreç görüntülenebilir
- Model-dataset ilişkisi otomatik olarak kaydedilir