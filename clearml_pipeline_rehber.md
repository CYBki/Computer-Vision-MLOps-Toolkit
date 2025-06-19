# ClearML Lokal Pipeline Rehberi

## İçindekiler
1. [ClearML Pipeline'a Giriş](#clearml-pipelinea-giriş)
2. [Kurulum ve Hazırlık](#kurulum-ve-hazırlık)
3. [Pipeline Oluşturma Yöntemleri](#pipeline-oluşturma-yöntemleri)
4. [Lokal Çalıştırma Ayarları](#lokal-çalıştırma-ayarları)
5. [Örnekler](#örnekler)
6. [En İyi Uygulamalar](#en-i̇yi-uygulamalar)

## ClearML Pipeline'a Giriş

ClearML Pipeline, makine öğrenmesi süreçlerini otomatikleştirmek ve birbirine bağlamak için kullanılan güçlü bir araçtır. Pipeline'lar birden fazla süreci düzenlemenin ve birbirine bağlamanın bir yoludur; bir sürecin çıktısını diğerinin girdisi olarak kullanır.

### Pipeline'ın Temel Özellikleri:
- **Modüler Yapı**: Her adım bağımsız bir görev (task) olarak tanımlanır
- **Otomatik Veri Akışı**: Adımlar arası veri transferi otomatik gerçekleşir
- **İzlenebilirlik**: Her adımın durumu ve çıktıları izlenebilir
- **Yeniden Çalıştırılabilirlik**: Pipeline'lar farklı parametrelerle tekrar çalıştırılabilir

## Kurulum ve Hazırlık

### 1. ClearML Kurulumu
```bash
pip install clearml
```

### 2. ClearML Yapılandırması
```bash
clearml-init
```

### 3. Gerekli İthalatlar
```python
from clearml import PipelineController, Task
from clearml.automation.controller import PipelineDecorator
import pandas as pd
import numpy as np
```

## Pipeline Oluşturma Yöntemleri

ClearML'de pipeline oluşturmanın iki ana yöntemi vardır:

### 1. PipelineController Sınıfı Kullanımı

ClearML'de iki ana yöntem vardır. Biri mevcut ClearML görevlerini kolayca zincirleyerek tek bir pipeline oluşturmaktır. Bu, pipeline'daki her adımın daha önce experiment manager kullanarak izlediğiniz bir görev olduğu anlamına gelir.

```python
from clearml import PipelineController

def create_pipeline():
    # Pipeline controller oluştur
    pipe = PipelineController(
        name="ML_Pipeline_Local",
        project="examples",
        version="1.0.0"
    )
    
    # Adımları tanımla
    pipe.add_step(
        name="data_preparation",
        base_task_project="examples",
        base_task_name="data_prep_task",
        parameter_override={'dataset_url': 'local_data.csv'}
    )
    
    pipe.add_step(
        name="model_training",
        base_task_project="examples", 
        base_task_name="train_model_task",
        parents=["data_preparation"]
    )
    
    # Pipeline'ı başlat
    pipe.start()
    
    print("Pipeline tamamlandı!")
```

### 2. PipelineDecorator Kullanımı

Pipeline'ı fonksiyon dekoratörleri ile oluşturma:

```python
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=['processed_data'], cache=True)
def data_preprocessing(raw_data_path: str):
    import pandas as pd
    
    # Veri yükleme
    df = pd.read_csv(raw_data_path)
    
    # Veri işleme
    df_processed = df.dropna()
    df_processed = df_processed.reset_index(drop=True)
    
    # İşlenmiş veriyi kaydet
    output_path = 'processed_data.csv'
    df_processed.to_csv(output_path, index=False)
    
    return output_path

@PipelineDecorator.component(return_values=['model_path'], cache=True)
def model_training(data_path: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Veri yükleme
    df = pd.read_csv(data_path)
    
    # Özellik ve hedef değişkenleri ayır
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Model eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Modeli kaydet
    model_path = 'trained_model.pkl'
    joblib.dump(model, model_path)
    
    return model_path

@PipelineDecorator.pipeline(name="ML_Pipeline_Decorator", project="examples")
def executing_pipeline(raw_data_url: str = 'data.csv'):
    # Pipeline adımlarını sırayla çalıştır
    processed_data = data_preprocessing(raw_data_url)
    model_path = model_training(processed_data)
    
    print(f"Pipeline tamamlandı! Model: {model_path}")
```

## Lokal Çalıştırma Ayarları

### 1. PipelineController ile Lokal Çalıştırma

Pipeline adımlarını da lokal olarak çalıştırmak için run_pipeline_steps_locally=True parametresini geçin.

```python
def run_pipeline_locally():
    pipe = PipelineController(
        name="Local_Pipeline",
        project="examples",
        version="1.0.0"
    )
    
    # Lokal çalıştırma için parametre ekle
    pipe.start_locally(run_pipeline_steps_locally=True)
```

### 2. PipelineDecorator ile Lokal Çalıştırma

```python
if __name__ == '__main__':
    # Pipeline'ı lokal modda çalıştır
    PipelineDecorator.run_locally()
    
    # Pipeline'ı başlat
    executing_pipeline(raw_data_url='local_dataset.csv')
    print('Pipeline başarıyla tamamlandı!')
```

### 3. Karma Mod (Hybrid Mode)

Pipeline mantığı lokal, adımlar uzaktan:

```python
@PipelineDecorator.pipeline(
    name="Hybrid_Pipeline", 
    project="examples",
    pipeline_execution_queue=None  # Pipeline mantığı lokal çalışır
)
def hybrid_pipeline():
    # Pipeline mantığı lokal çalışır
    # Ama her adım uzak worker'larda çalışabilir
    step1_result = data_preprocessing()
    step2_result = model_training(step1_result)
    return step2_result
```

## Örnekler

### Örnek 1: Basit Veri İşleme Pipeline'ı

```python
from clearml import PipelineController, Task
import pandas as pd

# Veri yükleme adımı
def create_data_loading_task():
    task = Task.init(project_name="pipeline_example", task_name="data_loading")
    
    # Sahte veri oluştur
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Veriyi artifact olarak kaydet
    task.upload_artifact('raw_data', artifact_object=data)
    task.close()
    return task.id

# Pipeline oluştur ve çalıştır
def run_data_pipeline():
    # Önce base task'ları oluştur
    data_task_id = create_data_loading_task()
    
    # Pipeline controller oluştur
    pipe = PipelineController(
        name="Data_Processing_Pipeline",
        project="pipeline_example",
        version="1.0.0"
    )
    
    # Veri yükleme adımı
    pipe.add_step(
        name="load_data",
        base_task_id=data_task_id
    )
    
    # Veri işleme adımı (fonksiyon olarak)
    pipe.add_function_step(
        name="process_data",
        function=process_data_function,
        function_kwargs={'input_data': '${load_data.artifacts.raw_data}'},
        function_return=['processed_data'],
        parents=["load_data"]
    )
    
    # Pipeline'ı lokal olarak başlat
    pipe.start_locally(run_pipeline_steps_locally=True)

def process_data_function(input_data):
    # Veri işleme mantığı
    processed = input_data.fillna(0)
    return processed

if __name__ == '__main__':
    run_data_pipeline()
```

### Örnek 2: Makine Öğrenmesi Pipeline'ı

```python
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=['X_train', 'X_test', 'y_train', 'y_test'])
def data_split(dataset_path: str, test_size: float = 0.2):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(dataset_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(return_values=['model'])
def train_model(X_train, y_train, n_estimators: int = 100):
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    return model

@PipelineDecorator.component(return_values=['accuracy'])
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

@PipelineDecorator.pipeline(name="ML_Training_Pipeline", project="ml_example")
def ml_pipeline(dataset_path: str = 'data.csv', test_size: float = 0.2, n_estimators: int = 100):
    # Veri bölme
    X_train, X_test, y_train, y_test = data_split(dataset_path, test_size)
    
    # Model eğitimi
    model = train_model(X_train, y_train, n_estimators)
    
    # Model değerlendirme
    accuracy = evaluate_model(model, X_test, y_test)
    
    print(f"Model doğruluğu: {accuracy}")
    return accuracy

# Pipeline'ı lokal çalıştır
if __name__ == '__main__':
    PipelineDecorator.run_locally()
    
    result = ml_pipeline(
        dataset_path='my_dataset.csv',
        test_size=0.25,
        n_estimators=150
    )
    
    print(f"Pipeline tamamlandı! Final doğruluk: {result}")
```

## En İyi Uygulamalar

### 1. Modüler Tasarım
- Her adımı bağımsız ve yeniden kullanılabilir yapın
- Tek sorumluluk prensibini uygulayın
- Giriş ve çıkış formatlarını net tanımlayın

### 2. Hata Yönetimi
```python
@PipelineDecorator.component(return_values=['result'])
def robust_step(input_data):
    try:
        # Ana işlem
        result = process_data(input_data)
        return result
    except Exception as e:
        print(f"Hata oluştu: {e}")
        # Varsayılan değer döndür veya hata fırlat
        raise
```

### 3. Parametre Yönetimi
```python
@PipelineDecorator.pipeline(name="Parametric_Pipeline", project="examples")
def parametric_pipeline(
    data_path: str = 'default.csv',
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
):
    # Parametreleri pipeline boyunca kullan
    pass
```

### 4. Önbellekleme (Caching)
```python
@PipelineDecorator.component(return_values=['processed_data'], cache=True)
def expensive_preprocessing(data):
    # Ağır işlem - önbelleklenir
    return processed_data
```

### 5. Loglamayı ve İzlemeyi Kullanın
```python
from clearml import Logger

@PipelineDecorator.component(return_values=['model'])
def logged_training(X_train, y_train):
    logger = Logger.current_logger()
    
    # Eğitim sürecini logla
    model = train_model(X_train, y_train)
    
    # Metrikleri logla
    logger.report_scalar("training", "accuracy", accuracy, 0)
    
    return model
```

### 6. Dosya Yönetimi
```python
import tempfile
import os

@PipelineDecorator.component(return_values=['output_path'])
def file_processing_step(input_path: str):
    # Geçici dizin kullan
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'processed_file.csv')
        
        # İşlemi gerçekleştir
        process_file(input_path, output_path)
        
        # Kalıcı konuma taşı
        final_path = 'outputs/processed_file.csv'
        os.makedirs('outputs', exist_ok=True)
        shutil.copy(output_path, final_path)
        
        return final_path
```

## Troubleshooting

### Yaygın Sorunlar ve Çözümleri

1. **Import Hatası**: Global import'ların her adımda tanımlandığından emin olun
2. **Veri Transferi**: Artifact'lar ve parametreler doğru şekilde tanımlanmalı
3. **Bellek Yönetimi**: Büyük veri setleri için disk tabanlı işlem kullanın
4. **Bağımlılık Yönetimi**: Her adımın requirements.txt'i net olmalı

## Sonuç

ClearML Pipeline'ları makine öğrenmesi projelerinizi organize etmek ve otomatikleştirmek için güçlü bir araçtır. Lokal geliştirme ve test için bu rehberdeki örnekleri kullanarak başlayabilir, daha sonra production ortamına geçebilirsiniz.

Pipeline'larınızı modüler, test edilebilir ve yeniden kullanılabilir şekilde tasarlayarak, ML süreçlerinizi daha verimli hale getirebilirsiniz.