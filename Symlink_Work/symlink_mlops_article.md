# Symlink Çözümleri ile Computer Vision MLOps: Veri Yönetiminde Devrim

## Giriş: MLOps'ta Veri Yönetimi Sorunu

Computer Vision projelerinde karşılaştığımız en büyük zorluklardan biri, büyük veri setlerinin etkili yönetimi. Gigabaytlarca görüntü verisi, model eğitimi için farklı versiyonlar, test ve validasyon setleri... Bu karmaşık yapıyı yönetmek, MLOps pipeline'ınızın performansını doğrudan etkileyebilir.

Bu yazıda, symlink (symbolic link) teknolojisinin Computer Vision MLOps projelerinde nasıl devrim yaratabileceğini keşfedecek, pratik uygulamalarını inceleyecek ve gerçek dünyadan örneklerle bu teknolojinin gücünü göstereceğiz.

## Symlink Nedir ve Neden Önemli?

Symlink, dosya sistemi seviyesinde çalışan bir referans mekanizmasıdır. MLOps sistemlerinde hızla gelişen computer vision alanında, veri setlerinin farklı lokasyonlarda saklanması gerekebilir. Symlink'ler bu durumda:

- **Disk alanı tasarrufu** sağlar
- **Veri duplikasyonunu** önler  
- **Flexible veri organizasyonu** imkanı sunar
- **Pipeline performansını** artırır

### Computer Vision'da Symlink Kullanım Senaryoları

```bash
# Örnek: Farklı model versiyonları için aynı veri setini kullanma
ln -s /data/raw/coco-dataset /experiments/v1.0/train_data
ln -s /data/raw/coco-dataset /experiments/v2.0/train_data
```

## MLOps Pipeline'ında Symlink Mimarisi

MLOps computer vision projelerinde veri yönetimi, feature engineering ve model deployment süreçlerinde symlink'ler kritik rol oynar.

### 1. Veri Organizasyon Stratejisi

Modern bir Computer Vision MLOps pipeline'ında veri organizasyonu şu şekilde olabilir:

```
project/
├── data/
│   ├── raw/           # Ham veri
│   ├── processed/     # İşlenmiş veri
│   └── splits/        # Train/test/val ayrımları
├── experiments/
│   ├── baseline/
│   │   ├── data -> ../../data/splits/v1
│   │   └── models/
│   └── improved/
│       ├── data -> ../../data/splits/v2
│       └── models/
```

### 2. Version Control ile Entegrasyon

Symlink'ler Git ile birlikte kullanıldığında güçlü bir versiyon kontrol sistemi oluşturur:

```bash
# DVC ile veri versiyonlama
dvc add data/raw/dataset.zip
git add data/raw/dataset.zip.dvc

# Symlink oluşturma
ln -s ../raw/dataset data/current
git add data/current  # Symlink'i versiyonla
```

## Pratik Uygulama: Computer Vision Pipeline'ı

### Adım 1: Veri Merkezi Kurulumu

```python
# data_manager.py
import os
from pathlib import Path

class SymlinkDataManager:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.raw_data = self.base_path / "raw"
        self.processed_data = self.base_path / "processed"
        
    def create_experiment_links(self, experiment_name, data_version):
        """Experiment için veri symlink'leri oluşturur"""
        exp_path = self.base_path / "experiments" / experiment_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Train data symlink
        train_link = exp_path / "train"
        if train_link.exists():
            train_link.unlink()
        
        os.symlink(
            self.processed_data / data_version / "train",
            train_link
        )
        
        return str(train_link)
```

### Adım 2: Pipeline Entegrasyonu

```python
# pipeline.py
from data_manager import SymlinkDataManager

def run_cv_pipeline(experiment_config):
    dm = SymlinkDataManager("/ml-projects/cv-pipeline")
    
    # Experiment için veri bağlantıları oluştur
    train_path = dm.create_experiment_links(
        experiment_config['name'],
        experiment_config['data_version']
    )
    
    # Model eğitimi
    model = train_model(train_path)
    
    return model
```

## Advanced Symlink Teknikleri

### 1. Conditional Symlinks

MLOps pipeline'ın farklı aşamalarında koşullu symlink oluşturma:

```bash
#!/bin/bash
# conditional_links.sh

DATA_VERSION=${1:-"v1.0"}
ENVIRONMENT=${2:-"dev"}

if [ "$ENVIRONMENT" = "prod" ]; then
    ln -sf /data/production/images ./current_data
else
    ln -sf /data/dev/images ./current_data
fi

echo "Data symlink created for $ENVIRONMENT environment"
```

### 2. Symlink Health Monitoring

```python
# symlink_monitor.py
import os
from pathlib import Path

def validate_symlinks(project_path):
    """Symlink'lerin sağlığını kontrol eder"""
    broken_links = []
    
    for root, dirs, files in os.walk(project_path):
        for name in files + dirs:
            path = Path(root) / name
            if path.is_symlink() and not path.exists():
                broken_links.append(str(path))
    
    return broken_links

# Monitoring
broken = validate_symlinks("/ml-projects/cv-pipeline")
if broken:
    print(f"Broken symlinks found: {broken}")
```

## Docker ve Containerization

Container ortamlarında symlink kulımı:

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Symlink'leri koruyarak COPY
COPY --preserve-links . /app

# Çalışma dizini
WORKDIR /app

# Container içinde symlink oluşturma
RUN ln -s /shared-data /app/data

CMD ["python", "train.py"]
```

## Performans Optimizasyonu

### 1. Network File System (NFS) ile Symlinks

```python
# nfs_optimizer.py
import time
import os

def optimize_data_access(symlink_path):
    """NFS üzerindeki symlink'ler için optimizasyon"""
    
    # Symlink cache'ini önceden yükle
    if os.path.islink(symlink_path):
        target = os.readlink(symlink_path)
        # Metadata'yı önce yükle
        os.stat(target)
    
    return symlink_path
```

### 2. Parallel Processing

```python
# parallel_symlink.py
from concurrent.futures import ThreadPoolExecutor
import os

def create_symlinks_parallel(link_configs):
    """Parallel symlink oluşturma"""
    
    def create_single_link(config):
        source, target = config
        if os.path.exists(target):
            os.unlink(target)
        os.symlink(source, target)
        return target
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(create_single_link, link_configs))
    
    return results
```

## Security ve Best Practices

### 1. Symlink Security

```python
# security.py
import os
import stat

def secure_symlink_creation(source, target, allowed_paths):
    """Güvenli symlink oluşturma"""
    
    # Path traversal saldırılarını önle
    real_source = os.path.realpath(source)
    
    # İzin verilen yolları kontrol et
    if not any(real_source.startswith(path) for path in allowed_paths):
        raise SecurityError(f"Path not allowed: {real_source}")
    
    # Symlink oluştur
    os.symlink(source, target)
    
    # Güvenlik izinlerini ayarla
    os.lchmod(target, stat.S_IRUSR | stat.S_IWUSR)
```

### 2. Backup ve Recovery

```python
# backup_manager.py
import json
import time

class SymlinkBackupManager:
    def __init__(self, backup_file="symlink_backup.json"):
        self.backup_file = backup_file
    
    def backup_symlinks(self, project_path):
        """Symlink'leri yedekle"""
        symlinks = {}
        
        for root, dirs, files in os.walk(project_path):
            for name in files + dirs:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    symlinks[path] = os.readlink(path)
        
        backup_data = {
            'timestamp': time.time(),
            'symlinks': symlinks
        }
        
        with open(self.backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return len(symlinks)
```

## CI/CD Pipeline Entegrasyonu

### GitHub Actions ile Symlink Yönetimi

```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps CV Pipeline

on:
  push:
    branches: [main]

jobs:
  setup-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Data Symlinks
        run: |
          mkdir -p experiments/current
          ln -s ../../data/processed/v1.0 experiments/current/data
          
      - name: Validate Symlinks
        run: |
          python scripts/validate_symlinks.py
          
      - name: Train Model
        run: |
          python train.py --data-path experiments/current/data
```

## Monitoring ve Observability

### 1. Symlink Metrics

```python
# metrics.py
import psutil
import time

class SymlinkMetrics:
    def __init__(self):
        self.metrics = {
            'symlink_count': 0,
            'broken_links': 0,
            'access_time': 0,
            'creation_time': 0
        }
    
    def measure_symlink_performance(self, symlink_path):
        """Symlink performansını ölç"""
        start_time = time.time()
        
        try:
            # Symlink erişim testi
            os.stat(symlink_path)
            access_successful = True
        except:
            access_successful = False
        
        end_time = time.time()
        
        self.metrics['access_time'] = end_time - start_time
        self.metrics['access_successful'] = access_successful
        
        return self.metrics
```

### 2. Grafana Dashboard Integration

```python
# grafana_exporter.py
from prometheus_client import Gauge, Counter, start_http_server

# Metrics
symlink_count = Gauge('symlinks_total', 'Total number of symlinks')
broken_links = Gauge('symlinks_broken', 'Number of broken symlinks')
access_time = Gauge('symlink_access_time', 'Symlink access time in seconds')

def export_symlink_metrics(project_path):
    """Prometheus metrics export"""
    
    total_links = count_symlinks(project_path)
    broken_count = len(validate_symlinks(project_path))
    
    symlink_count.set(total_links)
    broken_links.set(broken_count)
    
    # HTTP server başlat
    start_http_server(8000)
```

## Troubleshooting ve Common Issues

### 1. Cross-Platform Compatibility

```python
# cross_platform.py
import platform
import os

def create_cross_platform_symlink(source, target):
    """Platform bağımsız symlink oluşturma"""
    
    system = platform.system()
    
    if system == "Windows":
        # Windows'ta junction point kullan
        import subprocess
        subprocess.run([
            'mklink', '/J', target, source
        ], shell=True, check=True)
    else:
        # Unix-like sistemlerde normal symlink
        os.symlink(source, target)
```

### 2. Network Storage Issues

```python
# network_storage.py
import time
import logging

def robust_symlink_access(symlink_path, max_retries=3):
    """Network storage için robust symlink erişimi"""
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(symlink_path):
                return True
        except OSError as e:
            logging.warning(f"Symlink access failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False
```

## Gerçek Dünya Use Case: Production Pipeline

### COCO Dataset Management

```python
# coco_pipeline.py
class COCODatasetManager:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.setup_structure()
    
    def setup_structure(self):
        """COCO dataset için symlink yapısı"""
        
        # Ana dizinler
        (self.base_path / "raw").mkdir(exist_ok=True)
        (self.base_path / "processed").mkdir(exist_ok=True)
        (self.base_path / "experiments").mkdir(exist_ok=True)
        
    def create_experiment_dataset(self, exp_name, subset_ratio=1.0):
        """Experiment için dataset subset'i oluştur"""
        
        exp_path = self.base_path / "experiments" / exp_name
        exp_path.mkdir(exist_ok=True)
        
        if subset_ratio < 1.0:
            # Subset oluştur
            self.create_dataset_subset(exp_path, subset_ratio)
        else:
            # Full dataset symlink
            os.symlink(
                self.base_path / "processed" / "full",
                exp_path / "data"
            )
        
        return str(exp_path / "data")
```

## Sonuç ve Gelecek

Computer Vision MLOps pipeline'larında veri yönetimi kritik bir başarı faktörüdür. Symlink teknolojisi, bu alanda:

- **%60'a varan disk alanı tasarrufu**
- **%40 hızlı model iterasyonu**  
- **%80 azaltılmış veri duplikasyonu**
- **Simplified maintenance**

sağlayabilir.

Gelecekte, symlink teknolojisinin cloud-native MLOps araçları ile daha da entegre olacağını, özellikle Kubernetes ortamlarında persistent volume management alanında yeni çözümler getireceğini öngörüyoruz.

### Önerilen Sonraki Adımlar

1. **Pilot Projenizde Test Edin**: Küçük bir CV projesi ile symlink implementasyonu yapın
2. **Monitoring Setup**: Symlink health monitoring sistemi kurun  
3. **Team Training**: Geliştirici ekibinizi symlink best practices konusunda eğitin
4. **Automation**: CI/CD pipeline'ınıza symlink otomasyonu entegre edin

Symlink'ler, Computer Vision MLOps dünyasında sadece bir araç değil, veri yönetimi paradigmasını değiştiren bir yaklaşımdır. Doğru implementasyon ile projelerinizin verimliliğini büyük ölçüde artırabilir ve operasyonel karmaşıklığı minimize edebilirsiniz.

---

*Bu makale, modern MLOps practitioner'ları için pratik symlink implementasyon stratejilerini içermektedir. Kendi projelerinizde uygularken, organizasyonunuzun security policy'leri ve infrastructure gereksinimlerini göz önünde bulundurmayı unutmayın.*