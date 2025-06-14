
# YOLO Model Upgrade & Benchmarking Guide
## Car and Car Parts Detection on Raspberry Pi Drones & Android Tablets

### Executive Summary

This guide provides a comprehensive roadmap for upgrading from YOLOv5 to YOLOv8/v11, benchmarking performance across ONNX and NCNN formats, and deploying car detection models on edge devices.

**Key Performance Insights:**
- YOLOv11 offers 22% fewer parameters than YOLOv8 with higher accuracy
- NCNN format provides best performance on ARM-based devices (Raspberry Pi)
- YOLOv11n achieves 2.4ms inference time vs YOLOv8n's 4.1ms

---

## 1. Model Architecture Comparison

### YOLOv5 → YOLOv8 → YOLOv11 Evolution

| Model | Parameters | mAP@0.5:0.95 | Inference Speed | Key Improvements | Advantages/Disadvantages |
|-------|------------|--------------|-----------------|------------------|-------------------------|
| **YOLOv5n** | 1.9M | 45.7% | ~15ms | Baseline anchor-based detection | **+** Accuracy (mAP) is usually high<br>**-** Computation is heavier, inference time is longer. Tuning is complex |
| **YOLOv8n** | 3.2M | 37.3% | 4.1ms | Anchor-free, improved backbone | **+** Anchor-free design, faster inference<br>**-** Lower accuracy than YOLOv5, more parameters |
| **YOLOv11n** | 2.6M | 39.5% | 2.4ms | Optimized architecture, fewer params | **+** Fastest inference, best parameter efficiency<br>**-** Still lower accuracy than YOLOv5, very new model |




### Architecture Benefits

**YOLOv8 Improvements:**
- Anchor-free detection head
- New backbone and neck architecture
- Improved loss function
- Enhanced data augmentation

**YOLOv11 Advantages:**
- 22% fewer parameters than YOLOv8 with higher accuracy
- Enhanced C3k2 and C2PSA modules
- Improved spatial channel reconstruction
- Better feature fusion mechanisms

---

## 2. Format Comparison & Benchmarking

### Deployment Format Analysis

| Format | Use Case | Raspberry Pi Speed | Android Speed | Model Size | Pros | Cons |
|--------|----------|-------------------|---------------|------------|------|------|
| **PyTorch** | Training/Development | Slow | N/A | Large | Full features | CPU intensive |
| **ONNX** | Cross-platform | Medium | Fast | Medium | Universal | Some ops unsupported |
| **NCNN** | ARM devices | **Fastest** | **Fastest** | **Smallest** | ARM optimized | Limited to ARM |
| **TensorRT** | NVIDIA GPUs | N/A | Very Fast | Medium | GPU accelerated | NVIDIA only |

### Raspberry Pi Performance Benchmarks

Based on recent benchmarking data:

**YOLOv11n Performance:**
- NCNN: 2.4ms inference (417 FPS theoretical)
- ONNX: ~8-12ms inference 
- PyTorch: ~25-40ms inference

**YOLOv8n Performance:**
- NCNN: 4.1ms inference (244 FPS theoretical)
- ONNX: ~15-20ms inference
- PyTorch: ~40-60ms inference

---

## 3. Step-by-Step Implementation Guide

### Phase 1: Environment Setup

#### 3.1 Install Dependencies

```bash
# Install YOLOv8/v11 framework
pip install ultralytics

# Install export dependencies
pip install onnx onnxsim
pip install ncnn-python

# Install benchmarking tools
pip install pandas matplotlib seaborn
pip install psutil gpustat
```

#### 3.2 Verify Installation

```python
from ultralytics import YOLO
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Load a model to verify installation
model = YOLO('yolov8n.pt')
print("Installation successful!")
```

### Phase 2: Model Conversion & Export

#### 3.3 Export Models to Different Formats

```python
from ultralytics import YOLO

# Load models
yolo5_model = YOLO('yolov5s.pt')  # Your existing model
yolo8_model = YOLO('yolov8n.pt')  # New model
yolo11_model = YOLO('yolo11n.pt') # Latest model

models = {
    'yolov5s': yolo5_model,
    'yolov8n': yolo8_model,
    'yolo11n': yolo11_model
}

# Export to multiple formats
for model_name, model in models.items():
    print(f"Exporting {model_name}...")
    
    # Export to ONNX
    model.export(format='onnx', optimize=True, simplify=True)
    
    # Export to NCNN
    model.export(format='ncnn', half=True)
    
    # Export to TensorRT (if available)
    try:
        model.export(format='engine', half=True)
    except:
        print(f"TensorRT export failed for {model_name}")
```

### Phase 3: Custom Training Setup

#### 3.4 Prepare Dataset for Car Detection (example)

```python
# Dataset structure for car detection
"""
car_detection_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
"""

# data.yaml content
data_config = """
train: ./train/images
val: ./val/images
nc: 10  # number of classes
names: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 
        'wheel', 'headlight', 'windshield', 'door', 'bumper']
"""
```

#### 3.5 Training Configuration

```python
from ultralytics import YOLO

# YOLOv8 Training
def train_yolov8():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='car_detection_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        workers=8,
        project='car_detection',
        name='yolov8n_car_detection'
    )
    return model

# YOLOv11 Training
def train_yolov11():
    model = YOLO('yolo11n.pt')
    results = model.train(
        data='car_detection_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project='car_detection',
        name='yolo11n_car_detection'
    )
    return model
```

### Phase 4: Benchmarking Framework

#### 3.6 Comprehensive Benchmarking Script

```python
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class YOLOBenchmark:
    def __init__(self):
        self.results = []
        
    def benchmark_model(self, model_path, format_type, test_images, device='cpu'):
        """Benchmark a single model configuration"""
        model = YOLO(model_path)
        
        # Warmup runs
        for _ in range(5):
            model.predict(test_images[0], device=device, verbose=False)
        
        # Actual benchmarking
        times = []
        memory_usage = []
        
        for img_path in test_images:
            # Monitor memory before inference
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Time inference
            start_time = time.time()
            results = model.predict(img_path, device=device, verbose=False)
            end_time = time.time()
            
            # Monitor memory after inference
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        avg_memory = np.mean(memory_usage)
        
        # Model size
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        
        result = {
            'model': Path(model_path).stem,
            'format': format_type,
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'model_size_mb': model_size,
            'avg_memory_usage_mb': avg_memory,
            'device': device
        }
        
        self.results.append(result)
        return result
    
    def run_full_benchmark(self, models_config, test_images):
        """Run complete benchmark suite"""
        for config in models_config:
            print(f"Benchmarking {config['name']} - {config['format']}")
            try:
                result = self.benchmark_model(
                    config['path'], 
                    config['format'], 
                    test_images,
                    config.get('device', 'cpu')
                )
                print(f"  Avg time: {result['avg_inference_time_ms']:.2f}ms")
                print(f"  FPS: {result['fps']:.2f}")
                print(f"  Model size: {result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"  Error: {e}")
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inference time comparison
        df.pivot(index='model', columns='format', values='avg_inference_time_ms').plot(
            kind='bar', ax=axes[0,0], title='Average Inference Time (ms)'
        )
        axes[0,0].set_ylabel('Time (ms)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # FPS comparison
        df.pivot(index='model', columns='format', values='fps').plot(
            kind='bar', ax=axes[0,1], title='Frames Per Second'
        )
        axes[0,1].set_ylabel('FPS')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Model size comparison
        df.pivot(index='model', columns='format', values='model_size_mb').plot(
            kind='bar', ax=axes[1,0], title='Model Size (MB)'
        )
        axes[1,0].set_ylabel('Size (MB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        df.pivot(index='model', columns='format', values='avg_memory_usage_mb').plot(
            kind='bar', ax=axes[1,1], title='Memory Usage (MB)'
        )
        axes[1,1].set_ylabel('Memory (MB)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('yolo_benchmark_results.png', dpi=300, bbox_inches='tight')
        
        return df

# Usage example
def run_benchmark():
    benchmark = YOLOBenchmark()
    
    # Define models to benchmark
    models_config = [
        {'name': 'yolov5s', 'format': 'pytorch', 'path': 'yolov5s.pt'},
        {'name': 'yolov8n', 'format': 'pytorch', 'path': 'yolov8n.pt'},
        {'name': 'yolo11n', 'format': 'pytorch', 'path': 'yolo11n.pt'},
        {'name': 'yolov8n', 'format': 'onnx', 'path': 'yolov8n.onnx'},
        {'name': 'yolo11n', 'format': 'onnx', 'path': 'yolo11n.onnx'},
        {'name': 'yolov8n', 'format': 'ncnn', 'path': 'yolov8n_ncnn_model'},
        {'name': 'yolo11n', 'format': 'ncnn', 'path': 'yolo11n_ncnn_model'},
    ]
    
    # Test images (replace with your test set)
    test_images = ['test_car1.jpg', 'test_car2.jpg', 'test_car3.jpg']
    
    benchmark.run_full_benchmark(models_config, test_images)
    results_df = benchmark.generate_report()
    
    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmark completed! Results saved to benchmark_results.csv")
    
    return results_df
```

### Phase 5: Raspberry Pi Deployment

#### 3.7 Raspberry Pi Setup Script

```bash
#!/bin/bash
# Raspberry Pi deployment script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip3 install ultralytics
pip3 install opencv-python
pip3 install numpy

# Install NCNN for optimal performance
cd /tmp
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
make -j4
sudo make install

# Copy optimized models
mkdir -p ~/car_detection_models
# Copy your NCNN models here
```

#### 3.8 Raspberry Pi Inference Script

```python
import cv2
import time
from ultralytics import YOLO
import numpy as np

class RaspberryPiDetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect_cars(self, frame):
        """Perform car detection on a frame"""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            device='cpu',
            verbose=False
        )
        
        return results[0] if results else None
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes on frame"""
        if results is None:
            return frame
            
        boxes = results.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"Car {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def run_camera_detection(self, camera_index=0):
        """Run real-time detection from camera"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.detect_cars(frame)
            
            # Draw results
            frame = self.draw_detections(frame, results)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                print(f"FPS: {fps:.2f}")
            
            # Display frame
            cv2.imshow('Car Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Use NCNN model for best performance on Raspberry Pi
    detector = RaspberryPiDetector('yolo11n_ncnn_model')
    detector.run_camera_detection()
```

### Phase 6: Android Integration

#### 3.9 Android App Integration Points

```python
# Android-specific considerations for tablet deployment

class AndroidOptimizedDetector:
    def __init__(self, model_path, input_size=320):
        """
        Optimized for Android tablets
        input_size: Smaller size for better performance (320, 416, 640)
        """
        self.model = YOLO(model_path)
        self.input_size = input_size
        
    def prepare_for_mobile(self):
        """Export model optimized for mobile"""
        # Export to ONNX with mobile optimizations
        self.model.export(
            format='onnx',
            imgsz=self.input_size,
            optimize=True,
            simplify=True,
            dynamic=False
        )
        
        # Export to TensorFlow Lite if needed
        try:
            self.model.export(format='tflite', int8=True)
        except Exception as e:
            print(f"TFLite export failed: {e}")
```

---

## 4. Performance Optimization Tips

### 4.1 Model Selection Guidelines

**For Raspberry Pi Drones:**
- **Recommended: YOLOv11n + NCNN format**
- Input resolution: 320x320 or 416x416
- Expected performance: 15-25 FPS
- Model size: ~5-8MB

**For Android Tablets:**
- **Recommended: YOLOv11s + ONNX format**
- Input resolution: 640x640
- Expected performance: 30-60 FPS
- GPU acceleration available

### 4.2 Hardware-Specific Optimizations

**Raspberry Pi 4/5:**
```python
# Optimal settings for RPi
model_config = {
    'conf': 0.4,          # Lower confidence for faster inference
    'iou': 0.4,           # Lower IoU threshold
    'max_det': 50,        # Limit detections
    'imgsz': 320,         # Smaller input size
    'half': True,         # Use FP16 precision
    'device': 'cpu',      # Force CPU inference
}
```

### 4.3 Real-time Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.fps_history = []
        
    def log_inference(self, inference_time):
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
    
    def get_average_fps(self):
        if not self.inference_times:
            return 0
        avg_time = np.mean(self.inference_times)
        return 1000 / avg_time if avg_time > 0 else 0
    
    def print_stats(self):
        if self.inference_times:
            print(f"Avg inference time: {np.mean(self.inference_times):.2f}ms")
            print(f"Current FPS: {self.get_average_fps():.2f}")
```

---

## 5. Expected Performance Results

### 5.1 Benchmark Summary

Based on current benchmarking data and optimizations:

| Platform | Model | Format | Resolution | FPS | Accuracy | Use Case |
|----------|-------|--------|------------|-----|-----------|----------|
| RPi4 | YOLOv11n | NCNN | 320x320 | 20-25 | Good | Drone real-time |
| RPi4 | YOLOv8n | NCNN | 320x320 | 15-20 | Good | Drone basic |
| RPi5 | YOLOv11n | NCNN | 416x416 | 25-30 | Very Good | Drone enhanced |
| Android | YOLOv11s | ONNX | 640x640 | 45-60 | Excellent | Tablet monitoring |
| Android | YOLOv8s | ONNX | 640x640 | 35-45 | Excellent | Tablet standard |

### 5.2 Deployment Recommendations

**Phase 1: Start with YOLOv11n + NCNN**
- Fastest deployment path
- Best performance on edge devices
- Proven stability

**Phase 2: Scale with YOLOv11s for tablets**
- Higher accuracy when processing power allows
- Better for detailed analysis

**Phase 3: Custom training**
- Fine-tune on your specific car detection dataset
- Optimize for your specific use cases

---

## 6. Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: NCNN Export Fails**
```bash
# Solution: Update dependencies
pip install --upgrade ultralytics
pip install onnx>=1.12.0
```

**Issue 2: Raspberry Pi Performance Issues**
```python
# Solution: Optimize settings
import threading
import queue

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue()
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not self.q.empty():
                self.q.get()
            self.q.put(frame)
```

**Issue 3: Memory Issues on Raspberry Pi**
```python
# Solution: Memory management
import gc

def manage_memory():
    gc.collect()  # Force garbage collection
    
# Call periodically during inference
```

---

## 7. Next Steps

1. **Immediate Actions:**
   - Set up development environment
   - Export existing YOLOv5 models to baseline
   - Download and test YOLOv8n and YOLOv11n

2. **Short-term Goals:**
   - Complete benchmark comparisons
   - Optimize for Raspberry Pi deployment
   - Test Android tablet integration

3. **Long-term Optimization:**
   - Custom dataset training
   - Hardware-specific model optimization
   - Performance monitoring and analytics
