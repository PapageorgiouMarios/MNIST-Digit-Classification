# 📘 MNIST: Automatic Handwritten Digit Recognition

This project evaluates the performance of 3 different Neural Networks on the **MNIST handwritten digits dataset** using **PyTorch**.  

<img width="474" height="334" alt="image" src="https://github.com/user-attachments/assets/e0651a12-fa7f-46cc-93a7-8196e831607d" />


---

## 🌐 Neural Networks

1)  **MLP (Multi-Layer Perceptron)**
2) **CNN (Simple Convolutional Neural Network)**
3) **MobileNetV2 (pretrained, frozen feature extractor)**

---

## 📂 Project Structure

### **1. Environment Setup**
- Installs and imports:
  - `torch`
  - `torchvision`
  - `torchaudio`
- Detects GPU availability.
- Defines constants:
  - Batch size  
  - Learning rate  
  - Number of epochs  

---

### **2. Dataset Preparation**

Two different transformation pipelines are used:

#### **For MLP & CNN**
- Convert MNIST images to tensors.
- Normalize using MNIST mean & standard deviation.

#### **For MobileNetV2**
- Resize images to **224×224**.
- Convert grayscale images → **3-channel RGB**.
- Normalize using **ImageNet** mean & std.

---

## 🧠 Machine Learning Model Architectures

### 🔹 **1. MLP**

- One **hidden layer**, size = **100**
- Output layer with **10 classes** (digits 0–9)
- ReLU activation
- Flattens 28×28 images into a **784-dimensional** vector

---

### 🔹 **2. CNN**

1. **Conv layer:** 32 channels, kernel = 3×3, stride = 1  
2. **Max-pooling:** 2×2  
3. **ReLU**  
4. **Conv layer:** 64 channels, kernel = 3×3, stride = 1  
5. **Max-pooling:** 2×2  
6. **Dense layer:** input = 64×7×7 = 3136, output = 10  
7. **Softmax**

---

### 🔹 **3. MobileNetV2**

- Pretrained convolutional network on ImageNet (≈ 3.5 million parameters)
- Designed for mobile and embedded devices
- Final classification layer replaced with a **10-class output**
- **Feature extractor frozen**; Only final classifier is trained
- Expects **3-channel RGB images of 224×224**, so MNIST images are resized and converted from grayscale

---

## 🔬 Training & Evaluation

Each model is trained using:

- **Adam optimizer**
- **Cross-entropy loss**
- `train_one_epoch()` for training
- `evaluate()` for validation
- `run_experiment()` to:
  - Repeat experiments  
  - Collect statistics  
  - Produce summary metrics
  - 
---

## 📊 Results Summary

### **MLP**
- **Accuracy:** ~97.44%  
➡️ Very fast and surprisingly strong performance.

---

### **CNN**
- **Accuracy:** ~98.97% (best)  
➡️ Convolutions capture spatial structure efficiently, giving the best accuracy-to-speed ratio.

---

### **MobileNetV2 (Frozen)**
- **Accuracy:** ~95.89%  
➡️ Too slow and underperforms due to unnecessary complexity for MNIST.

---

## 📚 References

1. **MNIST Dataset** — 28×28 grayscale images of handwritten digits (60k train / 10k test)  
2. **PyTorch** — Deep learning framework used for implementation  
3. **ImageNet** — Large-scale dataset for pretraining computer vision models

<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/517b3f30-f813-4939-a2f7-dbaa8340fc85" />
<img width="230" height="100" alt="image" src="https://github.com/user-attachments/assets/a2195d70-2c88-4eb4-815c-7f180cdcf92e" />
