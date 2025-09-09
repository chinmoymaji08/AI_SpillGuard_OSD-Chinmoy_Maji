# 🛢️ AI_SpillGuard_Oil_Spill_Detection – Chinmoy Maji

## 📌 Project Overview  
This project implements an **AI-powered oil spill detection system** using **U-Net** for image segmentation. The goal is to automatically detect and segment oil spill regions from images, supporting environmental monitoring and disaster management.  


## 📊 Dataset  
- **Source:** Kaggle Oil Spill Dataset + additional SAR images.  
- **Classes:** Oil spill vs non-spill.  
- **Preprocessing:**  
  - Resized to 256×256  
  - Normalization  
  - Data augmentation (flip, rotation, brightness, contrast)  

---

## ⚙️ Model Development  
- **Architecture:** U-Net (PyTorch)  
- **Loss:** BCE + Dice Loss  
- **Optimizer:** Adam (lr=1e-4 with step decay)  
- **Training Epochs:** 40  
- **Evaluation Metrics:** Accuracy, mIoU, Dice  

---

## 📈 Results  

**Validation (Best Epoch):**  
- Accuracy: ~0.93  
- mIoU: ~0.82  
- Dice: ~0.88  

**Test Set:**  
- Accuracy: **0.9207**  
- mIoU: **0.7856**  
- Dice: **0.8536**  

### Sample Predictions  
Original → Ground Truth → Model Prediction  
