# 🛢️ AI_SpillGuard_Oil_Spill_Detection – Chinmoy Maji

## 📌 Project Overview
This project implements an oil spill detection system** using **DeepLabV3-ResNet50** for semantic segmentation. 
The system automatically detects and segments oil spill regions from remote sensing images, supporting environmental monitoring and disaster management efforts.

## 📊 Dataset
- **Source:** Kaggle Oil Spill Dataset + additional SAR images
- **Classes:**
  - Oil Spill
  - Non-Spill (background)
- **Preprocessing:**
  - Images resized to 256×256
  - Normalization
  - Data augmentation (horizontal/vertical flip, rotation, brightness, contrast adjustments)

## ⚙️ Model Development
- **Architecture:** DeepLabV3 with ResNet-50 backbone (PyTorch)
- **Loss Function:** Combination of BCE + Dice Loss
- **Optimizer:** Adam (lr=1e-4 with step decay)
- **Training Epochs:** 50
- **Evaluation Metrics:** Accuracy, mIoU, Dice, Precision, Recall, F1 Score

## 📈 Results
**Validation (Best Epoch):**
- Accuracy: ~0.95
- mIoU: ~0.88
- Dice: ~0.91

**Test Set (Final Model):**
- Accuracy: **0.9698**
- mIoU: **0.9030**
- Dice: **0.9354**
- Precision: **0.8228**
- Recall: **0.8313**
- F1 Score: **0.8238**

## 🎯 Key Improvements Over U-Net
- Switched from U-Net to **DeepLabV3-ResNet50**, achieving a **10–12% boost in IoU and Dice** scores.
- Better handling of small and irregular spill regions.
- Balanced precision and recall, minimizing false alarms while capturing most spills.

## 🖼️ Sample Predictions
Original → Ground Truth → Model Prediction
