# 🐾 Animal Classifier (100-Class Deep Learning App)

This project is a **deep learning-based animal image classifier** built using **PyTorch and EfficientNet-B0**.  
It classifies images into **100 different animal categories** and provides prediction probabilities along with informative bios for each animal.

This version is fully compatible with environments where **TensorFlow is not available**.

---

## 📥 1) Download the Dataset

Download the dataset here:

https://download.scidb.cn/download?fileId=294fff52fb06d5656cfe26175cf42796&path=/V1/Animal.zip&fileName=Animal.zip

After downloading:
- Extract the `.zip` file
- Place the `Animal` folder into:

data/raw/

Final structure should look like:

data/raw/
  antelope/
  badger/
  bat/
  ...
  zebra/

---

## ⚙️ 2) Install Dependencies

```bash
python -m pip install -r requirements.txt

```

## 🔀 3) Split the Dataset

This will automatically create:

- data/train/
- data/val/
- data/test/

```bash
python training/split_dataset.py --config configs/default.yaml
```

## 🧠 4) Train the Model

```bash
python training/train.py --config configs/default.yaml
```

Training Details:

  Model: EfficientNet-B0 (transfer learning)
    Batch size: 16
    Epochs: 18 total (8 + 10)
    Training stages:
    Stage 1: frozen backbone
    Stage 2: fine-tuning
    Trials: configurable (set to 1 for faster training on CPU)

The best model is saved to:

models/best_model.pt

## 📊 5) Generate Graphs (Recommended)

```bash
python training/plot_results.py
```
This generates:

  models/class_distribution.png
  models/loss_curve.png
  models/accuracy_curve.png
  models/model_performance.png

## 🖥️ 6) Launch the Web App

```bash
python app/app.py
```

Features:
  Upload an image
  Get predicted animal + probabilities
  View detailed animal bios
  Clean, user-friendly interface
  
📈 Example Results
  
  Typical performance:
    
    Validation Accuracy: ~91–92%
    Test Accuracy: ~91–92%
    Target Accuracy: 85% (achieved)
🧠 Key Techniques Used

  Transfer Learning (EfficientNet-B0)
  Data Augmentation:
  horizontal flip
  rotation
  color jitter
  Two-stage training (freeze → fine-tune)
  Multi-class classification (100 classes)
  Automatic dataset splitting
  Model selection via trials
  
⚠️ Notes
  Training on CPU can take several hours
  For faster runs:
  set trials: 1
  reduce epochs
  Accuracy depends on:
  dataset quality
  class balance
  image diversity

🧾 Project Summary

This project implements a deep learning-based image classification system capable of identifying 100 different animal species. Using transfer learning with EfficientNet-B0 and a two-stage training approach, the model achieves over 90% accuracy. The system includes a user-friendly interface where users can upload images and receive predictions along with educational animal descriptions.

🚀 Future Improvements

  Add confusion matrix visualization
  Deploy as a web app (Flask / Streamlit)
  Use GPU for faster training
  Extend to video or real-time classification
