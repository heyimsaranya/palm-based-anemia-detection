# Palm-Based Anemia Detection using Deep Learning

## 📌 Overview

This project presents a **non-invasive deep learning approach for anemia detection** using palm images. By leveraging **transfer learning with EfficientNetV2-S**, the model classifies individuals as **Anemic** or **Non-Anemic** based on visual features extracted from palm images.

This solution aims to support **early screening in low-resource settings**, reducing dependency on invasive blood tests.

---

## 🎯 Objectives

* Develop an accurate **image-based anemia classification model**
* Utilize **transfer learning** for improved performance
* Implement **Grad-CAM visualization** for model interpretability
* Enable **non-invasive, accessible screening**

---

## 📂 Dataset

* **Source:** Kaggle Palm Dataset for Anemia Detection
* **Link:** https://www.kaggle.com/datasets/shreyacgosavi/palm-dataset-anemia/data

### 📊 Dataset Details

* Total Images: **4260**
* Classes:

  * **Anemic** (Label 0)
  * **Non-Anemic** (Label 1)

### 🔹 4. Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC Curve

---

### 🔹 5. Model Explainability

* **Grad-CAM** used to visualize important regions influencing predictions

---

## 📈 Results

| Metric              | Value      |
| ------------------- | ---------- |
| Validation Accuracy | **98.44%** |
| Test Accuracy       | **98.75%** |
| ROC-AUC             | **0.9991** |

### 🔍 Observations

* High classification performance with minimal overfitting
* Strong separability between classes
* Grad-CAM shows focus on **palm coloration patterns**

---
