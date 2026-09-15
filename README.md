# Pneumonia Classification (Bacteria / Virus / Normal)

Deep learning project for classifying chest X-ray images into three categories: 
**Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**, using the Kermany chest X-ray dataset.

## Overview

This project applies transfer learning to detect and classify pneumonia from chest X-ray images, 
as part of ongoing preparation for a medical imaging internship at CHU Fes.

## Dataset

- Source: [Kermany Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes: Normal, Bacterial Pneumonia, Viral Pneumonia
- Patient-aware split using `GroupShuffleSplit` to prevent data leakage between train/test sets

## Method

- **Framework:** MONAI (PyTorch-based medical imaging library)
- **Architecture:** ResNet18 (transfer learning, fine-tuned)
- **Interpretability:** Grad-CAM visualizations to highlight regions influencing predictions

## Results

- Test accuracy: **81%** (after fine-tuning)

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle and place it as described in the notebook
4. Run `pneumonia-1.ipynb`

## Notes
Pneumonia remains one of the leading causes of illness and death worldwide, 
particularly among children and the elderly. Improving automated detection tools, even incrementally, can contribute to earlier diagnosis and better outcomes. Contributions are welcome 
if you have ideas or techniques that could improve the model's performance (architecture changes, data augmentation, regularization, hyperparameter tuning, etc.), feel 
free to open an issue or submit a pull request.
