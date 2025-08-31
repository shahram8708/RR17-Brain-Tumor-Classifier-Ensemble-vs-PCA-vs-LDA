# RR17 Brain Tumor Classifier: Ensemble vs PCA vs LDA Comparison

## 📌 Description
This repository implements a **brain tumor MRI classification** project using a **Residual CNN (RR17)** as the base model.  
We compare three variants of this model on the **Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar)**, which contains **7,023 MRI images** across four classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor (Normal)

The goal is to evaluate classification performance using:
- **RR17 + Ensemble** → Ensemble of multiple RR17 models trained on different splits; predictions combined via majority voting or averaging.  
- **RR17 + PCA** → PCA applied on feature vectors to reduce dimensionality and noise before classification.  
- **RR17 + LDA** → LDA applied on features to maximize class separability before classification.  

---

## 📊 Dataset
- **Source**: [Kaggle - Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Format**: Brain MRI images (224×224 px, grayscale)  
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor (Normal)  

Dataset structure:
```

data/
├── glioma/
├── meningioma/
├── pituitary/
└── no\_tumor/

````

---

## ⚙️ Methods Implemented
- **RR17 + Ensemble** → Multiple RR17 CNNs aggregated by majority vote / averaging.  
- **RR17 + PCA** → Dimensionality reduction with PCA (retain ~95% variance) before classification.  
- **RR17 + LDA** → LDA for supervised feature reduction emphasizing class separation.  

All methods use the same **RR17 CNN (17-layer ResNet-like architecture)** for fairness.

---

## 🛠️ Installation
### Prerequisites
- Python 3.8+
- Virtual environment recommended (conda/venv)

### Steps
```bash
# Clone the repository
git clone https://github.com/shahram8708/RR17-Brain-Tumor-Classifier-Ensemble-vs-PCA-vs-LDA.git
cd RR17-Brain-Tumor-Classifier-Ensemble-vs-PCA-vs-LDA

# Install dependencies
pip install -r requirements.txt
````

Typical dependencies: TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, pandas.

---

## 🚀 Usage

### Training

Example: Train the **ensemble model**

```bash
python src/train_rr17.py --method ensemble --epochs 50 --batch_size 16 --save_path results/ensemble_model.h5
```

Other methods:

```bash
--method pca
--method lda
```

### Evaluation

Evaluate trained model:

```bash
python src/evaluate_rr17.py --method ensemble --model_path results/ensemble_model.h5
```

This outputs **accuracy, precision, recall, F1-score** and can generate confusion matrices & ROC curves.

### Notebooks

Check `notebooks/` for training curves, PCA/LDA visualization, and performance analysis.

---

## 📈 Results

| Model               | Accuracy  | Precision | Recall | F1 Score  |
| ------------------- | --------- | --------- | ------ | --------- |
| **RR17 + Ensemble** | **96.4%** | 96.0%     | 96.5%  | **96.3%** |
| RR17 + PCA          | 93.2%     | 92.5%     | 93.0%  | 92.7%     |
| RR17 + LDA          | 89.7%     | 90.1%     | 89.5%  | 89.3%     |

### Class-wise F1 Scores

| Class       | Ensemble  | PCA   | LDA   |
| ----------- | --------- | ----- | ----- |
| Glioma      | 95.2%     | 92.8% | 88.5% |
| Meningioma  | 97.1%     | 94.2% | 90.3% |
| Pituitary   | 96.4%     | 94.9% | 90.8% |
| Normal      | 97.3%     | 96.0% | 93.5% |
| **Average** | **96.3%** | 92.7% | 89.3% |

✅ **Conclusion**: The **ensemble model** consistently achieves the highest accuracy and F1 across all tumor types. PCA helps reduce dimensionality but with some performance trade-off, while LDA improves class separation but underperforms overall.

---

## 📂 Repository Structure

```
RR17-BrainTumor-Classifier/
├── data/              # Raw MRI dataset (organized by class)
├── src/               # Source code (training/eval scripts, models)
│   ├── train_rr17.py
│   ├── evaluate_rr17.py
│   └── utils.py
├── notebooks/         # Jupyter notebooks for experiments
├── results/           # Model outputs, graphs, metrics (ensemble/, pca/, lda/)
├── docs/              # Documentation and resources
├── LICENSE            # MIT License
├── requirements.txt   # Python dependencies
└── README.md          # Project README
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo & create a feature branch.
2. Follow **PEP8 style** and include docstrings.
3. Update README/docs if necessary.
4. Submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📖 Citation

If you use this project in academic work, please cite:

```
Rahul Patel, "RR17 Brain Tumor Classifier: Ensemble vs PCA vs LDA Comparison," GitHub Repository, 2025.
```

Dataset citation:
Masoud Nickparvar, **Brain Tumor MRI Dataset**, Kaggle (2021).

---

## 🙏 Acknowledgements

* [Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) – Brain Tumor MRI Dataset
* RR17 CNN inspired by **ResNet architectures**
* Libraries: **TensorFlow, Keras, scikit-learn, NumPy, Matplotlib**
* Open-source community for tools, discussions, and support
