# A Privacy-Preserving Federated Learning Framework for Multi-Class Bengali Fake News Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This repository contains the code, results, and figures for our paper on **federated learning for multi-class Bengali fake news detection**. We compare three approaches:

1. **Centralized BanglaBERT** (baseline)
2. **Federated FedAvg**
3. **Federated FedProx** (proposed)

on the **BanFakeNews-2.0** dataset under realistic **Non-IID** data distribution (Dirichlet α=1.0, 5 clients).

## 📊 Results

| Method | Accuracy | F1 (macro) | F1 (weighted) |
|---|---|---|---|
| Centralized BanglaBERT | 0.7645 | 0.4923 | 0.7828 |
| Federated FedAvg | 0.7766 | 0.4749 | 0.7877 |
| **Federated FedProx** 🏆 | **0.8095** | **0.4990** | **0.8103** |

**Key Findings:**
- **FedProx outperforms Centralized BanglaBERT** by **+1.36%** macro F1
- **FedProx outperforms FedAvg** by **+5.06%** macro F1
- **Privacy-preserving federated learning can match or exceed centralized performance** — FedProx achieves **101.36%** of centralized macro F1 while preserving privacy by design

## 📁 Repository Structure

```
.
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── paper_draft.md                # Full paper manuscript
├── fake-news.ipynb               # Complete implementation notebook
├── figures/                      # Publication figures
│   ├── fig1_training_curves.png
│   ├── fig2_method_comparison.png
│   ├── federated_confusion_matrix.png
│   └── federated_training_curves.png
└── results/                      # Experimental results
    ├── table1_latex.tex
    ├── final_comparison_3way.json
    ├── federated_fedavg_history.json
    ├── federated_fedprox_history.json
    ├── federated_test_results.json
    ├── federated_fedprox_test_results.json
    └── comparison_table.json
```

## 📚 Dataset

**BanFakeNews-2.0** (60,544 Bengali news articles, 4 classes)

- **Classes:** Fake (0), True (1), Mostly True (2), Mostly Fake (3)
- **Train:** 42,380 (raw) → 12,000 (balanced)
- **Validation:** 9,082
- **Test:** 9,082

**Source:** [Kaggle](https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news)

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- scikit-learn 1.3+

## 🚀 Usage

**Run the complete pipeline:**

```bash
jupyter notebook fake-news.ipynb
```

**The notebook contains:**
1. Data loading and preprocessing
2. Class balancing (3,000 samples per class)
3. Federated client partitioning (Non-IID, Dirichlet α=1.0)
4. Centralized BanglaBERT training
5. Federated FedAvg training
6. Federated FedProx training
7. Test evaluation and figures

## 🔬 Methodology

- **Base Model:** [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) (ELECTRA-based, 110M params)
- **Trainable Layers:** Top 4 encoder layers (28.35M params)
- **Federated Setup:** 5 clients, 10 rounds, 2 local epochs per round
- **FedProx:** μ = 0.01
- **Optimizer:** AdamW (BERT lr=2e-5, head lr=1e-3)
- **Batch Size:** 16

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{hossain2025federated,
  title={A Privacy-Preserving Federated Learning Framework for Multi-Class Bengali Fake News Detection on Resource-Constrained Devices},
  author={Hossain, Md. Murad},
  journal={arXiv preprint},
  year={2025}
}
```

## 🔗 Pretrained Models

Models will be available on HuggingFace Hub *(coming soon)*.

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Md. Murad Hossain**  
Department of Computer Science and Engineering  
Uttara University, Dhaka, Bangladesh  
📧 Email: 2231081044@uttarauniversity.edu.bd

## 🙏 Acknowledgments

- BanFakeNews-2.0 dataset contributors
- BanglaBERT team (csebuetnlp)
- Kaggle for GPU compute
