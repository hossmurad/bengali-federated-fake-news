# A Privacy-Preserving Federated Learning Framework
# for Multi-Class Bengali Fake News Detection

## Overview
This repository contains the code and results for our paper on federated
learning for Bengali fake news detection. We compare Centralized BanglaBERT,
Federated FedAvg, and Federated FedProx on the BanFakeNews-2.0 dataset.

## Results

| Method | Accuracy | F1 (macro) |
|---|---|---|
| Centralized BanglaBERT | 0.7645 | 0.4923 |
| Federated FedAvg | 0.7766 | 0.4749 |
| **Federated FedProx** | **0.8095** | **0.4990** |

## Dataset
- **BanFakeNews-2.0** (60,544 Bengali news articles, 4 classes)
- Source: https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. `notebooks/01_data_preparation.ipynb` — Data cleaning & balancing
2. `notebooks/02_centralized_banglabert.ipynb` — Centralized baseline
3. `notebooks/03_federated_fedavg.ipynb` — Federated FedAvg
4. `notebooks/04_federated_fedprox.ipynb` — Federated FedProx

## Key Findings
- FedProx outperforms Centralized BanglaBERT by +1.36% macro F1
- FedProx outperforms FedAvg by +5.06% macro F1
- Privacy-preserving FL can match or exceed centralized performance

## Citation
```bibtex
@article{yourname2026federated,
  title={A Privacy-Preserving Federated Learning Framework for Multi-Class Bengali Fake News Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License
MIT License