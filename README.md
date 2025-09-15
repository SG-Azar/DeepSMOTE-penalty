# The DeepSMOTE Penalty Term

This repository contains code for our paper:

> **Rethinking the DeepSMOTE Penalty Term and Its
Role in Imbalanced Learning**  
> *Saeideh Ghanbari Azar, Tufve Nyholm, Tommy Löfstedt*  
> *International Conference on Tools with Artificial Intelligence, 2025*  
> DOI: *[to be added]*

If you use this code, please cite the paper above.

---

## What’s inside

- `create_imbalanced_fmnist.py` – Build **imbalanced** train/val sets and both **full** & **imbalanced** test sets from Fashion-MNIST (`./data/Processed/v2/...`). NOTE: Run this once to create an imbalanced data from raw FMNIST data and save to data/processed/v2
- `train_AE.py` – Autoencoder (AE) training variants:  
  - `train_DS_np`: AE w/o penalty  
  - `train_DS_ip`: AE with image-pair cyclic penalty  
  - `train_DS_ip_plus`: AE with class-weighted selection + cyclic penalty  
  - `train_DS_pp`: pair penalty against original images
- `generate_data.py` – Balance the train set via **latent-space SMOTE** + decoding. Saves balanced data and IDs to `./data/Balanced/`.
- `utils.py` – Models (Encoder, Decoder, CNN), latent **SMOTE**, metrics (Accuracy, MCC, Balanced Accuracy, Macro-F1), plotting, and seed setup.
- `Classification.py` – Classifier training (baseline CNN, CNN-ip, CNN-ip+) + bootstrap testing.
- `main_DS.py` – **Full pipeline**: AE → latent-SMOTE balancing → CNN training on balanced data → bootstrap tests.
- `main_CNN.py` – **Baseline**: Train CNN directly on imbalanced data → bootstrap tests.

---

## How to use

- Run `create_imbalanced_fmnist.py` once to create imbalanced data splits from raw FMNIST data (It will be saved to data/processed/v2)
- Run `main_DS.py` for **Full pipeline** and all DS-based methods in the paper: AE → latent-SMOTE balancing → CNN training on balanced data → bootstrap tests.
- Run `main_CNN.py` for **Baseline** CNN-based methods: Train CNN directly on imbalanced data → bootstrap tests.


---



