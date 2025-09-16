"""
run classification directly on imbalanced data
"""
import torch
import numpy as np

import time
import matplotlib.pyplot as plt

from utils import  set_seed
from Classification import classification_train, classification_test_bootstrap, classification_train_CNNip, classification_train_CNNipPlus

####### set seeds
seed = 66
set_seed(seed)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CL_config = {
        'patience': 150,
        'lam' : 4.11,
        'dropout': 0.5,
        'lr': 0.00068,
        'epochs': 150,
        'batch_size': 64,
    }


    # File paths for training and val data
    image_trn = './data/Processed/v2/trn_img/imb_trn_img.txt'
    group_trn = './data/Processed/v2/trn_lab/imb_trn_lab.txt'

    image_val = './data/Processed/v2/val_img/imb_val_img.txt'
    group_val = './data/Processed/v2/val_lab/imb_val_lab.txt'

    #################################################### Load train data:

    trn_data = np.loadtxt(image_trn).astype(np.float32)
    trn_labels = np.loadtxt(group_trn).astype(np.int64)

    if trn_data.ndim == 2:
        trn_data = trn_data.reshape(-1, 1, 28, 28)

    trn_data = torch.tensor(trn_data)
    trn_labels = torch.tensor(trn_labels)




    ##################################################### Classification
    print("Training classification model on balanced data...")
    t1 = time.time()
    model, _, results = classification_train(trn_data, trn_labels, image_val, group_val, CL_config, device)
    t2 = time.time()

    print('Elapsed train time:', t2-t1)


    #################################################### Final Classification Test (comment this out while hyperparameter search)

    print("Final classification test on balanced data...")

    test_data_bal = './data/Processed/v2/tst_img/all_tst_img.txt'
    test_labels_bal = './data/Processed/v2/tst_lab/all_tst_lab.txt'
    test_data_bal = np.loadtxt(test_data_bal).astype(np.float32)
    test_labels_bal = np.loadtxt(test_labels_bal).astype(np.int64)
    results_bal = classification_test_bootstrap(model, test_data_bal, test_labels_bal, CL_config, device)

    print("Final classification test on imbalanced data...")

    test_data_imb = './data/Processed/v2/tst_img/imb_tst_img.txt'
    test2_labels_imb = './data/Processed/v2/tst_lab/imb_tst_lab.txt'
    test_data_imb = np.loadtxt(test_data_imb).astype(np.float32)
    test_labels_imb = np.loadtxt(test2_labels_imb).astype(np.int64)
    results_imb = classification_test_bootstrap(model, test_data_imb, test_labels_imb, CL_config, device)


    ##############################################################################

    plt.show()


if __name__ == "__main__":
    main()









