"""
run overall pipeline : (AE training + balancing the data + training a classifier on balanced data + testing the classifier on balanced and imbalanced test sets)
"""

import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from train_AE import train_DS_ip, train_DS_ip_plus, train_DS_np, train_DS_pp
from generate_data import balance_data

from utils import plot_images, set_seed
from Classification import classification_train, classification_test_bootstrap

####### set seeds
seed = 66
set_seed(seed)



def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Define hyperparameters (args) for AE
    args = {
        'dim_h': 64,
        'lambda': 2.11,
        'gamma': 1,
        'n_channel': 1,
        'n_z': 300,
        'lr': 0.00001,
        'epochs': 150,
        'batch_size': 64,
        'train': True

    }
    # Define hyperparameters (args) for CNN
    CL_config = {
        'patience': 30,  #
        'dropout_rate': 0.5,
        'lr': 0.0001,
        'epochs': 150,
        'batch_size': 32,
    }

    ###################################################################### Prepare train data

    # Load the training data and labels from saved files (assuming images are stored in text files)
    train_img_dir = './data/Processed/v2/trn_img'
    train_lab_dir = './data/Processed/v2/trn_lab'

    img_filename = os.path.join(train_img_dir, 'imb_trn_img.txt')
    lab_filename = os.path.join(train_lab_dir, 'imb_trn_lab.txt')

    # Load the images and labels
    train_data = np.loadtxt(img_filename)  # Shape: (num_samples, 784)
    dec_y = np.loadtxt(lab_filename, dtype=int)  # Shape: (num_samples, )

    # Reshape the images into the required shape (num_samples, 1, 28, 28)
    dec_x = train_data.reshape(-1, 1, 28, 28)


    ###################################################################### Train AE
    t1= time.time()

    encoder, decoder = train_DS_ip(dec_x, dec_y,args, device)
    t2 = time.time()
    print('Elapsed time for AE training is:', t2-t1)

    ###################################################################### Balance train data

    balanced_data, balanced_labels, ID = balance_data(encoder, decoder,dec_x, dec_y, device)

    balanced_data = balanced_data.reshape(-1, 1, 28, 28)

    balanced_data = torch.tensor(balanced_data)

    balanced_labels = torch.tensor(balanced_labels)
    balanced_labels = balanced_labels.long()

    ###################################################################### Plot some images from train data:
    plot_images(balanced_data, balanced_labels, ID)

    ###################################################################### Prepare val data for classification

    image_val = './data/Processed/v2/val_img/imb_val_img.txt'
    group_val = './data/Processed/v2/val_lab/imb_val_lab.txt'

    ##################################################### Classification
    print("Training classification model on balanced data...")
    model, _, results = classification_train(balanced_data, balanced_labels, image_val, group_val, CL_config, device)

    ##################################################### Final Classification Test (comment this out until final evaluation)
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
