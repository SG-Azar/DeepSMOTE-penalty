"""
Creates the imbalanced train, val, test datasets from raw data with desired imbalanced ratios
"""

import os
import torch
import random
import numpy as np
from torchvision import datasets
from torchvision import transforms

####### set seeds

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 36
set_seed(seed)

root_dir = './data/Processed/v2/'


train_img_dir = os.path.join(root_dir, 'trn_img')
train_lab_dir = os.path.join(root_dir, 'trn_lab')

val_img_dir = os.path.join(root_dir, 'val_img')
val_lab_dir = os.path.join(root_dir, 'val_lab')



os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lab_dir, exist_ok=True)

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lab_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root=root_dir, train=True, download=True, transform=transform)

# Define the desired imbalance in the dataset
imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40] ####################################################


class_counts = {i: 0 for i in range(10)}
validation_counts = {i: 0 for i in range(10)}

# Calculate validation counts (10% of the imbalanced data per class)
val_counts = {i: max(1, int(0.1 * imbal[i])) for i in range(10)}
train_counts = {i: imbal[i] - val_counts[i] for i in range(10)}

train_img_filename = os.path.join(train_img_dir, 'imb_trn_img.txt')
train_lab_filename = os.path.join(train_lab_dir, 'imb_trn_lab.txt')
val_img_filename = os.path.join(val_img_dir, 'imb_val_img.txt')
val_lab_filename = os.path.join(val_lab_dir, 'imb_val_lab.txt')

with open(train_img_filename, 'w') as train_img_file, \
     open(train_lab_filename, 'w') as train_lab_file, \
     open(val_img_filename, 'w') as val_img_file, \
     open(val_lab_filename, 'w') as val_lab_file:

    for img, label in train_dataset:

        img_np = img.numpy().reshape(-1)

        if validation_counts[label] < val_counts[label]:
            np.savetxt(val_img_file, img_np[np.newaxis, :], fmt='%.6f')
            val_lab_file.write(f"{label}\n")
            validation_counts[label] += 1
        elif class_counts[label] < train_counts[label]:

            np.savetxt(train_img_file, img_np[np.newaxis, :], fmt='%.6f')
            train_lab_file.write(f"{label}\n")
            class_counts[label] += 1

        if all(validation_counts[i] >= val_counts[i] and class_counts[i] >= train_counts[i] for i in range(10)):
            break

print(f"Downloaded and saved imbalanced dataset:")
print(f"- Training data: {train_img_filename} and {train_lab_filename}")
print(f"- Validation data: {val_img_filename} and {val_lab_filename}")

###### Save test set

test_img_dir = os.path.join(root_dir, 'tst_img')
test_lab_dir = os.path.join(root_dir, 'tst_lab')

os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_lab_dir, exist_ok=True)

test_dataset = datasets.FashionMNIST(root=root_dir, train=False, download=True, transform=transform)

# Save the full test set (unchanged)
full_test_img_filename = os.path.join(test_img_dir, 'all_tst_img.txt')
full_test_lab_filename = os.path.join(test_lab_dir, 'all_tst_lab.txt')

with open(full_test_img_filename, 'w') as full_test_img_file, \
     open(full_test_lab_filename, 'w') as full_test_lab_file:

    for img, label in test_dataset:
        img_np = img.numpy().reshape(-1)  # Flatten the image
        np.savetxt(full_test_img_file, img_np[np.newaxis, :], fmt='%.6f')
        full_test_lab_file.write(f"{label}\n")

print(f"Saved full test set: {full_test_img_filename} and {full_test_lab_filename}")


###### Save imbalanced test set
# Define the new imbalanced distribution for the test set
test_imbal = [1000, 500, 250, 187, 125, 87, 50, 25, 15, 10]

test_class_counts = {i: 0 for i in range(10)}

imb_test_img_filename = os.path.join(test_img_dir, 'imb_tst_img.txt')
imb_test_lab_filename = os.path.join(test_lab_dir, 'imb_tst_lab.txt')

with open(imb_test_img_filename, 'w') as imb_test_img_file, \
     open(imb_test_lab_filename, 'w') as imb_test_lab_file:

    for img, label in test_dataset:
        if test_class_counts[label] < test_imbal[label]:

            img_np = img.numpy().reshape(-1)

            np.savetxt(imb_test_img_file, img_np[np.newaxis, :], fmt='%.6f')
            imb_test_lab_file.write(f"{label}\n")
            test_class_counts[label] += 1

        if all(test_class_counts[i] >= test_imbal[i] for i in range(10)):
            break

print(f"Saved imbalanced test set:")
print(f"- Images: {imb_test_img_filename}")
print(f"- Labels: {imb_test_lab_filename}")