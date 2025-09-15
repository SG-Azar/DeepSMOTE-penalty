
import random
import torch

import torch.nn as nn
from torch.utils.data import TensorDataset

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
import numpy as np
import torch.nn.functional as F


################################## Seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################## Models

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),

            nn.LeakyReLU(0.2, inplace=True))

        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):

        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']


        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            # nn.Tanh())
            nn.Sigmoid())

    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x


def Gen_SMOTE(X, y, n_to_sample, cl):
    n_neigh = min(6, len(X))
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)

    return samples, [cl] * n_to_sample



class CNN(nn.Module):
    def __init__(self, dropout=0.5, num_filters=64):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Keep the rest of the architecture the same
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(num_filters * 4 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, self.fc1.in_features)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False





## classification:

def classification_data_loader(balanced_data, balanced_labels, batch_size=64):

    dataset = TensorDataset(balanced_data, balanced_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


################################## Visualization

def plot_im(data, labels, title, n_rows, n_cols, unique_labels):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(title, fontsize=16)
    for col, label in enumerate(unique_labels):
        label_indices = np.where(labels == label)[0]
        selected_indices = np.random.choice(label_indices, n_rows, replace=False)
        for row, idx in enumerate(selected_indices):
            ax = axes[row, col]
            ax.imshow(data[idx].squeeze(), cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"Label {label}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_images(balanced_data, balanced_labels, ID):
    balanced_data = balanced_data.cpu().numpy()
    balanced_labels = balanced_labels.cpu().numpy()
    ID = np.array(ID)

    original_data = balanced_data[ID == 0]
    original_labels = balanced_labels[ID == 0]
    generated_data = balanced_data[ID == 1]
    generated_labels = balanced_labels[ID == 1]

    unique_labels = np.unique(balanced_labels)
    unique_labels_generated = np.unique(generated_labels)

    plot_im(
        original_data,
        original_labels,
        title="Original Samples (ID=0)",
        n_rows=7,
        n_cols=10,
        unique_labels=unique_labels
    )

    plot_im(
        generated_data,
        generated_labels,
        title="Generated Samples (ID=1)",
        n_rows=7,
        n_cols=9,
        unique_labels=unique_labels_generated
    )




def accuracy_metrics(all_labels, all_preds, num_classes=10):
    """
    Compute overall accuracy, class-specific accuracies, Matthews Correlation Coefficient (MCC),
    and balanced accuracy.

    Parameters:
    - all_labels: Ground truth labels (numpy array)
    - all_preds: Predicted labels (numpy array)
    - num_classes: Number of classes

    Returns:
    - overall_accuracy: Overall accuracy of the predictions
    - class_accuracies: Dictionary containing class-specific accuracies
    - mcc: Matthews Correlation Coefficient
    - balanced_accuracy: Balanced accuracy score
    """
    # overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # MCC
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    class_accuracies = {}

    # Accuracy for each class
    for cls in range(num_classes):
        class_indices = (all_labels == cls)

        if np.sum(class_indices) > 0:
            class_accuracy = accuracy_score(all_labels[class_indices], all_preds[class_indices])
            class_accuracies[cls] = class_accuracy
        else:
            class_accuracies[cls] = None

    return overall_accuracy, class_accuracies, mcc, balanced_accuracy


def calculate_metrics(y_true, y_pred, num_classes):
    # Overall Accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Class-specific Accuracies
    class_accuracies = []
    for i in range(num_classes):
        indices = (y_true == i)
        class_acc = accuracy_score(y_true[indices], y_pred[indices]) if np.sum(indices) > 0 else 0
        class_accuracies.append(class_acc)

    # Average Class-Specific Accuracy (ACSA)
    acsa = np.mean(class_accuracies)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Macro-Averaged F1 Score
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Macro-Averaged Precision-Recall AUC (PR-AUC)
    macro_pr_auc = 0
    for i in range(num_classes):
        binary_y_true = (y_true == i).astype(int)
        binary_y_pred = (y_pred == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_y_true, binary_y_pred)
        auc_score = auc(recall, precision)
        macro_pr_auc += auc_score
    macro_pr_auc /= num_classes  # Average over all classes

    # Return all computed metrics
    return {
        "Overall Accuracy": overall_accuracy,
        "Class Accuracies": class_accuracies,
        "ACSA": acsa,
        "Balanced Accuracy": balanced_acc,
        "MCC": mcc,
        "Macro F1 Score": macro_f1,
        "Macro PR-AUC": macro_pr_auc
    }

def plot_confusion_matrix(all_labels, all_preds, classes=10):

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')