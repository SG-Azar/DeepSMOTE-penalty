import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import classification_data_loader, accuracy_metrics, plot_confusion_matrix, CNN,calculate_metrics
from torchvision import models
# from torchsummary import summary
import matplotlib.pyplot as plt



def classification_train(train_img, train_labels, val_img, val_labels, args, device):
    """
    Trains baseline CNN (without any penalty used for CNN and all DS methods)
    :return: model : trained classification model
    """

    train_loader = classification_data_loader(train_img, train_labels, batch_size=args['batch_size'])
    ############################################# validation dataset

    val_data = np.loadtxt(val_img).astype(np.float32)
    val_labels = np.loadtxt(val_labels).astype(np.int64)

    if val_data.ndim == 2:
        val_data = val_data.reshape(-1, 1, 28, 28)

    val_data_tensor = torch.tensor(val_data)
    val_labels_tensor = torch.tensor(val_labels)

    val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    ######################################## Model
    num_classes = torch.unique(train_labels).numel()

    model = CNN(dropout=args['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    # summary(model, input_size=(1, 28, 28))

    ######################################## Train loop

    num_epochs = args['epochs']
    early_stop_counter = 0
    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state = None
    best_epoch = 0
    epoch_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())


        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)


        print(f'Classification epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            # validation results
            overall_accuracy, class_accuracies, mcc, balanced_accuracy = accuracy_metrics(all_labels, all_preds,
                                                                                          num_classes)
            val_results = {
                "overall_accuracy": overall_accuracy,
                "class_accuracies": class_accuracies,
                "mcc": mcc,
                "balanced_accuracy": balanced_accuracy,
            }

            early_stop_counter = 0  # Reset the counter when the validation loss improves
        else:
            early_stop_counter += 1

        # Stop training if no improvement for P epochs
        if early_stop_counter >= args['patience']:
            print(
                f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {args['patience']} epochs.")
            break

    # Restore the best model state
    model.load_state_dict(best_model_state)

    # overall validation results
    print("\033[1;33mValidation Results:\033[0m")
    print(f"Overall Accuracy: {val_results['overall_accuracy']:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {val_results['mcc']:.4f}")
    print(f"Balanced Accuracy: {val_results['balanced_accuracy']:.4f}")

    # class-specific accuracies
    print("Class-Specific Accuracies:")
    for cls, acc in val_results['class_accuracies'].items():
        if acc is not None:
            print(f"Class {cls}: {acc:.4f}")
        else:
            print(f"Class {cls}: No samples available")

    # Plot training and validation loss curves

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses,'-o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses,'-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve for classification')
    plt.legend()
    plt.grid(axis='y')

    return model, best_epoch, val_results


def classification_train_CNNip(balanced_data, balanced_labels, image_val, label_val, args, device):
    """
    Train CNN classifier with implemented penalty : CNN-ip
    """

    train_loader = classification_data_loader(
        balanced_data, balanced_labels,
        train_split=args['split'], batch_size=args['batch_size']
    )

    # Load validation data
    val_data = np.loadtxt(image_val).astype(np.float32)
    val_labels = np.loadtxt(label_val).astype(np.int64)

    if val_data.ndim == 2:
        val_data = val_data.reshape(-1, 1, 28, 28)

    val_data_tensor = torch.tensor(val_data)
    val_labels_tensor = torch.tensor(val_labels)

    val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    num_classes = torch.unique(balanced_labels).numel()
    model = CNN(dropout=args['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])


    num_epochs = args['epochs']
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epoch_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss_main = criterion(outputs, labels)

            with torch.no_grad():
                selected_class = int(torch.randint(0, num_classes, (1,)))
                class_indices = (balanced_labels == selected_class).nonzero(as_tuple=True)[0]
                if class_indices.numel() > 0:
                    n_penalty = min(images.size(0), class_indices.numel())
                    sampled_indices = class_indices[torch.randperm(class_indices.size(0))[:n_penalty]]
                    penalty_images = balanced_data[sampled_indices].to(device)
                    penalty_labels = balanced_labels[sampled_indices].to(device)
                else:
                    penalty_images = images
                    penalty_labels = labels

            penalty_outputs = model(penalty_images)
            loss_penalty = criterion(penalty_outputs, penalty_labels)

            lambda_penalty = args.get('lam', 0.1)
            loss_total = loss_main + lambda_penalty * loss_penalty

            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        # === Validation ===
        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            overall_accuracy, class_accuracies, mcc, balanced_accuracy = accuracy_metrics(
                all_labels, all_preds, num_classes)
            val_results = {
                "overall_accuracy": overall_accuracy,
                "class_accuracies": class_accuracies,
                "mcc": mcc,
                "balanced_accuracy": balanced_accuracy,
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= args['patience']:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    model.load_state_dict(best_model_state)

    # === Final Validation Output ===
    print("\033[1;33mValidation Results:\033[0m")
    print(f"Overall Accuracy: {val_results['overall_accuracy']:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {val_results['mcc']:.4f}")
    print(f"Balanced Accuracy: {val_results['balanced_accuracy']:.4f}")
    print("Class-Specific Accuracies:")
    for cls, acc in val_results['class_accuracies'].items():
        if acc is not None:
            print(f"Class {cls}: {acc:.4f}")
        else:
            print(f"Class {cls}: No samples available")

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, '-o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, '-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve for Classification')
    plt.legend()
    plt.grid(axis='y')

    return model, best_epoch, val_results


def classification_train_CNNipPlus(balanced_data, balanced_labels, image_val, label_val, args, device):
    """
    Train CNN classifier with improved implemented penalty : CNN-ip+
    """

    train_loader = classification_data_loader(
        balanced_data, balanced_labels,
        train_split=args['split'], batch_size=args['batch_size']
    )

    # Load validation data
    val_data = np.loadtxt(image_val).astype(np.float32)
    val_labels = np.loadtxt(label_val).astype(np.int64)

    if val_data.ndim == 2:
        val_data = val_data.reshape(-1, 1, 28, 28)

    val_data_tensor = torch.tensor(val_data)
    val_labels_tensor = torch.tensor(val_labels)

    val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)



    num_epochs = args['epochs']
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epoch_losses = []
    val_losses = []

    num_classes = torch.unique(balanced_labels).numel()
    model = CNN(dropout=args['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Precompute inverse frequency class sampling probabilities
    class_counts = torch.bincount(balanced_labels, minlength=num_classes).float()
    inv_freq = 1.0 / class_counts
    sampling_probs = inv_freq / inv_freq.sum()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # === Main classification loss ===
            outputs = model(images)
            loss_main = criterion(outputs, labels)

            # === Penalty loss on a selected class ===
            with torch.no_grad():

                selected_class = int(torch.multinomial(sampling_probs, 1).item())
                class_indices = (balanced_labels == selected_class).nonzero(as_tuple=True)[0]
                if class_indices.numel() > 0:
                    n_penalty = min(images.size(0), class_indices.numel())
                    sampled_indices = class_indices[torch.randperm(class_indices.size(0))[:n_penalty]]
                    penalty_images = balanced_data[sampled_indices].to(device)
                    penalty_labels = balanced_labels[sampled_indices].to(device)
                else:
                    penalty_images = images
                    penalty_labels = labels

            penalty_outputs = model(penalty_images)
            loss_penalty = criterion(penalty_outputs, penalty_labels)

            # Combine losses
            lambda_penalty = args.get('lam', 0.1)
            loss_total = loss_main + lambda_penalty * loss_penalty

            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        # === Validation ===
        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            overall_accuracy, class_accuracies, mcc, balanced_accuracy = accuracy_metrics(
                all_labels, all_preds, num_classes)
            val_results = {
                "overall_accuracy": overall_accuracy,
                "class_accuracies": class_accuracies,
                "mcc": mcc,
                "balanced_accuracy": balanced_accuracy,
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= args['patience']:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    model.load_state_dict(best_model_state)

    # === Final Validation Output ===
    print("\033[1;33mValidation Results:\033[0m")
    print(f"Overall Accuracy: {val_results['overall_accuracy']:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {val_results['mcc']:.4f}")
    print(f"Balanced Accuracy: {val_results['balanced_accuracy']:.4f}")
    print("Class-Specific Accuracies:")
    for cls, acc in val_results['class_accuracies'].items():
        if acc is not None:
            print(f"Class {cls}: {acc:.4f}")
        else:
            print(f"Class {cls}: No samples available")

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, '-o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, '-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve for Classification')
    plt.legend()
    plt.grid(axis='y')

    return model, best_epoch, val_results


def classification_test_bootstrap(model, test_data, test_labels, args, device, num_bootstrap=1000):
    """
    Test a trained classification model and evaluate its performance + Perform bootstrapping

    :param model: Trained classification model
    :param test_data: Test dataset (images)
    :param test_labels: Ground truth labels
    :param args: Hyperparameters like batch size
    :param device: 'cuda' or 'cpu'
    :param num_bootstrap: Number of bootstrap resamples (default: 100)

    :return: Mean ± standard error for all classification metrics.
    """

    # Reshape test data if needed
    if test_data.ndim == 2:
        test_data = test_data.reshape(-1, 1, 28, 28)

    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    model.eval()

    num_classes = torch.unique(test_labels).numel()
    all_results = []

    with torch.no_grad():
        for _ in range(num_bootstrap):
            indices = np.random.choice(len(test_data), len(test_data), replace=True)
            bootstrap_data = test_data[indices]
            bootstrap_labels = test_labels[indices]

            all_preds = []
            for i in range(0, len(bootstrap_data), args['batch_size']):
                batch_data = bootstrap_data[i:i + args['batch_size']]

                outputs = model(batch_data)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            test_results = calculate_metrics(bootstrap_labels.cpu().numpy(), all_preds, num_classes=num_classes)

            all_results.append(test_results)


    N = len(all_results)
    aggregated_results = {}
    for key in all_results[0]:
        values = [result[key] for result in all_results]
        aggregated_results[key] = {
            'mean': np.mean(values, axis=0),
            'se': np.std(values, axis=0) / np.sqrt(N)
        }

    # results
    print("\033[1;35mBootstrap Test Results (Standard Error):\033[0m")
    for metric, stats in aggregated_results.items():
        if metric == "Class Accuracies":
            print("\nClass-Specific Accuracies:")
            for i in range(len(stats['mean'])):
                print(f"  Class {i}: {stats['mean'][i]:.4f} ± {stats['se'][i]:.4f} (SE)")
        else:
            print(f"{metric}: {stats['mean']:.4f} ± {stats['se']:.4f} (SE)")

    return aggregated_results


