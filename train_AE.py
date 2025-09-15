'''
Functions for training the AE
'''

from utils import Encoder, Decoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

import matplotlib.pyplot as plt


def train_DS_ip(dec_x, dec_y,args, device):
    encoder = Encoder(args)
    decoder = Decoder(args)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    num_workers = 0

    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x,
                              tensor_y)
    train_loader = torch.utils.data.DataLoader(mnist_bal,
                                               batch_size=args['batch_size'], shuffle=True, num_workers=num_workers)

    best_loss = np.inf

    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

        train_losses = []
        mse_losses = []
        discr_losses = []

        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0

            encoder.train()
            decoder.train()

            for images, labs in train_loader:
                encoder.zero_grad()
                decoder.zero_grad()

                images, labs = images.to(device), labs.to(device)

                z_hat = encoder(images)
                x_hat = decoder(z_hat)

                mse = criterion(x_hat, images)

                ### Pick a Random Class : uniform class sampling
                tc = np.random.choice(10, 1)

                ### Extract All Samples of the Chosen Class
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc]

                ### Randomly Pick up to args['batch_size'] Samples
                xlen = len(xbeg)
                nsamp = min(xlen, args['batch_size'])
                ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                xclass = xbeg[ind]   # Get the randomly selected images
                yclass = ybeg[ind]   # Get labels

                ### Shift Image Order to Create Pairs
                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)
                xcnew = (xclass[[xcplus], :])

                ### reshape and convert and move to gpu
                xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)

                ### Encode selected samples to latent space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)

                zclass = encoder(xclass)
                zclass = zclass.detach().cpu().numpy()

                #### Same cyclic shifting on the latent representations
                xc_enc = (zclass[[xcplus], :])
                xc_enc = np.squeeze(xc_enc)

                ### Convert back to tensor and decode
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                ximg = decoder(xc_enc)

                ### Compute the Second Loss Term
                mse2 = criterion(ximg, xcnew)

                comb_loss = mse + args['lambda'] * mse2
                comb_loss.backward()

                enc_optim.step()
                dec_optim.step()

                train_loss += comb_loss.item() * images.size(0)
                tmse_loss += mse.item() * images.size(0)
                tdiscr_loss += mse2.item() * images.size(0)

            train_loss = train_loss / len(train_loader)
            tmse_loss = tmse_loss / len(train_loader)
            tdiscr_loss = tdiscr_loss / len(train_loader)

            train_losses.append(train_loss)
            mse_losses.append(tmse_loss)
            discr_losses.append(tdiscr_loss)

            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                                                                                                 train_loss, tmse_loss,
                                                                                                 tdiscr_loss))

            if train_loss < best_loss:
                best_loss = train_loss

        # Plot the training loss vs. epoch
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Total Train Loss')
        plt.plot(mse_losses, label='MSE Loss')
        plt.plot(discr_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()

    return encoder, decoder


def train_DS_pp(dec_x, dec_y,args, device):
    encoder = Encoder(args)
    decoder = Decoder(args)


    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    num_workers = 0

    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x,
                              tensor_y)
    train_loader = torch.utils.data.DataLoader(mnist_bal,
                                               batch_size=args['batch_size'], shuffle=True, num_workers=num_workers)


    best_loss = np.inf

    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

        train_losses = []
        mse_losses = []
        discr_losses = []

        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0

            encoder.train()
            decoder.train()

            for images, labs in train_loader:
                encoder.zero_grad()
                decoder.zero_grad()

                images, labs = images.to(device), labs.to(device)

                z_hat = encoder(images)
                x_hat = decoder(z_hat)

                mse = criterion(x_hat, images)

                ### Pick a Random Class
                tc = np.random.choice(10, 1)

                ### Extract All Samples of the Chosen Class
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc]

                ### Randomly Pick up to args['batch_size'] Samples
                xlen = len(xbeg)
                nsamp = min(xlen, args['batch_size'])
                ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                xclass = xbeg[ind]   # Get the randomly selected images
                yclass = ybeg[ind]   # Get labels

                ### Shift Image Order to Create Pairs
                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)
                xcnew = (xclass[[xcplus], :])

                ### reshape and convert and move to gpu
                xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)

                ### Encode selected samples to latent space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)

                zclass = encoder(xclass)
                zclass = zclass.detach().cpu().numpy()

                #### Same cyclic shifting on the latent representations
                xc_enc = (zclass[[xcplus], :])
                xc_enc = np.squeeze(xc_enc)

                ### Convert back to tensor and decode
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                ximg = decoder(xc_enc)

                ### Compute the Second Loss Term
                mse2 = criterion(ximg, xclass)
                ################################################################################################

                comb_loss = args['gamma'] *mse + args['lambda'] * mse2
                comb_loss.backward()

                enc_optim.step()
                dec_optim.step()

                train_loss += comb_loss.item() * images.size(0)
                tmse_loss += mse.item() * images.size(0)
                tdiscr_loss += mse2.item() * images.size(0)

            # print avg training statistics
            train_loss = train_loss / len(train_loader)
            tmse_loss = tmse_loss / len(train_loader)
            tdiscr_loss = tdiscr_loss / len(train_loader)

            train_losses.append(train_loss)
            mse_losses.append(tmse_loss)
            discr_losses.append(tdiscr_loss)

            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                                                                                                 train_loss, tmse_loss,
                                                                                                 tdiscr_loss))

            if train_loss < best_loss:
                best_loss = train_loss

        # Plot the training loss vs. epoch
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Total Train Loss')
        plt.plot(mse_losses, label='MSE Loss')
        plt.plot(discr_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()

    return encoder, decoder

def train_DS_ip_plus(dec_x, dec_y,args, device):
    encoder = Encoder(args)
    decoder = Decoder(args)


    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    num_workers = 0

    # torch.Tensor returns float so if want long then use torch.tensor
    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x,
                              tensor_y)  # wrapper that turns these tensors into a dataset where each sample is a tuple (x, y).
    train_loader = torch.utils.data.DataLoader(mnist_bal,
                                               batch_size=args['batch_size'], shuffle=True, num_workers=num_workers)

    # Compute class counts
    unique_classes, class_counts = np.unique(dec_y, return_counts=True)

    # Compute inverse-proportional probabilities
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

    best_loss = np.inf

    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

        train_losses = []
        mse_losses = []
        discr_losses = []

        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0

            encoder.train()
            decoder.train()

            for images, labs in train_loader:

                encoder.zero_grad()
                decoder.zero_grad()

                images, labs = images.to(device), labs.to(device)

                z_hat = encoder(images)
                x_hat = decoder(z_hat)

                mse = criterion(x_hat, images)

                # weighted sampling for class selection
                tc = np.random.choice(unique_classes, p=class_weights)

                ### Extract All Samples of the Chosen Class
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc]

                ### Randomly Pick up to args['batch_size'] Samples
                xlen = len(xbeg)
                nsamp = min(xlen, args['batch_size'])
                ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                xclass = xbeg[ind]   # Get the randomly selected images
                yclass = ybeg[ind]   # Get labels

                ### Shift Image Order to Create Pairs
                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)
                xcnew = (xclass[[xcplus], :])

                xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)

                ### Encode selected samples to latent space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)

                zclass = encoder(xclass)
                zclass = zclass.detach().cpu().numpy()

                #### Same cyclic shifting on the latent representations
                xc_enc = (zclass[[xcplus], :])
                xc_enc = np.squeeze(xc_enc)

                ### Convert back to tensor and decode
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                ximg = decoder(xc_enc)

                ### Compute the Second Loss Term
                mse2 = criterion(ximg, xcnew)

                comb_loss = mse + args['lambda'] * mse2
                comb_loss.backward()

                enc_optim.step()
                dec_optim.step()

                train_loss += comb_loss.item() * images.size(0)
                tmse_loss += mse.item() * images.size(0)
                tdiscr_loss += mse2.item() * images.size(0)

            # print avg training statistics
            train_loss = train_loss / len(train_loader)
            tmse_loss = tmse_loss / len(train_loader)
            tdiscr_loss = tdiscr_loss / len(train_loader)

            train_losses.append(train_loss)
            mse_losses.append(tmse_loss)
            discr_losses.append(tdiscr_loss)

            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                                                                                                 train_loss, tmse_loss,
                                                                                                 tdiscr_loss))

            if train_loss < best_loss:
                best_loss = train_loss

        # Plot the training loss vs. epoch
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Total Train Loss')
        plt.plot(mse_losses, label='MSE Loss')
        plt.plot(discr_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()

    return encoder, decoder



def train_DS_np(dec_x, dec_y,args, device):
    encoder = Encoder(args)
    decoder = Decoder(args)


    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    num_workers = 0

    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x,
                              tensor_y)
    train_loader = torch.utils.data.DataLoader(mnist_bal,
                                               batch_size=args['batch_size'], shuffle=True, num_workers=num_workers)


    best_loss = np.inf
    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

        train_losses = []

        for epoch in range(args['epochs']):  # train loop
            train_loss = 0.0

            encoder.train()
            decoder.train()

            for images, labs in train_loader:
                encoder.zero_grad()
                decoder.zero_grad()

                images, labs = images.to(device), labs.to(device)

                z_hat = encoder(images)
                x_hat = decoder(z_hat)

                mse = criterion(x_hat, images)

                mse .backward()

                enc_optim.step()
                dec_optim.step()

                train_loss += mse.item() * images.size(0)

            train_loss = train_loss / len(train_loader)

            train_losses.append(train_loss)

            print('Epoch: {} \tTrain Loss: {:.6f} '.format(epoch, train_loss))

            if train_loss < best_loss:
                best_loss = train_loss

        # Plot the training loss vs. epoch
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Total Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()

    return encoder, decoder

