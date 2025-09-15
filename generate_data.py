from utils import Gen_SMOTE
import torch
import numpy as np


def balance_data(encoder, decoder, dec_x, dec_y, device):
    encoder.eval()
    decoder.eval()

    orig_data = torch.Tensor(dec_x).to(device)

    # encode
    with torch.no_grad():
        encoded_data = encoder(orig_data)

    # decode
    with torch.no_grad():
        recons_images = decoder(encoded_data)

    recons_images = recons_images.cpu().numpy()
    encoded_data = encoded_data.cpu().numpy()

    recons_labels = dec_y

    def biased_get_class1(c):
        xbeg = dec_x[dec_y == c]
        ybeg = dec_y[dec_y == c]
        return xbeg, ybeg

    unique_labels, counts = np.unique(dec_y, return_counts=True)
    imbal = np.zeros(10, dtype=int)  # Assuming labels are from 0 to 9

    for label, count in zip(unique_labels, counts):
        imbal[label] = count

    resx = []
    resy = []
    latent_vectors = []
    latent_labels = []

    for i in range(1, 10):
        xclass, yclass = biased_get_class1(i)

        xclass = torch.Tensor(xclass).to(device)
        latent_xclass = encoder(xclass)
        latent_xclass_np = latent_xclass.detach().cpu().numpy()

        n = imbal[0] - imbal[i]
        xsamp, ysamp = Gen_SMOTE(latent_xclass_np, yclass, n, i)  # SMOTE
        ysamp = np.array(ysamp)

        xsamp = torch.Tensor(xsamp).to(device)
        ximg = decoder(xsamp)
        ximn = ximg.detach().cpu().numpy()

        resx.append(ximn)
        resy.append(ysamp)
        latent_vectors.append(xsamp.cpu().numpy())
        latent_labels.append(ysamp)

    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    resx1 = resx1.reshape(resx1.shape[0], -1)

    dec_x1 = dec_x.reshape(dec_x.shape[0], -1)
    recons_images1 = recons_images.reshape(recons_images.shape[0], -1)

    ########## Save Original Data
    np.savetxt('./data/Balanced/trn_img_original.txt', dec_x1)
    np.savetxt('./data/Balanced/trn_lab_original.txt', dec_y)

    ########## combine smote with reconstructed samples
    combx_re = np.vstack((resx1, recons_images1))
    comby_re = np.hstack((resy1, recons_labels))

    # Create ID vector: 1 for generated samples, 0 for original samples
    id_generated = np.ones(resx1.shape[0], dtype=int)  # ID = 1 for generated samples
    id_original = np.zeros(recons_images1.shape[0], dtype=int)  # ID = 0 for original samples
    ID = np.hstack((id_generated, id_original))

    # Combine latent vectors
    latent_vectors = np.vstack(latent_vectors)
    latent_labels = np.hstack(latent_labels)
    latent_vectors_combined = np.vstack((latent_vectors, encoded_data))  # z_smote + z_orig
    latent_labels_combined = np.hstack((latent_labels, recons_labels))

    # Save data
    np.savetxt('./data/Balanced/trn_img_balanced_re.txt', combx_re)
    np.savetxt('./data/Balanced/trn_lab_balanced_re.txt', comby_re)
    np.savetxt('./data/Balanced/trn_id_balanced_re.txt', ID)
    np.savetxt('./data/Balanced/trn_latent_balanced_re.txt', latent_vectors_combined)
    np.savetxt('./data/Balanced/trn_latent_labels_re.txt', latent_labels_combined)

    return combx_re, comby_re, ID

