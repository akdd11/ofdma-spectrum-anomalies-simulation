
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

import os
import random
from tqdm import tqdm

# --------------------------------
# Basic Functions
# --------------------------------

def set_random(seed: int) -> None:
    """
    Set random seeds and enable deterministic behavior
    for reproducible PyTorch training.

    Args:
        seed (int): Random seed.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True)


def load_config(yaml_path) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# --------------------------------
# Image manupulation & Augmentation
# --------------------------------

def aggregate_subcarriers(spectrogram, num_sc_per_rb) -> np.ndarray:
    """
    Aggregate spectrogram subcarriers into resource blocks.

    Args:
        spectrogram (np.ndarray):
            Input spectrogram with shape (num_subcarriers, num_time_steps).

        num_sc_per_rb (int):
            Number of subcarriers per resource block.

    Returns:
        np.ndarray:
            Aggregated spectrogram.
    """
    num_rows, num_cols = spectrogram.shape
    num_aggregated_rows = num_rows // num_sc_per_rb
    aggregated_spectrogram = np.zeros((num_aggregated_rows, num_cols))

    for i in range(num_aggregated_rows):
        start_row = i * num_sc_per_rb
        end_row = start_row + num_sc_per_rb
        aggregated_spectrogram[i, :] = 10*np.log10(np.sum(10**(spectrogram[start_row:end_row, :] / 10), axis=0))

    return aggregated_spectrogram


def fold_image_horizontal(img: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Circularly shift image to the right by `shift` pixels.

    Args:
        img: Tensor of shape (C, H, W)
        shift: Number of pixels to shift right

    Returns:
        Folded image tensor
    """
    return torch.roll(img, shifts=shift, dims=-1)


def flip_x_axis(img: torch.Tensor) -> torch.Tensor:
    """
    Flip image along the x-axis.

    Args:
        img: Tensor of shape (C, H, W)

    Returns:
        Flipped image tensor
    """
    return torch.flip(img, dims=[-1])



# --------------------------------
# Low-pass filter
# --------------------------------


def low_pass_filter_fft(image: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Apply a low-pass filter using the Fourier transform.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    cutoff : float
        Cutoff frequency radius (in pixels).

    Returns
    -------
    np.ndarray
        Filtered image.
    """

    if image.ndim != 2:
        raise ValueError("Input must be a 2D array")

    # Fourier transform
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)

    h, w = image.shape
    cy, cx = h // 2, w // 2

    # Create circular low-pass mask
    Y, X = np.ogrid[:h, :w]
    mask = (Y - cy)**2 + (X - cx)**2 <= cutoff**2

    # Apply mask
    F_shift_filtered = F_shift * mask

    # Inverse Fourier transform
    F_ishift = np.fft.ifftshift(F_shift_filtered)
    image_filtered = np.fft.ifft2(F_ishift)

    return np.real(image_filtered)


# --------------------------------
# Latent Similarity
# --------------------------------

def max_cosine_similarities_plot(vectors):
    """
    vectors: array-like of shape (20, D) 
             (20 vectors of dimension D)

    Returns:
        np.ndarray of shape (20,)
        each entry = mean cosine similarity of that vector to the others
    """
    vectors = np.asarray(vectors)
    
    # normalize vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / norms

    # cosine similarity matrix (20 x 20)
    cos_sim_matrix = vectors_norm @ vectors_norm.T

    # exclude self-similarity by masking diagonal
    np.fill_diagonal(cos_sim_matrix, 0)
    # maximum similarity to another vector
    max_cos_sim = cos_sim_matrix.max(axis=1)
    plt.plot(np.arange(0,21), max_cos_sim, marker='o', linestyle='--')
    plt.xticks(range(0,21))
    plt.xlabel("SUs")
    plt.ylim(0.5,1)
    plt.ylabel("Cosine Similarity")
    plt.title("Similarity of the SUs in the latent space")
    plt.grid()
    plt.show()
    return max_cos_sim



def max_cosine_similarities(vectors):
    """
    vectors: array-like of shape (20, D) 
             (20 vectors of dimension D)

    Returns:
        np.ndarray of shape (20,)
        each entry = mean cosine similarity of that vector to the others
    """
    vectors = np.asarray(vectors)
    
    # normalize vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / norms

    # cosine similarity matrix (20 x 20)
    cos_sim_matrix = vectors_norm @ vectors_norm.T

    # exclude self-similarity by masking diagonal
    np.fill_diagonal(cos_sim_matrix, 0)
    # maximum similarity to another vector
    max_cos_sim = cos_sim_matrix.max(axis=1)
 
    return max_cos_sim


def Jammers_Lebel_Assignment(labels_list):
    
    true_labels = []

    for jammer in labels_list:
        if jammer == 0:
            true_labels.append("Normal")

        elif jammer == 1:
            true_labels.append("Barrage")
            
        elif jammer == 2:
            true_labels.append("Deceptive")
            
        elif jammer == 3:
            true_labels.append("Pilot")
            
        elif jammer == 4:
            true_labels.append("Sweep")
            
        elif jammer == 5:
            true_labels.append("Random_hop")

    return true_labels


# ------------------------------------------------
# Calculating the Unsupervised Anommaly Scores
# ------------------------------------------------

def compute_loss(
        original: np.ndarray, 
        reconstructed: np.ndarray
        ) -> np.ndarray:
    """Clipped, element-wise reconstruction loss."""
    
    #return torch.clamp((original - reconstructed) * original, min=0)
    return torch.clamp(original - reconstructed, min=0)

def Validation_Scores(
        validloader, 
        encoder, 
        decoder, 
        device
        ):
    
    S_v_r = []
    
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        for id_v, img, jammer, _, _  in tqdm(validloader, desc="Scores On the Validation Set"):
            assert all(x[:6] == id_v[0][:6] for x in id_v), "Sensing units are from different sampels"
        
            img = img.float()
            img = img.to(device)
                
            z, _ = encoder(img) 
            decoded_x = decoder(z)
            loss = compute_loss(img, decoded_x)
            
            f_loss = loss.cpu().numpy()
            
            for i in range(loss.shape[0]):
                f_loss[i][0] = low_pass_filter_fft(loss[i][0].cpu().numpy(), 30)

            score_r = np.max(f_loss)
            S_v_r.append(score_r)

    # Compute mean & standard deviation of the scores in the validation set
    mu_r = np.mean(S_v_r)
    std_r = np.sqrt(np.var(S_v_r))

    return {
        "S_v_r": [S_v_r],
        "mu_r": [mu_r], 
        "std_r": [std_r]
        }

def Test_Scores(testloader,
                encoder, decoder,
                valid_dict,
                device):

    Jammers = []
    
    S_rec = []
    mu_r = valid_dict["mu_r"][0]
    std_r = valid_dict["std_r"][0]

    Powers = []
    
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
            
        for id_v, img, jammer, _, p  in tqdm(testloader, desc="Scores On the Test Set"): 
            assert all(x[:6] == id_v[0][:6] for x in id_v), "Sensing units are from different sampels"
            
            valid_p = [su_p for su_p in p if not np.isnan(su_p)]
            assert all(su_p == valid_p[0] for su_p in valid_p), "Jammer Power for all sensing units in one sample should be same"
            
            img = img.float()
            img = img.to(device)
                
            z, _ = encoder(img)
            decoded_x = decoder(z)

            loss = compute_loss(img, decoded_x)
            
            f_loss = loss.cpu().numpy()
            
            for i in range(loss.shape[0]):
                f_loss[i][0] = low_pass_filter_fft(loss[i][0].cpu().numpy(), 30)

            score_r = np.max(f_loss)

            r = (score_r - mu_r) / std_r
            S_rec.append(r)

            if np.isnan(p[0]):
                Powers.append("No Jammer")
            else:
                Powers.append(p[0].cpu().numpy())

            if jammer[0] == 0:
                Jammers.append("Normal")
            elif jammer[0] == 1:
                Jammers.append("Barrage")
                
            elif jammer[0] == 2:
                Jammers.append("Deceptive")
                
            elif jammer[0] == 3:
                Jammers.append("Pilot")
                
            elif jammer[0] == 4:
                Jammers.append("Sweep")
                
            elif jammer[0] == 5:
                Jammers.append("Random_hop")

    return Jammers, S_rec, Powers
