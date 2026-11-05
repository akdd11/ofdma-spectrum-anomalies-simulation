# Import ---------------------------------------------------

import argparse
import os
import sys

import itertools
import datetime
import numpy as np

from tqdm import tqdm

import albumentations as trans
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_repo_name = "Example_Use"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from utils.functions import set_random, load_config
from src.data_loading import SpectralImagesUnsupervised, load_labels, get_dataset_path
from src.models import Encoder, Decoder


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
set_random(42)


config_path = os.path.join(_module_path, "src", "config.yml")
config = load_config(config_path)

# ------------------------------------------------
# Batch size for train and inference 
# ------------------------------------------------
train_batch_size = config["hyperparameters"]["train_batch_size"]
inference_batch_size = config["hyperparameters"]["inference_batch_size"]

# ------------------------------------------------
# Image Transformations 
# ------------------------------------------------

base_transform = trans.Compose(
    [
        ToTensorV2(),
    ]
)

train_transform = trans.Compose(
    [
        #trans.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        base_transform,
    ]
)


# ------------------------------------------------
# Data Loading 
# ------------------------------------------------

def get_dataset(root, df, load_dt_images=True):

    train_samples = df[df["split_unsupervised"] == "train"]
    trainset = SpectralImagesUnsupervised(
        root,
        transform=train_transform,
        dataframe=train_samples,
        load_dt_images=load_dt_images,
        data_type="train",
    )

    normal_valid_samples = df[df["split_unsupervised"] == "valid"]
    normal_validset = SpectralImagesUnsupervised(
        root,
        transform=base_transform,
        dataframe=normal_valid_samples,
        load_dt_images=load_dt_images,
        data_type="valid",
        max = trainset.max,
        min = trainset.min,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    normal_validloader = DataLoader(
        normal_validset,
        batch_size=inference_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return trainloader, normal_validloader


# ------------------------------------------------
# Training Loop
# ------------------------------------------------

def train(
    now,
    device,
    n_epochs,
    optimizer,
    scheduler,
    encoder,
    decoder,
    trainloader,
    normal_validloader,
    reconstruction_loss,
):

    try:
        trial_name = f"VAE_{now}"
        out_dir = os.path.join(
            _module_path, "Weights", "OUT", trial_name
        )
        os.makedirs(out_dir, exist_ok=True)

        log_dir = os.path.join(
            _module_path, "Weights", "LOG", trial_name
        )
        os.makedirs(log_dir, exist_ok=True)
        log = SummaryWriter(log_dir)

        val = 3
        for epoch in tqdm(range(n_epochs), desc="Training Epoch"):

            encoder.train()
            decoder.train()

            kl_list = []
            r_list = []

            for _, orig_img, _, _, _ in trainloader:

                orig_img = orig_img.float()
                orig_img = orig_img.to(device)
                combined_inputs = orig_img
                combined_targets = orig_img

                optimizer.zero_grad()

                mu, log_var = encoder(combined_inputs)
                std = torch.exp(0.5 * log_var)
                epsilon = torch.randn_like(std).to(device)
                z = mu + std * epsilon
                decoded_x = decoder(z)
                r_loss = reconstruction_loss(
                    decoded_x, combined_targets
                    )
                kl = -0.5 * torch.mean(
                    1 + log_var - mu.pow(2) - log_var.exp()
                    )
                
                loss = r_loss + 0.0001 * kl

                kl_list.append(kl.item())
                r_list.append(r_loss.item())
                
                loss.backward()
                optimizer.step()
            
            if scheduler != None:
                scheduler.step()

            log.add_scalar("Rec_Loss/Train", np.nanmean(r_list), global_step=epoch)
            log.add_scalar("KL_Loss/Train", np.nanmean(kl_list), global_step=epoch)

            with torch.inference_mode():

                encoder.eval()
                decoder.eval()

                valid_Rec_loss = []
                valid_kl_loss = []

                for _, img, _, _, _ in normal_validloader:


                    img = img.float()
                    img = img.to(device)

                    mu, log_var = encoder(img)
                    std = torch.exp(0.5 * log_var)
                    epsilon = torch.randn_like(std).to(device)
                    z = mu + std * epsilon
                    decoded_x = decoder(z)
                    v_r_loss = reconstruction_loss(decoded_x, img)
                    v_kl = -0.5 * torch.mean(
                        1 + log_var - mu.pow(2) - log_var.exp()
                        )
                    valid_Rec_loss.append(v_r_loss.item())
                    valid_kl_loss.append(v_kl.item())

            log.add_scalar(
                "Rec_Loss/Valid", np.nanmean(valid_Rec_loss), global_step=epoch
            )
            log.add_scalar(
                "KL_Loss/Valid", np.nanmean(valid_kl_loss), global_step=epoch
            )

            torch.save(
                encoder.state_dict(),
                os.path.join(out_dir, f"encoder_final.pt"),
                )
            torch.save(
                decoder.state_dict(),
                os.path.join(out_dir, f"decoder_final.pt")
                )
        
            if np.nanmean(valid_Rec_loss) < 0.9 * val:
                val = np.mean(valid_Rec_loss)
                torch.save(
                    encoder.state_dict(),
                    os.path.join(out_dir, f"encoder_best.pt"),
                    )
                torch.save(
                    decoder.state_dict(),
                    os.path.join(out_dir, f"decoder_best.pt"),
                    )


        log.close()

    except Exception as e:

        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # ------------------------------------------------
    # Parse arguments
    # ------------------------------------------------
    #parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--data_path", 
    #    type=str,
    #    help= "The path for the dataset", 
    #    required=True,
    #    )
    
    #args = parser.parse_args()
    #data_path = args.data_path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_number",
        type=int,
        default=config["dataset_nr"],
        help="Dataset number, i.e., subfolder of the datasets folder given in datapath.txt of the repository root (default: dataset_nr in src/config.yml)",
    )
    args = parser.parse_args()

    # ------------------------------------------------
    # GPU setup
    # ------------------------------------------------
    try:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print("Device:", device)
        Multiple_GPU = True

    except Exception as e:
        print(e)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        Multiple_GPU = False

    now = datetime.datetime.now().strftime("%m%d-%H%M%S")
    data_path = get_dataset_path(args.dataset_number)
    df = load_labels(data_path)
    max_lr = 0.001
    trainloader, normal_validloader = get_dataset(data_path, df, load_dt_images=False)


    # ------------------------------------------------
    # Model, Optimizer, and Scheduler setup
    # ------------------------------------------------
    encoder = Encoder()
    decoder = Decoder()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    n_epochs = config["hyperparameters"]["unsupervised"]["n_epochs"]
    #n_epochs = 300

    reconstruction_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(
            itertools.chain(encoder.parameters(), decoder.parameters()), lr=max_lr
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 200],
            gamma=0.1  # Multiply LR by 0.1 at each milestone
        )

    # ------------------------------------------------
    # Training Loop
    # ------------------------------------------------
    train(
        now,
        device,
        n_epochs,
        optimizer,
        scheduler,
        encoder,
        decoder,
        trainloader,
        normal_validloader,
        reconstruction_loss,
    )
