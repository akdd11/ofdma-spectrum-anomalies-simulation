# Import -------------------------------------------------------
import argparse
import numpy as np
from tqdm import tqdm

import sys
import os
import datetime

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
from src.models import ResNet18Regression, VectorClassifier
from src.data_loading import load_labels, SpectralImagesRegressionClassification, get_datapath
from src.train_unsupervised import base_transform


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

train_transform = trans.Compose(
    [
        #trans.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        base_transform,
    ]
)

# ------------------------------------------------
# Data Loading 
# ------------------------------------------------

def get_dataset(root, df, excluded, load_dt_images=False):
    
    train_samples = df[
        (df["split_supervised"] == "train") &
        (~df["jammer_type"].isin(excluded))
    ]
    trainset = SpectralImagesRegressionClassification(
        root,
        transform=train_transform,
        dataframe=train_samples,
        load_dt_images=load_dt_images,
        data_type="train",
    )

    print("\n--- SJR Statistics in Training set ---")
    print(f"Maximum SJR (Jammer Present) : {trainset.sjr_max}")
    print(f"Minimum SJR (Jammer Present) : {trainset.sjr_min}")
    print("--------------------------------------\n")

    print("\n--- Spectrum Statistics in Training set ---")
    print(f"Maximum value : {trainset.max}")
    print(f"Minimum value : {trainset.min}")
    print("-------------------------------------------\n")

    valid_samples = df[
        (df["split_supervised"] == "valid") &
        (~df["jammer_type"].isin(excluded))
        ]
    
    validset = SpectralImagesRegressionClassification(
        root,
        transform=base_transform,
        dataframe=valid_samples,
        load_dt_images=load_dt_images,
        data_type="valid",
        max = trainset.max,
        min = trainset.min,
        sjr_max= trainset.sjr_max,
        sjr_min= trainset.sjr_min,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    validloader = DataLoader(
        validset,
        batch_size=inference_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return trainloader, validloader


# ------------------------------------------------
# Training Loop 1
# ------------------------------------------------

def train_regression_model(
        now,
        device,
        n_epochs,
        optimizer,
        scheduler,
        model,
        trainloader,
        validloader,
        loss_function,
        exclude,
    ):

    try:
        
        ex = f"_{''.join(exclude)}_"
        print(f"\n--- Training on all data except for jammers {ex} ---")

        trial_name = f"__SJR_Regression{ex}{now}"
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

            model.train()

            l_list = []

            for _, orig_img, _, _, _, sjr in trainloader:

                orig_img = orig_img.squeeze(-1).float().to(device)

                sjr = sjr.float().to(device)
                labels = 1 - sjr

                optimizer.zero_grad()

                pred = model(orig_img)
                l_loss = loss_function(
                    pred, labels
                    )
                
                loss = l_loss

                l_list.append(loss.item())
                
                loss.backward()
                optimizer.step()


            if scheduler != None:
                scheduler.step()

            log.add_scalar("Loss/Train", np.nanmean(l_list), global_step=epoch)

            with torch.inference_mode():

                model.eval()

                valid_l_loss = []

                for _, img, _, _, _, sjr in validloader:


                    img = img.squeeze(-1).float().to(device)

                    sjr = sjr.float().to(device)
                    labels = 1 - sjr

                    pred = model(img)
                    v_loss = loss_function(pred, labels)
                    valid_l_loss.append(v_loss.item())

            log.add_scalar(
                "Loss/Valid", np.nanmean(valid_l_loss), global_step=epoch
            )

            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"Regression_model.pt"),
                )
        
            if np.nanmean(valid_l_loss) < 0.9 * val:
                val = np.nanmean(valid_l_loss)
                torch.save(
                    model.state_dict(),
                    os.path.join(out_dir, f"Regression_model_best.pt"),
                    )


        log.close()

    except Exception as e:

        print(f"An error occurred in regression loop: {e}")


# ------------------------------------------------
# Training Loop 2
# ------------------------------------------------

def train_classifier(
        regression_model, 
        classifier_model, 
        trainloader, 
        validloader, 
        optimizer, 
        scheduler, 
        criterion, 
        num_epochs, 
        device,
        exclude,
        ):

    try:
        ex = f"_{''.join(exclude)}_"
        trial_name = f"__Classification{ex}{now}"
        out_dir = os.path.join(
            _module_path, "Weights", "OUT", trial_name
        )
        os.makedirs(out_dir, exist_ok=True)


        for epoch in range(num_epochs):

            regression_model.eval()
            classifier_model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for _, img, ab, _, _, _  in trainloader:

                img = img.squeeze(-1).float().to(device)

                vec = regression_model(img)
                
                labels = (ab > 0).float().to(device)

                # Forward pass
                logits = classifier_model(vec)

                # Loss
                loss = criterion(logits, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()

                predictions = (torch.sigmoid(logits) > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                if scheduler != None:
                    scheduler.step()

            train_epoch_loss = total_loss
            train_epoch_acc = correct / total

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_epoch_loss:.4f} "
                f"Train Accuracy: {train_epoch_acc:.4f}"
                )

            total_loss = 0.0
            correct = 0
            total = 0

            with torch.inference_mode():
                for _, img, ab, _, _, _ in validloader:

                    regression_model.eval()
                    classifier_model.eval()

                    img = img.squeeze(-1).float().to(device)

                    vec = regression_model(img)
                
                                
                    labels = (ab > 0).float().to(device)
        
                    # Forward pass
                    logits = classifier_model(vec)
        
                    # Loss
                    loss = criterion(logits, labels)

                    # Statistics
                    total_loss += loss.item()
        
                    predictions = (torch.sigmoid(logits) > 0.5).float()
        
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)


            valid_epoch_loss = total_loss
            valid_epoch_acc = correct / total

            print(
                f"Validation Loss: {valid_epoch_loss:.4f} "
                f"Validation Accuracy: {valid_epoch_acc:.4f}"
                )

        torch.save(
        classifier_model.state_dict(),
        os.path.join(out_dir, f"Classification.pt"),
        )
    
    except Exception as e:

        print(f"An error occurred in classification loop: {e}")
    

if __name__ == "__main__":
    # ------------------------------------------------
    # Parse arguments
    # ------------------------------------------------
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--data_path", 
    #    type=str,
    #    help= "The path for the dataset", 
    #    required=True,
    #    )
    parser.add_argument(
        "--jammers_excluded", 
        type=str,
        nargs="+",
        default=[],
        help="List of jammers to exclude", 
        required=False,
        )
    
    args = parser.parse_args()
    #data_path = args.data_path
    jammers_excluded = args.jammers_excluded

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
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        Multiple_GPU = False

    now = datetime.datetime.now().strftime("%m%d")
    data_path = get_datapath(_repo_name)
    df = load_labels(data_path)
    train_loader, valid_loader = get_dataset(data_path, df, excluded=jammers_excluded, load_dt_images=False)
    
    # ------------------------------------------------
    # Model, Optimizer, and Scheduler (Regression)
    # ------------------------------------------------
    
    r_model = ResNet18Regression(
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    r_loss_function = nn.MSELoss()

    r_optimizer = torch.optim.AdamW(
        r_model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    r_scheduler = torch.optim.lr_scheduler.StepLR(
        r_optimizer,
        step_size=25,
        gamma=0.1,
    )


    r_n_epochs=config["hyperparameters"]["supervised"]["Regression_n_epochs"]

    # ------------------------------------------------
    # Training (Regression)
    # ------------------------------------------------

    train_regression_model(
        now,
        device,
        r_n_epochs,
        r_optimizer,
        r_scheduler,
        r_model,
        train_loader,
        valid_loader,
        r_loss_function,
        jammers_excluded,
    )

    print("Regression Training Completed!")

    # --------------------------------------------------
    # Model, Optimizer, and Scheduler (Classification)
    # --------------------------------------------------

    classifier_model = VectorClassifier(
        input_dim=21,
        hidden_dim=32
    ).to(device)

    # Freez the encoder
    for param in r_model.parameters():
        param.requires_grad = False

    c_optimizer = torch.optim.Adam(
        classifier_model.parameters(),
        lr=0.001
    )

    c_n_epochs=config["hyperparameters"]["supervised"]["Classification_n_epochs"]

    c_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        c_optimizer,
        max_lr=0.001,
        epochs=c_n_epochs,
        steps_per_epoch=len(train_loader),
    )

    c_loss_function = nn.BCEWithLogitsLoss()

    train_classifier(
        r_model, 
        classifier_model, 
        train_loader, 
        valid_loader, 
        c_optimizer, 
        c_scheduler, 
        c_loss_function, 
        c_n_epochs, 
        device,
        jammers_excluded,
        )
    
    
    if Multiple_GPU:
        torch.distributed.destroy_process_group()
        print("Training Completed!")
    else:
        print("Training Completed!")
