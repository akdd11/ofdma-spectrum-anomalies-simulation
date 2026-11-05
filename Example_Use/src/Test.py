# Import -------------------------------------------------------
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from scipy.stats import norm
from tqdm import tqdm

import sys
import os
from typing import Literal


import sys
import os
import datetime

import albumentations as trans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


_repo_name = "Example_Use"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from utils.functions import set_random, load_config, Jammers_Lebel_Assignment, Validation_Scores, Test_Scores
from src.models import ResNet18Regression, VectorClassifier, Encoder, Decoder
from src.data_loading import SpectralImagesUnsupervised, load_labels, SpectralImagesRegressionClassification, get_dataset_path
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


# ------------------------------------------Supervised---------------------------------------------


# ------------------------------------------------
# Supervised Data Loading 
# ------------------------------------------------

def supervised_trainset_statistics(root, df, title, load_dt_images=False):
    
    train_samples = df[df["split_supervised"] == "train"]
    trainset = SpectralImagesRegressionClassification(
        root,
        transform=train_transform,
        dataframe=train_samples,
        load_dt_images=load_dt_images,
        data_type="train",
    )

    print(f"\n--- SJR Statistics in Training set {title}---")
    print(f"Maximum SJR : {trainset.sjr_max}")
    print(f"Minimum SJR : {trainset.sjr_min}")
    print("------------------------------------------------------\n")

    print(f"\n--- Spectrum Statistics in Training set {title} ---")
    print(f"Maximum value : {trainset.max}")
    print(f"Minimum value : {trainset.min}")
    print("---------------------------------------------------------\n")

    return trainset.sjr_max, trainset.sjr_min, trainset.max, trainset.min


def supervised_dataloading(root, df, sjr_max, sjr_min, value_max, value_min, load_dt_images=False):
    
    test_samples = df[df["split_supervised"] == "test"]
    testset = SpectralImagesRegressionClassification(
        root,
        transform=base_transform,
        dataframe=test_samples,
        load_dt_images=load_dt_images,
        data_type="tset",
        max = value_max,
        min = value_min,
        sjr_max= sjr_max,
        sjr_min= sjr_min,
    )

    testloader = DataLoader(
        testset,
        batch_size=inference_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    return testloader

# --------------------------------------------------
# Evaluation Supervised
# --------------------------------------------------

def Evaluation_Supervised(regression_model, classifier_model, testloader, device, model_type="Including"):

    """
    model_type: "Including" or "Excluding"
    
    """
    regression_model.eval()
    classifier_model.eval()

    total = 0
    scores_list = []
    labels_list = []
    correct = 0

    with torch.inference_mode():
        for _, img, ab, _, _, _ in tqdm(testloader):

            img = img.squeeze(-1).float().to(device)

            vec = regression_model(img)
        
            labels = (ab > 0).float().to(device)

            # Forward pass
            logits = classifier_model(vec)

            Scores = torch.sigmoid(logits)

            predictions = (torch.sigmoid(logits) > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            scores_list.extend(list(Scores.cpu().numpy()))
            labels_list.extend(ab)


        test_acc = correct / total

        print(
            f"Test Accuracy {model_type}: {test_acc:.4f}"
            )

    return scores_list, labels_list



# ------------------------------------------Unsupervised---------------------------------------------


# ------------------------------------------------
# Unsupervised Data Loading 
# ------------------------------------------------

def unsupervised_trainset_statistics(root, df, load_dt_images=False):
    
    train_samples = df[df["split_unsupervised"] == "train"]
    trainset = SpectralImagesUnsupervised(
        root,
        transform=base_transform,
        dataframe=train_samples,
        load_dt_images=load_dt_images,
        data_type="train",
    )

    print(f"\n--- Spectrum Statistics in Unsupervised Training set ---")
    print(f"Maximum value : {trainset.max}")
    print(f"Minimum value : {trainset.min}")
    print("-----------------------------------------------------------\n")

    return trainset.max, trainset.min


def unsupervised_valid_dataloading(root, df, trainset_max, trainset_min, load_dt_images=False):

    valid_samples = df[df["split_unsupervised"] == "valid"]
    validset = SpectralImagesUnsupervised(
        root,
        transform=base_transform,
        dataframe=valid_samples,
        load_dt_images=load_dt_images,
        data_type="test",
        max=trainset_max,
        min=trainset_min,
    )

    valid_loader = DataLoader(
        validset,
        batch_size=inference_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return valid_loader

def unsupervised_test_dataloading(root, df, trainset_max, trainset_min, load_dt_images=False):

    test_samples = df[df["split_unsupervised"] == "test"]
    testset = SpectralImagesUnsupervised(
        root,
        transform=base_transform,
        dataframe=test_samples,
        load_dt_images=load_dt_images,
        data_type="test",
        max=trainset_max, 
        min=trainset_min,
    )

    test_loader = DataLoader(
        testset,
        batch_size=inference_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader



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
        "--approach",
        type=str,
        choices=["supervised", "unsupervised"],
        required=True,
        help="supervised or unsupervised",
    )

    parser.add_argument(
        "--jammers",
        type=str,
        nargs="+",
        default=[],
        help="List of jammers we want to have in our test set",
    )

    parser.add_argument(
        "-d",
        "--dataset_number",
        type=int,
        default=config["dataset_nr"],
        help="Dataset number, i.e., subfolder of the datasets folder given in datapath.txt of the repository root (default: dataset_nr in src/config.yml)",
    )

    args = parser.parse_args()

    #args = parser.parse_args()
    #data_path = args.data_path
    approach = args.approach
    jammers_test = args.jammers

    data_path = get_dataset_path(args.dataset_number)

    df = load_labels(data_path)


    valid_jammers = np.unique(df["jammer_type"])

    for j in jammers_test:
        assert j in valid_jammers, f"Jammer '{j}' must be one of {valid_jammers}"


    # ------------------------------------------------
    # GPU setup
    # ------------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    now = datetime.datetime.now().strftime("%m%d")

    if approach == "unsupervised":
        # --------------------------------------------------
        # Setup The VAE Model
        # --------------------------------------------------
        encoder = Encoder()
        decoder = Decoder()
            
        encoder.load_state_dict(
            torch.load("./Weights/OUT/VAE_1027-220937/encoder_final.pt", map_location = device, weights_only=True),
            strict=False
            )
        decoder.load_state_dict(
            torch.load("./Weights/OUT/VAE_1027-220937/decoder_final.pt", map_location = device, weights_only=True),
            strict=False
            )

        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # --------------------------------------------------
        # Data Loading
        # --------------------------------------------------
        trainset_max, trainset_min = unsupervised_trainset_statistics(data_path, df, load_dt_images=False)
        #trainset_max = 37.6813109332835
        #trainset_min = -95.63239091753363
        print(f"\n--------Loading the Validation Dataset----------")
        valid_loader = unsupervised_valid_dataloading(data_path, df, trainset_max, trainset_min, load_dt_images=False)

        # --------------------------------------------------
        # Scores
        # --------------------------------------------------
        print(f"\n-------Calculating the Validation Scores--------")
        Valid_Dict = Validation_Scores(valid_loader, encoder, decoder, device)
        V_scores = np.asarray(Valid_Dict["S_v_r"])
        
        os.makedirs(f"./Anomaly_Scores", exist_ok=True)
        np.save(f"./Anomaly_Scores/Validation_Scores.npy", V_scores)

        for jammer_ex in jammers_test:
                    
            print(f"\n-------Testig The Unsupervised Models on {jammer_ex} Jammers--------")

            df_test = df[df["jammer_type"].isin([jammer_ex, "no jammer"])]
            test_loader = unsupervised_test_dataloading(data_path, df_test, trainset_max, trainset_min, load_dt_images=False)
            Jammers, S_rec, Powers = Test_Scores(test_loader, encoder, decoder, Valid_Dict, device)

            Jammers = np.asarray(Jammers)
            Scores = np.asarray(S_rec, dtype=float)
            Powers = np.asarray(Powers)

            os.makedirs(f"./Anomaly_Scores/{jammer_ex}", exist_ok=True)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Labels_Unsupervised.npy", Jammers)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Scores_Unsupervised_model.npy", Scores)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Powers.npy", Powers)

        
    elif approach == "supervised":

        # --------------------------------------------------
        # Setup All Including Model
        # --------------------------------------------------

        input_dim = 21 #number of sensing units

        classifier_model_t = VectorClassifier(
            input_dim=input_dim,
            hidden_dim=32
        ).to(device)

        classifier_model_t.load_state_dict(
            torch.load("./Weights/OUT/__Classification__1028/Classification.pt", map_location = device, weights_only=True),
            strict=False
        )

        regression_model_t = ResNet18Regression(
            pretrained=True,
            freeze_backbone=True,
            ).to(device)

        regression_model_t.load_state_dict(
            torch.load("./Weights/OUT/__SJR_Regression__1028/Regression_model.pt", map_location = device, weights_only=True),
            strict=False
            )

        sjr_max, sjr_min, value_max, value_min = supervised_trainset_statistics(data_path, df, "All jammers Included", load_dt_images=False)
        
        for jammer_ex in jammers_test:
            
            print(f"\n-------Testig Two Supervised Models on {jammer_ex} Jammers--------")

            # --------------------------------------------------
            # Setup The Excluding Model
            # --------------------------------------------------

            classifier_model_e = VectorClassifier(
                input_dim=input_dim,
                hidden_dim=32
            ).to(device)

            classifier_model_e.load_state_dict(
                torch.load(f"./Weights/OUT/__Classification_{jammer_ex}_1028/Classification.pt", map_location = device, weights_only=True),
                strict=False
            )


            regression_model_e = ResNet18Regression(
                pretrained=True,
                freeze_backbone=True,
                ).to(device)

            regression_model_e.load_state_dict(
                torch.load(f"./Weights/OUT/__SJR_Regression_{jammer_ex}_1028/Regression_model.pt", map_location = device, weights_only=True),
                strict=False
                )

            df_exc = df[df["jammer_type"] != jammer_ex]
            df_test = df[df["jammer_type"].isin([jammer_ex, "no jammer"])]

            sjr_max_exc, sjr_min_exc, value_max_exc, value_min_exc = supervised_trainset_statistics(
                data_path, df_exc, 
                f"Jammer {jammer_ex} Excluded", 
                load_dt_images=False
                )
            
            testloader_all = supervised_dataloading(data_path, df_test, sjr_max, sjr_min, value_max, value_min, load_dt_images=False)
            testloader_exc = supervised_dataloading(data_path, df_test, sjr_max_exc, sjr_min_exc, value_max_exc, value_min_exc, load_dt_images=False)
            
            scores_list_t, labels_list = Evaluation_Supervised(regression_model_t, classifier_model_t, testloader_all, device, model_type="Including")
            scores_list_e, _ = Evaluation_Supervised(regression_model_e, classifier_model_e, testloader_exc, device, model_type="Excluding")


            true_labels = Jammers_Lebel_Assignment(labels_list)
            Jammers = np.asarray(true_labels)
            Scores_t = np.asarray(scores_list_t, dtype=float)
            Scores_e = np.asarray(scores_list_e, dtype=float)

            os.makedirs(f"./Anomaly_Scores/{jammer_ex}", exist_ok=True)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Labels_Supervised.npy", Jammers)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Scores_Supervised_including_model.npy", Scores_t)
            np.save(f"./Anomaly_Scores/{jammer_ex}/Scores_Supervised_excluding_model.npy", Scores_e)
            

        
    else:
        print("Error:\n --approach should be mensioned either unsupervised or supervised")