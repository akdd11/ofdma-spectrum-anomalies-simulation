# Import ---------------------------------------------------

import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm

import cv2
import albumentations as trans
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset

_repo_name = "Example_Use"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from utils.functions import aggregate_subcarriers


# Datapath ----------------------------------------------------

def get_datapath(repo_name: str):
    """Get the path to the dataset directory.

    Parameters
    ----------
    repo_name : str
        The name of the repository, e.g., 'ofdma-spectrum-anomalies-simulation

    Returns
    -------
    datapath: str
        The path to the dataset directory.
    """
    module_path = __file__[: __file__.find(repo_name)]
    with open(os.path.join(module_path, "datapath.txt"), "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip().startswith("#") or line.strip().startswith("<"):
            continue

        if len(line.strip()) > 1:
            if os.path.isdir(line.strip()):
                
                print("Data Path: ", line.strip())

                return line.strip()
            else:
                print(
                    f"Line {idx + 1} in datapath.txt does not point to a valid directory: {line.strip()}"
                )

    raise ValueError(
        "No valid datapath found in the datapath.txt file. "
        "Please check the file and ensure it contains a valid path."
    )



# Labels & Configuration --------------------------------------


def load_labels(data_path, filename="labels.csv"):
    """
    Load labels from a CSV file.
    """
    csv_path = os.path.join(data_path, filename)
    df = pd.read_csv(csv_path)
    df['name'] = df.index
    return df

def load_spectrogram_min_max(path, filename="spectrogram_min_max.csv"):
    """
    Read spectrogram min/max values from a CSV and return (min_val, max_val) as floats.
    """
    csv_path = os.path.join(path, filename)
    df = pd.read_csv(csv_path)
    min_val = df["min_val"].iloc[0]
    max_val = df["max_val"].iloc[0]
    return float(min_val), float(max_val)


def compute_aggregate_min_max(min_val, max_val, shape=(110, 70), num_sc_per_rb=12):
    """
    Compute aggregated min/max values.
    """
    x1 = np.full(shape, min_val, dtype=float)
    x2 = np.full(shape, max_val, dtype=float)
    frame1 = aggregate_subcarriers(x1, num_sc_per_rb=num_sc_per_rb)
    frame2 = aggregate_subcarriers(x2, num_sc_per_rb=num_sc_per_rb)
    min_aggregate_val = float(np.max(frame1))
    max_aggregate_val = float(np.max(frame2))
    return min_aggregate_val, max_aggregate_val


# Data Classes ------------------------------------------------

class SpectralImagesUnsupervised(Dataset):
    """
    Dataset class for loading spectrogram image patches for an Unsupervised Learning task.
    """

    def __init__(self, path: str, transform, dataframe, load_dt_images=False, data_type="train", max=None, min=None):

        if load_dt_images:
            raise NotImplementedError("Loading dt_images is currently not implemented.")

        if data_type != "train":
            if max is None:
                raise ValueError(f"max value in {data_type} dataset must not be None.")
            if min is None:
                raise ValueError(f"min value in {data_type} dataset must not be None.")
                
            self.max = max
            self.min = min
        else:
            self.max = None
            self.min = None
        
        self.transform = transform  
        self.dir = os.path.join(path, "Images")  # directory where spectrogram PNGs are stored
        self.id = []  
        self.frames = []
        self.jammer = []  # list to hold jam type labels (string initially, mapped later)
        self.num_transmitters = []  # list to hold number of legitimate transmitters for each example
        self.jammer_power = []  # list to hold jammer power value for each example

        # Load ospectrogram global min/max from metadata CSV
        min_val, max_val = load_spectrogram_min_max(path)
        # Compute min/max after the same aggregation used for inputs (so normalization is consistent)
        #min_aggregate_val, max_aggregate_val = compute_aggregate_min_max(min_val, max_val)

        if os.path.isdir(self.dir):
            image_files = set(os.listdir(self.dir)) 

            dataframe = dataframe[
                dataframe["name"].apply(lambda x: f"spectrogram-{x:05d}-00.png" in image_files)
            ]

            # Iterate over remaining examples and cache per-sensing unit images
            for n in tqdm(dataframe["name"], desc="Caching images"):
                for su in range(21):  # there are 21 sensing unit per sample (0..20)
                    su_id = f'{n:05d}-{su:02d}'  # format example and sensing unit into an id string
                    self.id.append(su_id)  

                    img = cv2.imread(
                        os.path.join(self.dir, f"spectrogram-{su_id}.png"), cv2.IMREAD_GRAYSCALE
                    )

                    img = (img / 255) * (max_val - min_val) + min_val
                    img = aggregate_subcarriers(img, num_sc_per_rb=12)

                    # Normalize aggregated image to [0, 1] using aggregated min/max
                    #img = (img - min_aggregate_val) / (max_aggregate_val - min_aggregate_val)

                    # If a transform is provided, apply it (albumentations expects keyword 'image')
                    self.frames.append(img)

                    self.jammer.append(dataframe.loc[dataframe['name'] == n, 'jammer_type'].iloc[0])
                    self.num_transmitters.append(
                        dataframe.loc[dataframe['name'] == n, 'num_legitimate_transmitters'].iloc[0]
                    )
                    self.jammer_power.append(dataframe.loc[dataframe['name'] == n, 'jammer_power'].iloc[0])

            if data_type == "train":
                train_max = np.max(self.frames)
                train_min = np.min(self.frames)
                r = train_max-train_min
                g = r/0.8

                self.min = train_min - 0.1*g
                self.max = train_max + 0.1*g

            print(f"maximum and minimum values used for normalisation: {self.max}, {self.min} ")

            for i, frame in enumerate(tqdm(self.frames, desc="normalisation")):
                img = (frame - self.min) / (self.max - self.min)
                np.clip(img, 0.0, 1.0, out=img)

                if self.transform is not None:
                    img = self.transform(image=img)["image"]

                self.frames[i] = img


            # Print available label distribution (strings) for debugging
            print("Sample Labels: \n", np.unique(self.jammer, return_counts=True))

            # Map jammer labels to integer classes; unknown -> -1
            jammer_map = {
                "no jammer": 0,
                "barrage": 1,
                "deceptive": 2,
                "pilot": 3,
                "sweep": 4,
                "random_hop": 5,
            }

            # Replace string labels with integer codes
            self.jammer = [jammer_map.get(jam, -1) for jam in self.jammer]

            # Print integer class distribution for debugging
            print("Jammer Classes: \n", np.unique(self.jammer, return_counts=True))

    def __len__(self) -> int:
        # Return number of cached frames
        return len(self.frames)

    def __getitem__(self, index):
        # Return a tuple for training/evaluation: (id, image, jammer_class, num_transmitters, jammer_power)
        return (
            self.id[index],
            self.frames[index],
            float(self.jammer[index]),
            float(self.num_transmitters[index]),
            float(self.jammer_power[index]),
        )





class SpectralImagesSupervised(Dataset):
    """
    Dataset class for loading spectrogram images for a Supervised task.
    This class averages all sensing-unit frames for each example into a single image.
    """

    def __init__(self, path: str, transform, dataframe, load_dt_images=True):

        if load_dt_images:
            raise NotImplementedError("Loading dt_images is currently not implemented.")

        self.transform = transform
        self.dir = os.path.join(path, "Images")
        self.id = []
        self.frames = []
        self.jammer = []
        self.num_transmitters = []
        self.jammer_power = []

        # Load spectrogram global min/max and compute aggregated min/max (same as unsupervised)
        min_val, max_val = load_spectrogram_min_max(path)
        min_aggregate_val, max_aggregate_val = compute_aggregate_min_max(min_val, max_val)

        if os.path.isdir(self.dir):
            image_files = set(os.listdir(self.dir))
            
            dataframe = dataframe[
                dataframe["name"].apply(lambda x: f"spectrogram-{x:05d}-00.png" in image_files)
            ]

            for n in tqdm(dataframe["name"], desc="Caching images"):
                self.id.append(n)
                frames = []
                for su in range(21):
                    su_id = f'{n:05d}-{su:02d}'
                    img_path = os.path.join(self.dir, f"spectrogram-{su_id}.png")
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = (img / 255.0) * (max_val - min_val) + min_val
                    img = aggregate_subcarriers(img, num_sc_per_rb=12)
                    img = (img - min_aggregate_val) / (max_aggregate_val - min_aggregate_val)
                    frames.append(img)

                # Average all sensing-unit frames into a single image for this example
                img = np.mean(frames, axis=0)

                if self.transform is not None:
                    transformed = self.transform(image=img)
                    self.frames.append(transformed["image"])
                else:
                    self.frames.append(img)

                self.jammer.append(dataframe.loc[dataframe['name'] == n, 'jammer_type'].iloc[0])
                self.num_transmitters.append(
                    dataframe.loc[dataframe['name'] == n, 'num_legitimate_transmitters'].iloc[0]
                )
                self.jammer_power.append(dataframe.loc[dataframe['name'] == n, 'jammer_power'].iloc[0])

            print("Labels: \n", np.unique(self.jammer, return_counts=True))
            # Map jammer labels to integers
            jammer_map = {
                "no jammer": 0,
                "barrage": 1,
                "deceptive": 2,
                "pilot": 3,
                "sweep": 4,
                "random_hop": 5,
            }
            self.jammer = [jammer_map.get(jam, -1) for jam in self.jammer]

            print("Jammer Classes: \n", np.unique(self.jammer, return_counts=True))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        return (
            self.id[index],
            self.frames[index],
            float(self.jammer[index]),
            float(self.num_transmitters[index]),
            float(self.jammer_power[index]),
        )



class SpectralImagesRegressionClassification(Dataset):
    """
    Dataset class for loading spectrogram image patches for Classification Task on the top of SJR Predictions.
    """


    def __init__(self, path: str, transform, dataframe, load_dt_images=False, data_type="train", max=None, min=None, sjr_max=None, sjr_min=None):

        if load_dt_images:
            raise NotImplementedError("Loading dt_images is currently not implemented.")

        if data_type != "train":
            if max is None:
                raise ValueError(f"max value in {data_type} dataset must not be None.")
            if min is None:
                raise ValueError(f"min value in {data_type} dataset must not be None.")
                
            self.max = max
            self.min = min

            self.sjr_max = sjr_max
            self.sjr_min = sjr_min
        else:
            self.max = None
            self.min = None
            self.sjr_max = None
            self.sjr_min = None


        self.transform = transform  
        self.dir = os.path.join(path, "Images")  # directory where spectrogram PNGs are stored
        self.id = []  
        self.frames = []
        self.sjrs = []
        self.jammer = []  # list to hold jam type labels (string initially, mapped later)
        self.num_transmitters = []  # list to hold number of legitimate transmitters for each example
        self.jammer_power = []  # list to hold jammer power value for each example

        # Load ospectrogram global min/max from metadata CSV
        min_val, max_val = load_spectrogram_min_max(path)
        # Compute min/max after the same aggregation used for inputs (so normalization is consistent)
        #min_aggregate_val, max_aggregate_val = compute_aggregate_min_max(min_val, max_val)

        if os.path.isdir(self.dir):
            image_files = set(os.listdir(self.dir)) 

            dataframe = dataframe[
                dataframe["name"].apply(lambda x: f"spectrogram-{x:05d}-00.png" in image_files)
            ]

            # Iterate over remaining examples and cache per-sensing unit images
            for n in tqdm(dataframe["name"], desc="Caching images"):
                self.id.append(f'{n:05d}') 
                
                sample_frames = []
                sample_sjr = []
                for su in range(21):  # there are 21 sensing unit per sample (0..20)
                    
                    su_id = f'{n:05d}-{su:02d}'  # format example and sensing unit into an id string 

                    img = cv2.imread(
                        os.path.join(self.dir, f"spectrogram-{su_id}.png"), cv2.IMREAD_GRAYSCALE
                    )

                    img = (img / 255) * (max_val - min_val) + min_val
                    img = aggregate_subcarriers(img, num_sc_per_rb=12)

                    # Normalize aggregated image to [0, 1] using aggregated min/max
                    #img = (img - min_aggregate_val) / (max_aggregate_val - min_aggregate_val)
                    sample_frames.append(img)
                    sample_sjr.append(dataframe.loc[dataframe['name'] == n, f'sjr_by_su_{su}'].iloc[0])

                sample_img = np.array(sample_frames)
                sample_sjr_vec = np.array(sample_sjr)

                # If a transform is provided, apply it (albumentations expects keyword 'image')
                self.frames.append(np.expand_dims(sample_img, axis=-1))

                self.jammer.append(dataframe.loc[dataframe['name'] == n, 'jammer_type'].iloc[0])
                self.num_transmitters.append(
                    dataframe.loc[dataframe['name'] == n, 'num_legitimate_transmitters'].iloc[0]
                )
                self.jammer_power.append(dataframe.loc[dataframe['name'] == n, 'jammer_power'].iloc[0])
                self.sjrs.append(sample_sjr_vec)
                

            if data_type == "train":
                train_max = np.max(self.frames)
                train_min = np.min(self.frames)
                r = train_max-train_min
                g = r/0.8

                self.min = train_min - 0.1*g
                self.max = train_max + 0.1*g
                
                sjrs = np.asarray(self.sjrs)
                self.sjr_max = np.max(sjrs[np.isfinite(sjrs)])
                self.sjr_min = np.min(sjrs)

            #print(f"maximum and minimum values used for spectrum normalisation: {self.max}, {self.min} ")

            for i, frame in enumerate(tqdm(self.frames, desc="normalisation")):

                img = (frame - self.min) / (self.max - self.min)
                np.clip(img, 0.0, 1.0, out=img)

                if self.transform is not None:
                    for su in range(img.shape[0]):
                      img[su] = self.transform(image=img[su])["image"].reshape(110,70,1)


                self.frames[i] = img

            
            for i, r in enumerate(tqdm(self.sjrs, desc="sjr normalisation")):

                assert r.shape[0] == 21, "number of sensing units are not equal to 21"

                if np.any(np.isinf(r)):
                    self.sjrs[i] = np.ones_like(r)
                else:
                    
                    normalised_vector = (r - self.sjr_min) / (self.sjr_max - self.sjr_min)
                    self.sjrs[i] = normalised_vector

            # Print available label distribution (strings) for debugging
            print(f"\n------ Samples for {data_type} ------")
            print("Sample Labels: \n", np.unique(self.jammer, return_counts=True))
            

            # Map jammer labels to integer classes; unknown -> -1
            jammer_map = {
                "no jammer": 0,
                "barrage": 1,
                "deceptive": 2,
                "pilot": 3,
                "sweep": 4,
                "random_hop": 5,
            }

            # Replace string labels with integer codes
            self.jammer = [jammer_map.get(jam, -1) for jam in self.jammer]

            # Print integer class distribution for debugging
            print("Jammer Classes: \n", np.unique(self.jammer, return_counts=True))
            print("--------------------------------------\n")

    def __len__(self) -> int:
        # Return number of cached frames
        return len(self.frames)

    def __getitem__(self, index):
        # Return a tuple for training/evaluation: (id, image, jammer_class, num_transmitters, jammer_power)
        return (
            self.id[index],
            self.frames[index],
            float(self.jammer[index]),
            float(self.num_transmitters[index]),
            float(self.jammer_power[index]),
            self.sjrs[index],
        )



class SpectralImagesSample(Dataset):
    """
    Dataset class for loading spectrogram images from a certain sample.
    """

    def __init__(self, sample_id, path: str, transform, dataframe, load_dt_images=False, max=None, min=None):

        if load_dt_images:
            raise NotImplementedError("Loading dt_images is currently not implemented.")
        
        if max is None:
            raise ValueError(f"max value must not be None.")
        if min is None:
            raise ValueError(f"min value must not be None.")
            
        self.max = max
        self.min = min
    
        self.transform = transform  
        self.dir = os.path.join(path, "Images")  # directory where spectrogram PNGs are stored
        self.id = []
        self.frames = []  
        self.jammer = []  # list to hold jam type labels (string initially, mapped later)
        self.num_transmitters = []  # list to hold number of legitimate transmitters for each example
        self.jammer_power = []  # list to hold jammer power value for each example

        # Load ospectrogram global min/max from metadata CSV
        min_val, max_val = load_spectrogram_min_max(path)
        ## Compute min/max after the same aggregation used for inputs (so normalization is consistent)
        #min_aggregate_val, max_aggregate_val = compute_aggregate_min_max(min_val, max_val)

        if os.path.isdir(self.dir):

            for su in range(21):  # there are 21 sensing unit per sample (0..20)
                su_id = f'{sample_id:05d}-{su:02d}'  # format example and sensing unit into an id string
                self.id.append(su_id)  

                img = cv2.imread(
                    os.path.join(self.dir, f"spectrogram-{su_id}.png"), cv2.IMREAD_GRAYSCALE
                )

                img = (img / 255) * (max_val - min_val) + min_val
                img = aggregate_subcarriers(img, num_sc_per_rb=12)

                self.frames.append(img)

                self.jammer.append(dataframe.loc[dataframe['name'] == sample_id, 'jammer_type'].iloc[0])
                self.num_transmitters.append(
                    dataframe.loc[dataframe['name'] == sample_id, 'num_legitimate_transmitters'].iloc[0]
                )
                self.jammer_power.append(dataframe.loc[dataframe['name'] == sample_id, 'jammer_power'].iloc[0])

            # Print available label distribution (strings) for debugging
            print("Sample Labels: \n", np.unique(self.jammer, return_counts=True))

            print(f"maximum and minimum values used for normalisation: {self.max}, {self.min} ")

            for i, frame in enumerate(tqdm(self.frames, desc="normalisation")):
                img = (frame - self.min) / (self.max - self.min)
                np.clip(img, 0.0, 1.0, out=img)

                if self.transform is not None:
                    img = self.transform(image=img)["image"]

                self.frames[i] = img

    def __len__(self) -> int:
        # Return number of cached frames
        return len(self.frames)

    def __getitem__(self, index):
        # Return a tuple for training/evaluation: (id, image, jammer_class, num_transmitters, jammer_power)
        return (
            self.id[index],
            self.frames[index],
            self.jammer[index],
            float(self.num_transmitters[index]),
            float(self.jammer_power[index]),
        )
