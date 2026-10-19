# Anomaly Detection

Here we implement and evaluate two approches for anomaly detection in OFDMA systems using our dataset.

<p>
  <img src="Figures/Pilot_sample.jpg" width="600">
</p>

---

# Dataset

Download the dataset from the following link:

 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20341906.svg)](https://doi.org/10.5281/zenodo.20341906)

After downloading the dataset, unzip and place it inside the `Example_Use/Dataset` directory.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/akdd11/ofdma-spectrum-anomalies-simulation.git
cd ofdma-spectrum-anomalies-simulation/Example_Use
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

To train the unsupervised (VAE) model, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-4
```

To train the supervised (ResNet) model, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-2
```

Or, to fine-tune the ResNet18 model using the pretrained weights located at `initial_path`, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-2 --initial_weights "initial_path"
```


The trained weights are provided in the `Example_Use/Weights` directory. 

---

# Threshold-based Baseline

A non-learned energy detector is provided in `src/threshold_detector.py`. It uses the same input as the VAE, i.e., the spectrograms aggregated to RBs on the frequency axis. For every SU and time-frequency bin (RB x OFDM symbol), the power threshold is a quantile of the received power of the normal training samples (`split_unsupervised == "train"`). The anomaly score is the largest exceedance of this threshold map, z-normalized per SU on the normal validation samples and averaged over the SUs. The ROC curve and AUC per jammer type are computed on the test split as for the VAE. Run:

```bash
python src/threshold_detector.py --quantiles 0.5 0.9 0.95 0.99 0.999
```

The dataset is taken from the datasets folder given in `datapath.txt` of the repository root, with the dataset number from `src/conf/dataset_generation.yaml` (override with `-d`). For every quantile, the ROC curve (PDF), the AUC per jammer type and jammer power, and the scores of the validation and test samples (e.g., to compute F1 with a threshold on the validation scores) are written to `Results/threshold` (relative to `Example_Use`, change with `--out_dir`). The spectrograms are loaded with the utilities in `src/utils/data_utils.py` of the repository root.

You can use the jupyter notebooks provided to evaluate each method and visualise the detected anomalies for each sample.


---

# Model Preference

We recommend the VAE model based on two key advantages:
* Its ability to generalize detection performance to previously unseen jammer types
* It's possible to localize jammers within a spectrum image for specific jammer types

