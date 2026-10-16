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

# Threshold-based Baselines

Three non-learned baselines are provided, which follow the evaluation protocol of the VAE (fitted only on normal samples of `split_unsupervised == "train"`, decision threshold at the 90th percentile of the normal validation scores, AUC and F1 per jammer type on the test split):

* `src/threshold_energy.py`: global energy detector, mean received power over the grid per SU.
* `src/threshold_re_map.py`: resource-element-wise threshold map, a high quantile of the normal power per SU and resource element.
* `src/threshold_cfar.py`: 2D cell-averaging CFAR, local power ratio of every resource element to its surrounding.

Each statistic is computed per SU, z-normalized per SU on held-out normal training samples and fused over the SUs (max and mean). Run, e.g.:

```bash
python src/threshold_energy.py --data_path ".\Dataset"
```

Run with `-h` for the detector parameters. Results (metrics, scores per sample, configuration) are written to a timestamped folder in `Example_Use/Results/threshold`. The shared loading, normalization and evaluation code is in `src/threshold_common.py`.

You can use the jupyter notebooks provided to evaluate each method and visualise the detected anomalies for each sample.


---

# Model Preference

We recommend the VAE model based on two key advantages:
* Its ability to generalize detection performance to previously unseen jammer types
* It's possible to localize jammers within a spectrum image for specific jammer types

