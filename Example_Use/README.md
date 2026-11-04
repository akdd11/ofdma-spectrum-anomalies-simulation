# Anomaly Detection

Here we implement and evaluate two approches for anomaly detection in OFDMA systems using our dataset.

<p>
  <img src="Figures/Pilot_sample.jpg" width="600">
</p>

---

# Dataset

Download the dataset from the following link:

 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20341906.svg)](https://doi.org/10.5281/zenodo.20341906)

After downloading the dataset, extract the files and specify the path to the extracted data directory in `..//datapath.txt`.

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
python src/train_unsupervised.py
```

To train the supervised model, run:

```bash
python src/train_supervised.py
```

To exclude specific jammer types from the training set, use the --jammers_excluded argument followed by the names of the jammers you want to exclude:

```bash
python src/train_supervised.py --jammers_excluded 'jammer1' 'jammer2'
```

---

# Evaluation

To obtain anomaly scores for each jammer type using the selected detection method, run `src/Test.py` with the corresponding arguments.

For example, to evaluate the unsupervised method on barrage and deceptive jammers, run:

```bash
python src/Test.py --approach "unsupervised" --jammers 'barrage', 'deceptive'
```


The trained weights are provided in the `Example_Use/Weights` directory. 
You can use the jupyter notebooks provided to evaluate obtained anomaly scores. 
Additionally, you can visualize the anomalies detected by the VAE model for each sample.


---

# Model Preference

We recommend the VAE model based on two key advantages:
* Its ability to generalize detection performance to previously unseen jammer types
* It's possible to localize jammers within a spectrum image for specific jammer types

