# OFDMA Spectrum Anomaly Simulation

## Abstract

This repository belongs to the paper "Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset", available on arXiv: [arXiv:2606.02102](https://arxiv.org/abs/2606.02102). It provides a modular, open-source simulation framework for generating physics-driven datasets of spectrum anomalies in OFDMA systems. It combines Blender and the Mitsuba add-on for ray-traced channel generation with Sionna-based processing to produce labeled spectrograms suitable for training and benchmarking machine learning models for spectrum anomaly detection. The framework supports configurable scenarios (legitimate transmitters, jammers, varied propagation conditions), produces detailed labels for each sample, and includes scripts to reproduce dataset generation and evaluation. The code and datasets accompany the paper "Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset", available on arXiv, and are intended to enable reproducible research and quantitative comparison of anomaly detection methods.

The dataset is available for download at Zenodo:

 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20341906.svg)](https://doi.org/10.5281/zenodo.20341906)

The three steps towards generating a dataset, namely scene generation, data generation, and creating image data and labels, are described in the following sections.

## Scene Generation

The key of this simulation is a scene, in which ray tracing is executed to obtain channel frequency responses (CFRs) between transmitters and sensing units. The corresponding scene is created with a Python script in Blender, which creates the scene and exports it in a format that can be loaded by Sionna. The scene used to generate the dataset, together with the sensing units and a random set of legitimate transmitters and a jammer, is shown in the figure below.

<p align="center">
    <img src="./figures/scene0_render.png" width="500">
</p>

### Requirements

* Blender 4.2.19 LTS
* Mitsuba Add-on for Blender 0.4.0 (follow the installation instructions [here](https://github.com/mitsuba-renderer/mitsuba-blender))

### Usage

Open Blender and go to the Scripting workspace. Open the `blender-python/create_scenario_with_obstacles.py`. It creates a scene according to the specifications in `blender-python\conf\scene_attributes.yaml`. The scene is exported to the `scenes` directory in the repository root. The exported scene can then be loaded by Sionna to perform ray tracing.

## Data Generation

### Requirements

The simulation workflow has been developed with Ubuntu 24.04, Python 3.10.4, and Sionna 1.2.2. The requirements for the Python virtual environment are listed in the file `requirements_simulation.txt`.

### Create custom intermediate data

Entry path for the simulation is the script `src/dataset_generation.py`, which executes the ray tracing and creates custom intermediate data. Simulation parameters can be configured in the file `src\conf\dataset_generation.yaml`. The dataset number and number of samples can also be configured via the command line, run with `-h` for details.The coordinates of the sensing units are specified in the file `src\conf\su_coordinates.yaml`.

The generated data is stored in the directory that is specified in the file `datapath.txt` which is located in the repository root. In this directory, a subdirectory with the specified dataset number is created, in which the custom intermediate data is stored in a subdirectory named `custom`.

### Hardware impairments

Optional receiver-side hardware impairments (SU STFT timing offset, SU LO frequency offset, RX DC offset, RX IQ imbalance) can be enabled under the `impairments:` block in `src/conf/dataset_generation.yaml`. A master `impairments.enabled` switch gates all of them at once (for an ideal-vs-impaired ablation); each impairment also has its own `enabled` flag so individual impairments can be toggled independently. See `hardware_impairments.md` for the physical rationale, parameterization and literature references behind every impairment.

Impairment parameters are properties of a sensing unit's hardware, not of a sample: they are drawn once per SU (seeded by `impairments.su_hardware_seed`) and held fixed for the whole dataset, rather than redrawn per sample. They are persisted to `su_hardware.csv` in the dataset folder (one row per SU), with columns `timing_offset_samples`, `lo_freq_offset_ppm`, `dc_offset_over_noise_db`, `dc_offset_phase_rad`, `iq_gain_imbalance_db`, `iq_phase_imbalance_deg`, and `iq_irr_db` (derived, informational only). The models are implemented in `src/utils/impairment_utils.py`.

### Create image data and labels

To further utilize the data, the script `src/custom_data_export.py` can be executed, which creates spectrograms and labels from the custom intermediate data. The dataset number can be configured via the command line, run `python src/custom_data_export.py -d <dataset_number>` to specify it. The output container is selected with `--output-format {png,hdf5}` (default `png`); both formats apply the identical 8-bit quantization scheme described below and contain the same values, just packaged differently on disk. Two kinds of per-sample data are generated in either format:
* Spectrograms per sensing unit (SU): normalized spectrograms, one per sample and SU. The spectrograms are normalized over the entire dataset, so the same value range is used for all spectrograms, allowing for a consistent representation across different samples and sensing units. Rather than using the raw dataset-wide minimum/maximum, the spectrograms are first **clipped** to a percentile-based range before being scaled to 8 bit: the lower bound is the 0.1st percentile of all spectrogram values (jammed and non-jammed), and the upper bound is the 99.99th percentile of spectrogram values from jammed samples only. This is because the raw min/max are dominated by a handful of extreme outlier values, which otherwise waste most of the 256 available grey levels on a value range that almost no pixel occupies; clipping the lower tail (dominated by noise floor, carrying no information) and the upper tail conservatively (only over jammed samples, so jammer power - the anomaly-relevant signal - is not clipped away) recovers close to 3x the effective resolution. The clipping bounds are computed per dataset from its own sample data (so they adapt automatically if the dataset is regenerated with different parameters), and are contained in the file `spectrogram_min_max.csv` (columns `min_val`/`max_val`, kept for backwards compatibility), which is stored in the root of the dataset folder. See `notebooks/analyze_clipping_percentiles.ipynb` for the analysis behind this choice.
* Resource allocation grids: contain the resource allocation of the transmitters, in which 0 corresponds to not allocated and any other value corresponds to the allocated transmitter index.

With `--output-format png` (default), these are stored as 8-bit PNG files in the same directory as the custom intermediate data, in the subdirectory `images`: spectrograms as `spectrogram-{sample_idx}-{su_idx}.png` and resource allocations as `alloc_res-{sample_idx}.png`.

With `--output-format hdf5`, both are stored in a single file `dataset.h5` in the root of the dataset folder, as two datasets:
* `spectrograms`, shape `(num_samples, num_su, num_freq_bins, num_time_bins)`, dtype `uint8`.
* `resource_allocations`, shape `(num_samples, num_freq_bins, num_time_bins)`, dtype `uint8`.

Both datasets are gzip-compressed and hold the exact same per-sample data as the PNG output (verified to be pixel-identical), just addressed by array index instead of by filename. They are combined in one file rather than split across two, since they are written together from the same per-sample loop over the same intermediate data — this makes the sample-index correspondence between the two datasets structurally guaranteed rather than relying on two files being kept in sync by convention, at negligible cost since HDF5 lets a consumer read one dataset without touching the other. `labels.csv` and `spectrogram_min_max.csv` are kept as separate files regardless of `--output-format`, since they already need to be joined with the spectrogram data by sample index for either output format.

**Using the HDF5 dataset in a PyTorch pipeline:** do not open the HDF5 file inside a `Dataset.__init__`. `DataLoader` with `num_workers > 0` creates its worker processes by forking (on Linux/Mac), and an `h5py.File` handle that is already open at that point gets inherited by every worker as a copy of the same underlying HDF5 state (read buffers, caches, locks) — since that state isn't safe to share across processes, concurrent reads through it can hang, crash, or silently return corrupted data. Store only the file path in `__init__` and open the file lazily on first use inside `__getitem__` (e.g. `if self.h5file is None: self.h5file = h5py.File(self.path, "r")`), so that each worker process opens and owns its own independent handle. This has no effect on `num_workers=0`, since there is only ever one process touching the file in that case.

In addition, in the same directory, a file named `labels.csv` is created, which contains the following labels for each sample:
* `jammer_type`: Type of the jammer (if there is no jammer, the value is "no jammer").
* `jammer_power`: Transmit power of the jammer in dBm (if there is no jammer, the value is NaN).
* `jammer_location`: Location of the jammer (if there is no jammer, the value is NaN).
* `num_legitimate_transmitters`: Number of legitimate transmitters in the scene.
* `snr_by_su_<su_idx>`: Signal-to-noise ratio (SNR) at each sensing unit (SU) in dB.
* `sjr_by_su_<su_idx>`: Signal-to-jammer ratio (SJR) at each sensing unit (SU) in dB (`inf` if there is no jammer, since the jammer power is then zero).
* `jammer_occupancy`: Fraction of the resource elements of the time-frequency grid that are occupied by the jammer (NaN if there is no jammer). Identical for all SUs, since it only depends on the jammer signal.
* `jsnr_local_by_su_<su_idx>`: Local jammer-to-signal-plus-noise ratio (JSNR) at each SU in dB (NaN if there is no jammer). Computed only over the jammer-occupied resource elements, as the ratio of mean jammer power to mean signal-plus-noise power there (noise power is estimated over the full grid to avoid distortion by a small footprint).
* `db_contrast_global_by_su_<su_idx>`: Global dB contrast at each SU (NaN if there is no jammer). The mean, over all resource elements of the grid, of the per-pixel dB difference between the spectrogram with and without the jammer.
* `db_contrast_local_by_su_<su_idx>`: Local dB contrast at each SU (NaN if there is no jammer). Same per-pixel dB difference as above, but averaged only over the jammer-occupied resource elements.
* `split_supervised`: Train/valid/test assignment for the supervised protocol (`"train"`, `"valid"`, `"test"`, or empty if unused). See "Data Splits" below.
* `split_unsupervised`: Train/valid/test assignment for the unsupervised protocol (`"train"`, `"valid"`, `"test"`, or empty if unused). See "Data Splits" below.

#### Data Splits

`split_supervised` and `split_unsupervised` are computed once at generation time (`generate_dataset_splits` in `src/utils/data_utils.py`) so that the benchmark's train/valid/test assignment is a fixed, reproducible property of the released dataset rather than something recomputed per experiment. The split is stratified: within each jammer type (including `"no jammer"`), samples are shuffled with a fixed seed and cut into `train_frac`/`test_frac`/`valid_frac` fractions, configurable in `src/conf/dataset_generation.yaml` under `split:` (default 65%/25%/10%).

* **The test set is identical between the two columns** — the same samples are held out for both protocols, so results are directly comparable.
* **Unsupervised training only ever uses `"no jammer"` samples**: `split_unsupervised` is `"train"`/`"valid"` only for normal samples; every other jammer type is only ever `"test"` or unused there, since this consumes the entire remaining pool of normal samples.
* **Supervised training uses all jammer types except those in `left_out_types`** (default `["random_hop"]`), which is only ever assigned `"test"` (at the same per-class fraction as every other type) and never `"train"`/`"valid"`. This is used to evaluate generalization to a jammer type unseen during training; see the paper for details.
* Rows with an empty split value are not part of either protocol's train/valid/test set for that column (this only occurs for `left_out_types` in `split_supervised`, since the fractions are otherwise applied to every sample of every class).

#### Jammer impact metrics

The SNR and the SJR are averaged over the whole time-frequency grid. This is a meaningful measure of difficulty when the anomaly score of a detector is pooled by averaging over the spectrogram as well, because a sparse jammer is then diluted in the same way. However, a jammer that only occupies a small part of the grid, such as the pilot jammer, can leave a strong trace on the few resource elements it actually affects while its grid-wide average power stays low. The following metrics are provided in addition, in order to describe this local impact.

The jammer occupancy $\rho$ is the fraction of resource elements that are occupied by the jammer. It is the quantity that relates the grid-wide averages to the local ones, since approximately

$$\mathrm{SJR}_\mathrm{global} \approx \mathrm{SJR}_\mathrm{local} - 10 \log_{10}(\rho).$$

It only depends on the jammer and is therefore identical for all SUs. The occupancy differs by orders of magnitude between the jammer types, which is why the grid-wide SJR alone is not directly comparable across them.

The local JSNR compares the jammer to everything else it competes with, evaluated only on the resource elements the jammer occupies,

$$\mathrm{JSNR}_\mathrm{local} = 10 \log_{10} \frac{\overline{|J|^2}}{\overline{|S|^2} + \overline{|N|^2}},$$

where $S$, $J$ and $N$ denote the signal, jammer and noise contribution to the spectrogram and the averages of $S$ and $J$ are taken over the occupied resource elements. Signal and noise are combined, because a detector operating on the spectrogram cannot separate them, they both act as background. Positive values indicate that the jammer dominates its own footprint.

The dB contrast measures the change of the spectrogram caused by the jammer directly in the logarithmic domain the spectrogram images are provided in,

$$\Delta = 20 \log_{10}|S + J + N| - 20 \log_{10}|S + N|.$$

Since the spectrograms are normalized with dataset-wide minimum and maximum values, a contrast of $\Delta$ corresponds to the same number of grey levels in every image, so this metric is expressed in the units a detector actually operates on. Two variants are provided: `db_contrast_global` is the mean of $\Delta$ over all resource elements and thereby corresponds to a mean pooled anomaly score, whereas `db_contrast_local` is the mean over the occupied resource elements only and describes how much the affected part of the spectrogram is lifted.

**NOTE**: The occupancy refers to the resource elements the jammer signal is generated on. Because of the finite analysis window, the visible footprint in the spectrogram can be somewhat wider than the occupancy suggests (most notably for the pilot jammer, whose energy sits on isolated subcarriers), and the local metrics describe the strongly affected resource elements.


### Load the data

As a starting point, the notebook `notebooks/plot_spectrograms.ipynb` shows how to load the generated spectrograms (and labels) and plot them.


#### Spectrograms

Below are example images of the generated spectrograms. Note, that the provided images are in grayscale, but are shown for better visibility here with a colorscale.

<table align="center" style="border: none;">
  <tr>
    <td align="center">
      <img src="./assets/example_barrage_sample_14000_su_4.png" width="300"><br>
      <b>Barrage Jammer</b>
    </td>
    <td align="center">
      <img src="./assets/example_deceptive_sample_10000_su_2.png" width="300"><br>
      <b>Deceptive Jammer</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./assets/example_pilot_sample_16002_su_16.png" width="300"><br>
      <b>Pilot Jammer</b>
    </td>
    <td align="center">
      <img src="./assets/example_random_hop_sample_18001_su_16.png" width="300"><br>
      <b>Random Hop Jammer</b>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="./assets/example_sweep_sample_12006_su_19.png" width="300"><br>
      <b>Sweep Jammer</b>
    </td>
  </tr>
</table>

#### Resource Allocation

The notebook `notebooks/plot_resource_allocation.png` shows how to load the resource allocation images. 

**NOTE**: The resource allocation images are 8-bit PNG images, with the pixel value corresponding to the transmitter index (0: not allocated). Hence, the images seem almost completely black, but the information is still correctly contained in the images. For visualization below and in the paper, a discrete color scheme has been applied to highlight the allocations to the users.

<p align="center">
  <img src="./assets/resource_allocation_sample_12000.png" width="300">
</p>


### Documentation

Generate the documentation for the simulation framework from the root of the repository with the following command:

```bash
pydoctor --config pydoctor.ini
```

## Detection

The baseline models for supervised and unsupervised detection are separated from the simulation framework. The corresponding code and supplements can be found in the `Example_Use` folder, which also has a separate README file. The code for the baseline models is provided as a starting point and can be further developed and improved. The code for the baseline models is not required to generate the dataset, but it can be used to evaluate the generated dataset and to provide a benchmark for future research on spectrum anomaly detection in OFDMA systems.

# Citation

If your are using the code or dataset in this repository, please cite the following paper:

```bibtex
@misc{schösser2026spectrumanomalydetectionofdma,
      title={Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset}, 
      author={Anton Schösser and Mohammadhadi Salehi and Sinuo Ma and Philipp Schulz and Gerhard Fettweis},
      year={2026},
      eprint={2606.02102},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.02102}, 
}
```