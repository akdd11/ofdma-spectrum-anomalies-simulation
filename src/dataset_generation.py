"""For a given scene, this script generates spectrograms for an OFDMA system based on ray tracing.

Most configuration can be done in the config file at conf/dataset_generation.yaml.
"""

__docformat__ = "numpy"

import os

import yaml

# limit to one GPU (can be specified in the config file)
_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "conf", "dataset_generation.yaml"
)
with open(_config_path, "r") as _f:
    _gpu_id = yaml.safe_load(_f).get("gpu_id", 0)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_gpu_id))

import argparse
import pickle as pkl
from datetime import datetime
import hydra
import logging
import numpy as np
from omegaconf import open_dict
from tqdm import trange

import sionna.rt as srt

# import own modules
_repo_name = "ofdma-spectrum-anomalies-simulation"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]

from utils import rt_utils, ofdm_utils, impairment_utils
from utils.data_utils import get_datapath
from utils.datatypes import Transmitter, Jammer, Sample

_datapath = get_datapath(_repo_name)


def generate_sample(cfg, scene, jammer_type, su_hardware):
    """Generate one sample (realization) for the dataset.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing the parameters for the dataset generation.
    scene : sionna.rt.Scene
        Scene object containing the environment.
    jammer_type : str
        The type of the jammer for the sample, e.g., "normal", "deceptive", "sweep", "barrage", or "pilot".
    su_hardware : pd.DataFrame
        Per-SU hardware impairment parameters, fixed for the whole dataset.
        See `impairment_utils.generate_su_hardware_params`.

    Returns
    -------
    sample : Sample
        Sample object containing the information of the generated sample.
    """

    sample = Sample()

    # random number of legitimate transmitters
    num_tx = np.random.randint(cfg.min_tx, cfg.max_tx + 1)

    # assign resources to the legitimate transmitters
    allocated_resources = ofdm_utils.allocate_resources(
        num_tx, num_rb, cfg.n_slots, plot_allocation=False
    )

    # place the legitimate transmitters at random positions
    # and add them to the sample
    for tx_idx in range(num_tx):
        tx_pos = rt_utils.get_random_tx_location(
            cfg.scene_nr, scene_size, tx_height=1.5
        )
        tx = Transmitter(tx_pos, allocated_resources[tx_idx])
        sample.add_transmitter(tx)

        logging.debug(f"TX {tx_idx} position: {tx_pos}")

    # jammer type according to predefined list
    if jammer_type != "normal":
        jammer_pos = rt_utils.get_random_tx_location(
            cfg.scene_nr, scene_size, tx_height=1.5
        )
        # jammer orientation is randomly rotated in the x-y-plane
        # important due to directional antenna pattern
        jammer_orientation = [np.random.uniform(0, 2 * np.pi), 0, 0]
        jammer_power = np.random.choice(
            np.arange(cfg.jam_power.min, cfg.jam_power.max + 1, cfg.jam_power.step)
        )
        jammer = Jammer(
            jammer_pos,
            jammer_orientation,
            jammer_power,
            jammer_type,
            jammer_pattern,
        )
        sample.add_jammer(jammer)

    if cfg.plots.scenario_2D:
        rt_utils.plot_scenario2D(
            cfg.scene_nr, sample, add_device_labels=True, su_coordinates=su_coordinates
        )

    # Execute ray tracing for every transmitter and jammer ----------------------------------------

    # add the devices to the scene
    for tx_idx, tx in enumerate(sample.transmitters):
        stx = srt.Transmitter(
            name=f"tx{tx_idx}", position=np.array(tx.location)
        )  # numpy array for mitsuba compatibility
        scene.add(stx)

    # configure isotropic antennas for legitimate transmitters
    scene = rt_utils.configure_tx_antenna_pattern(scene, "iso")

    if cfg.plots.scenario_rendered:
        # jammers are only added for visualization purposes
        for jam_idx, jam in enumerate(sample.jammers):
            sjam = srt.Transmitter(
                name=f"jam{jam_idx}", position=jam.location, orientation=jam.orientation
            )
            scene.add(sjam)

        scene.render(camera="scene-cam-0")

        # remove the jammers so that they are not simulated in the ray tracing
        for jam_idx in range(len(sample.jammers)):
            scene.remove(f"jam{jam_idx}")

    # execute ray tracing
    paths = p_solver(
        scene=scene,
        max_depth=cfg.sionna.max_depth,
        samples_per_src=int(cfg.sionna.num_rays),
        specular_reflection=True,
        refraction=True,
    )

    # calculate the channel frequency response
    h = paths.cfr(
        fft_freq,
        sampling_frequency=cfg.subcarrier_spacing * cfg.nfft,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )

    # remove the legitimate transmitters from the scene
    for tx_idx in range(len(sample.transmitters)):
        scene.remove(f"tx{tx_idx}")

    # execute ray tracing for the jammer
    if len(sample.jammers) > 0:

        # configure the jammer antenna pattern
        scene = rt_utils.configure_tx_antenna_pattern(scene, jammer_pattern)

        # add the jammer to the scene
        for jam_idx, jam in enumerate(sample.jammers):
            sjam = srt.Transmitter(name=f"jam{jam_idx}", position=jam.location)
            scene.add(sjam)
        paths_jam = p_solver(
            scene=scene,
            max_depth=cfg.sionna.max_depth,
            samples_per_src=int(cfg.sionna.num_rays),
            specular_reflection=True,
            refraction=True,
        )
        h_jam = paths_jam.cfr(
            fft_freq,
            sampling_frequency=cfg.subcarrier_spacing * cfg.nfft,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )

        # combine the CFRs of the authorized transmitters if the jammer is present
        h = np.concatenate((h, h_jam), axis=2)  # axis 2 in the index of the tx

        # remove the jammers from the scene
        for jam_idx, jam in enumerate(sample.jammers):
            scene.remove(f"jam{jam_idx}")

    # Create spectrograms -------------------------------------------------------------------------

    if cfg.plots.frequency_responses:
        rt_utils.plot_frequency_responses(fft_freq, h, len(sample.jammers))

    # use the assigned resources together with the assigned resources to create the spectrograms
    sample = rt_utils.create_spectrograms(sample, cfg, h, su_hardware, noise=True)

    if cfg.plots.all_spectrograms:
        sample.plot_all_spectrograms()

    return sample


def generate_dataset(cfg, scene, su_coordinates):
    """Generate a dataset for the given scene.

    The dataset consists of samples with different legitimate transmitter counts and jamming types.
    The dataset is stored in the datapath specified in the datapath.txt file.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing the parameters for the dataset generation.
    scene : sionna.rt.Scene
        Scene object containing the environment.
    su_coordinates : list
        List of sensing unit coordinates, used only to determine the number
        of SUs for drawing the per-SU hardware impairment parameters.
    """

    # generate which samples are jammed (incl. type) and which are normal
    jammer_types = ["deceptive", "sweep", "barrage", "pilot", "random_hop"]
    no_samples_by_jammer = int(
        cfg.nr_samples * cfg.jam_probability // len(jammer_types)
    )
    no_normal_samples = cfg.nr_samples - no_samples_by_jammer * len(jammer_types)

    sample_type = ["normal"] * no_normal_samples
    for idx_jammer, jammer in enumerate(jammer_types):
        sample_type += [jammer] * no_samples_by_jammer

    samples = []
    batch_idx = 0

    # check if the folder for the dataset exists, if not create it
    if not os.path.exists(os.path.join(_datapath, f"{cfg.dataset_nr}", "custom")):
        os.makedirs(os.path.join(_datapath, f"{cfg.dataset_nr}", "custom"))

    # hardware impairment parameters are a property of the SU, not of the
    # sample (D2 in hardware_impairments.md): drawn once here, held fixed for
    # the whole dataset, and persisted next to the other per-dataset metadata.
    su_hardware = impairment_utils.generate_su_hardware_params(len(su_coordinates), cfg)
    su_hardware.to_csv(os.path.join(_datapath, f"{cfg.dataset_nr}", "su_hardware.csv"))

    for idx_sample in trange(cfg.nr_samples, desc="Generating samples"):

        logging.debug(f"Sample {idx_sample} -----------------------")

        for _ in range(3):
            # very rarely, a ValueError occurs that needs to be caught and
            # is handled by simply trying to generate the sample again
            try:
                samples.append(
                    generate_sample(cfg, scene, sample_type[idx_sample], su_hardware)
                )
                break
            except ValueError:
                print("ValueError occured in sample generation - Regenerating sample.")

        if len(samples) == cfg.batch_size or idx_sample == cfg.nr_samples - 1:
            # store if batch is full or if it is the last sample
            with open(
                os.path.join(
                    _datapath,
                    f"{cfg.dataset_nr}",
                    "custom",
                    f"samples-{batch_idx:04d}.pkl",
                ),
                "wb",
            ) as f:
                pkl.dump(samples, f)

            samples = []
            batch_idx += 1


def parse_arguments():
    """Parse command line arguments.

    Returns
    -------
    args : Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate a dataset for a given scene and configuration."
    )

    parser.add_argument(
        "-d",
        "--dataset-number",
        type=int,
        required=False,
        help="Dataset number to process.",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=False,
        help="Number of samples to generate. If not set, the number of samples from the config file is used.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Initialization ----------------------------------------------------------------------------------

    # load config file which contains the parameters into cfg object
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="dataset_generation")

    if cfg.logging:
        logger = logging.getLogger(__name__)
        # logs are written to file and stored in the logs folder
        if not os.path.exists(os.path.join(_module_path, "logs")):
            os.makedirs(os.path.join(_module_path, "logs"))
        log_dir = os.path.join(_module_path, "logs")
        log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        logging.basicConfig(
            filename=os.path.join(log_dir, log_filename),
            level=logging.DEBUG,
        )

    # parse and handle command line arguments
    args = parse_arguments()

    if args.dataset_number is not None:
        if cfg.dataset_nr != int(args.dataset_number):
            print(
                f"Dataset number {args.dataset_number} is set via command line argument."
            )
            cfg.dataset_nr = int(args.dataset_number)

    if args.num_samples is not None:
        cfg.nr_samples = args.num_samples

    logging.info(f"Generating dataset with {cfg.nr_samples} samples.")

    # Load and configure the scene --------------------------------------------------------------------

    scene_path = os.path.join(
        _module_path, "scenes", f"scene{cfg.scene_nr}", "scene.xml"
    )

    su_coordinates = rt_utils.get_su_coordinates(_module_path)

    # loading the scene and add the sensing units
    # note: only one scene object can be loaded, to represent different antenna patterns,
    # two simulation runs need to be executed
    scene = rt_utils.init_scene(scene_path, cfg.f_c, su_coordinates)

    scene_size = rt_utils.get_scene_size(scene)

    # create another scene for the jammer to be able to simulate antenna patterns for the jammer that
    # differ from the antenna patterns of legitimate transmitters
    if cfg.jam_pattern == "iso":
        jammer_pattern = "iso"
    elif cfg.jam_pattern == "directional":
        jammer_pattern = "tr38901"
    else:
        raise ValueError(
            'Unknown jamming pattern. Please choose either "iso" or "directional".'
        )

    p_solver = srt.PathSolver()

    # Compute some parameters which result from the configuration -------------------------------------

    # number of available RBs (12 subcarriers per RB)
    if cfg.bandwidth == 20e6 and cfg.subcarrier_spacing == 15e3:
        num_rb = 110  # this is the common configuration, not the theoretically calculated 111
    else:
        num_rb = int(cfg.bandwidth / cfg.subcarrier_spacing) // cfg.subcarriers_per_rb

    # some initialization with respect to the subcarriers of the FFT that are really in use
    num_subcarriers = int(num_rb * cfg.subcarriers_per_rb)
    idx_first_sc = (
        cfg.nfft - num_subcarriers
    ) // 2  # index of the lowest subcarrier in the FFT that is potentially in use
    symbol_duration = cfg.slot_length / cfg.symbols_per_slot
    fft_freq = (
        srt.utils.subcarrier_frequencies(cfg.nfft, cfg.subcarrier_spacing) + cfg.f_c
    )

    p_noise_dbm = float(
        rt_utils.calc_noise_power_dbm(
            cfg.nfft * cfg.subcarrier_spacing,
            cfg.su_noise_figure,
            cfg.additional_impairments,
        )
    )

    # add previous parameters to the cfg object
    with open_dict(cfg):
        cfg.num_rb = num_rb
        cfg.num_subcarriers = num_subcarriers
        cfg.idx_first_sc = idx_first_sc
        cfg.p_noise_dbm = p_noise_dbm

    generate_dataset(cfg, scene, su_coordinates)
