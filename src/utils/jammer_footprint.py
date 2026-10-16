"""Inference of the jammer footprint from the published dataset.

The resource elements (REs) occupied by the jammer are not part of the dataset
(only the occupancy, i.e., their share, is contained in the labels). The
functions in this module infer the footprint from the spectrograms and the
resource allocation images, exploiting the known structure of each jammer type.
They are used to highlight the jammed regions in example spectrograms, see
`src/plot_figures.py`.

All spectrograms are expected in dB with shape (num_sc, num_symbols), and the
resource allocation images with the same shape (index of the legitimate
transmitter per RE, 0: unallocated). Boxes are returned as half-open index
ranges (row0, row1, col0, col1), i.e., (SC, SC, symbol, symbol).
"""

__docformat__ = "numpy"

import numpy as np
from scipy import ndimage


def to_blocks(arr, sc_per_rb, sym_per_slot):
    """Reshape (num_sc, num_symbols) to (num_rb, num_slots, REs per RB and slot)."""
    num_sc, num_sym = arr.shape
    num_rb, num_slots = num_sc // sc_per_rb, num_sym // sym_per_slot
    return (
        arr.reshape(num_rb, sc_per_rb, num_slots, sym_per_slot)
        .transpose(0, 2, 1, 3)
        .reshape(num_rb, num_slots, -1)
    )


def regions_to_boxes(mask):
    """Bounding boxes of the connected regions of a mask.

    Gaps of one SC (comb of the pilot jammer) are bridged.
    """
    lbl, _ = ndimage.label(ndimage.binary_dilation(mask, structure=np.ones((3, 1))))
    return [
        (s[0].start, s[0].stop, s[1].start, s[1].stop) for s in ndimage.find_objects(lbl)
    ]


def infer_pilot_mask(spec, alloc, sym_per_slot, pilot_symbols):
    """Infer the REs occupied by the pilot jammer.

    The pilot jammer targets the pilot symbols of one legitimate transmitter
    (every second SC across its band, in each of its active slots). The target
    is the transmitter whose potential pilot REs show the largest excess over
    the neighbouring symbols.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram in dB.
    alloc : np.ndarray
        Resource allocation image.
    sym_per_slot : int
        Number of OFDM symbols per slot.
    pilot_symbols : list
        Symbol indexes of the pilots within a slot.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of the jammed REs.
    """
    best_score, best_mask = -np.inf, None
    for tx in np.unique(alloc[alloc > 0]):
        rows = np.where((alloc == tx).any(axis=1))[0]
        slots = np.unique(np.where(alloc == tx)[1] // sym_per_slot)
        mask = np.zeros(alloc.shape, dtype=bool)
        for slot in slots:
            for pilot_symbol in pilot_symbols:
                mask[rows.min() : rows.max() + 1 : 2, slot * sym_per_slot + pilot_symbol] = True
        neighbours = np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
        score = np.median(spec[mask]) - np.median(spec[neighbours])
        if score > best_score:
            best_score, best_mask = score, mask
    return best_mask


def infer_deceptive_mask(spec, alloc, sc_per_rb, sym_per_slot, thr_db=3.0):
    """Infer the REs occupied by the deceptive jammer.

    The deceptive jammer occupies a contiguous RB range in a subset of the
    slots. Each (RB, slot) block is compared to the noise floor (unallocated) or
    to the weakest slot of the same transmitter on this RB (allocated, the
    channel is static).

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram in dB.
    alloc : np.ndarray
        Resource allocation image.
    sc_per_rb : int
        Number of SCs per RB.
    sym_per_slot : int
        Number of OFDM symbols per slot.
    thr_db : float
        Minimum excess of a block over its reference to be flagged as jammed.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of the jammed REs.
    """
    # median: robust against the DC offset of the SU
    blocks = np.median(to_blocks(spec, sc_per_rb, sym_per_slot), axis=-1)
    tx_of_block = np.max(to_blocks(alloc, sc_per_rb, sym_per_slot), axis=-1)
    ref = np.full(blocks.shape, np.median(blocks[tx_of_block == 0]))
    for rb in range(blocks.shape[0]):
        for tx in np.unique(tx_of_block[rb][tx_of_block[rb] > 0]):
            same_tx = tx_of_block[rb] == tx
            ref[rb, same_tx] = blocks[rb, same_tx].min()
    flag = (blocks - ref) > thr_db

    # fit the RB range (maximum sum segment) and the active slots alternately
    active_slots = flag.any(axis=0)
    for _ in range(3):
        score = (flag[:, active_slots] - 0.5).sum(axis=1)
        best, cur, rb_low, rb_high, start = -np.inf, 0.0, 0, 0, 0
        for rb, val in enumerate(score):
            cur, start = (val, rb) if cur <= 0 else (cur + val, start)
            if cur > best:
                best, rb_low, rb_high = cur, start, rb + 1
        active_slots = flag[rb_low:rb_high].mean(axis=0) > 0.5

    mask = np.zeros_like(flag)
    mask[rb_low:rb_high, active_slots] = True
    return np.repeat(np.repeat(mask, sc_per_rb, axis=0), sym_per_slot, axis=1)


def infer_sweep_box(spec, sc_per_rb, sym_per_slot, thr_db=12.0):
    """Infer the box enclosing the swept band.

    Legitimate signals are constant over a slot, the sweep is not: REs exceeding
    the median of their SC within the slot are flagged. Vertical lines
    (legitimate transmitters switching at a slot boundary with residual timing
    offset of the SU) are discarded.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram in dB.
    sc_per_rb : int
        Number of SCs per RB.
    sym_per_slot : int
        Number of OFDM symbols per slot.
    thr_db : float
        Minimum excess of an RE over the median of its SC within the slot.

    Returns
    -------
    boxes : list
        A single box (row0, row1, col0, col1), empty if nothing is detected.
    """
    s = spec.reshape(spec.shape[0], -1, sym_per_slot)
    residual = (s - np.median(s, axis=-1, keepdims=True)).reshape(spec.shape)
    flag = ndimage.binary_closing(residual > thr_db, structure=np.ones((3, 3)))
    lbl, _ = ndimage.label(flag)
    comps = []
    for k, sl in enumerate(ndimage.find_objects(lbl)):
        n_rows, n_cols = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if n_cols <= 2 and n_rows > 2 * sc_per_rb:
            continue
        comps.append((sl, np.sum(lbl[sl] == k + 1)))
    if not comps:
        return []
    # drop spurious detections
    largest = max(size for _, size in comps)
    comps = [sl for sl, size in comps if size >= 0.2 * largest]
    return [
        (
            min(sl[0].start for sl in comps),
            max(sl[0].stop for sl in comps),
            min(sl[1].start for sl in comps),
            max(sl[1].stop for sl in comps),
        )
    ]


def infer_tone_boxes(
    spec, thr_db=12.0, win_sc=9, min_cols=2, dc_guard_sc=3, thr_grow_db=(3.0, 10.0)
):
    """Infer one box per burst of the random hopping tone jammer.

    A tone is narrower than an RB and stands out against the median of the
    neighbouring SCs, while legitimate signals (>= 1 RB wide) do not. The SCs
    around DC are excluded, since the DC offset of the SU looks like a tone.
    The on/off switching spreads the energy of the first and last symbol of a
    burst over more SCs, so these are added in a second step along the tone's
    SC: if the symbol still exceeds the neighbouring SCs and the SC's median.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram in dB.
    thr_db : float
        Minimum excess over the neighbouring SCs to detect a burst.
    win_sc : int
        Number of SCs of the median filter along frequency.
    min_cols : int
        Minimum number of symbols of a burst.
    dc_guard_sc : int
        Number of SCs on each side of DC that are excluded.
    thr_grow_db : tuple
        Minimum excess over the neighbouring SCs and over the median of the SC
        to extend a burst by one symbol.

    Returns
    -------
    boxes : list
        Boxes (row0, row1, col0, col1).
    """
    residual = spec - ndimage.median_filter(spec, size=(win_sc, 1), mode="nearest")
    flag = ndimage.binary_closing(residual > thr_db, structure=np.ones((3, 3)))
    dc = spec.shape[0] // 2
    flag[dc - dc_guard_sc : dc + dc_guard_sc + 1] = False
    lbl, _ = ndimage.label(flag)
    boxes = []
    for sl in ndimage.find_objects(lbl):
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if c1 - c0 < min_cols:
            continue
        row = r0 + np.argmax(residual[r0:r1, c0:c1].max(axis=1))
        excess_row = spec[row] - np.median(spec[row])

        def is_tone(col):
            return residual[row, col] > thr_grow_db[0] and excess_row[col] > thr_grow_db[1]

        while c0 > 0 and is_tone(c0 - 1):
            c0 -= 1
        while c1 < spec.shape[1] and is_tone(c1):
            c1 += 1
        boxes.append((r0, r1, c0, c1))
    return boxes


def infer_jammer_boxes(
    jammer_type, spec, alloc, sc_per_rb, sym_per_slot, pilot_symbols, occupancy=None
):
    """Infer boxes around the REs occupied by the jammer.

    The footprint of the pilot and deceptive jammer is identical for all SUs and
    aligned to the resource grid, hence it is best inferred on the SU that sees
    the jammer strongest (highest local JSNR). The sweep and tone boxes are
    inferred on the SU the boxes are drawn for, since the spectral splatter of
    very strong narrowband jammers enlarges them.

    Parameters
    ----------
    jammer_type : str
        Jammer type as in the labels.
    spec : np.ndarray
        Spectrogram in dB.
    alloc : np.ndarray
        Resource allocation image.
    sc_per_rb : int
        Number of SCs per RB.
    sym_per_slot : int
        Number of OFDM symbols per slot.
    pilot_symbols : list
        Symbol indexes of the pilots within a slot.
    occupancy : float, optional
        Jammer occupancy from the labels. If given, the inferred pilot and
        deceptive footprints are checked against it.

    Returns
    -------
    boxes : list
        Boxes (row0, row1, col0, col1). Empty for "no jammer" and for the
        barrage jammer, which occupies the whole grid.
    """
    if jammer_type in ("no jammer", "barrage"):
        return []
    if jammer_type in ("pilot", "deceptive"):
        if jammer_type == "pilot":
            mask = infer_pilot_mask(spec, alloc, sym_per_slot, pilot_symbols)
        else:
            mask = infer_deceptive_mask(spec, alloc, sc_per_rb, sym_per_slot)
        if occupancy is not None and abs(mask.mean() / occupancy - 1) > 0.05:
            print(
                f"Warning: inferred {jammer_type} footprint deviates from the labelled "
                f"occupancy (ratio {mask.mean() / occupancy:.2f})"
            )
        return regions_to_boxes(mask)
    if jammer_type == "sweep":
        return infer_sweep_box(spec, sc_per_rb, sym_per_slot)
    if jammer_type == "random_hop":
        return infer_tone_boxes(spec)
    raise ValueError(f"Unknown jammer type: {jammer_type}")
