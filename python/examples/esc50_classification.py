"""RARE demo: monophonic classification on ESC-10, at the thesis' parameters.

This reproduces the protocol behind table:q of chapter2RTASC.tex, which reports
**64.7 (2.9)** correct classification on ESC-10 at quantization=257, as closely
as the published code allows:

- ``create_dataset.m``'s pipeline including its additive noise (gasm.rare.dataset)
- frame_length = shift_length = 512, freqbin = quantization = 257 (T, D, n_fft, q)
- decision over m = 39 consecutive frames (main_monophonic.m l.128)
- energy gate at -60 dB (param.threshold)
- ``kernel="cross_entropy"``, which is what the MATLAB actually computed: the
  rounding is commented out in both compute_feature.m and identification.m
- a random stratified 80/20 split over sounds, as split_dataset_folds.m does

Measured over 3 draws (macro-averaged per-class accuracy):

    0.592 (0.023)   this configuration, what this script prints
    0.579 (0.028)   without create_dataset.m's additive noise
    0.521 (0.031)   using ESC-50's own curated 5 folds instead of a random split
    0.290           with kernel="multinomial" (the manuscript as written)

0.592 (0.023) against the published 0.647 (0.029) is about 1.5 combined
standard deviations, on different data (the thesis' A-Volute training set no
longer exists), so it agrees about as well as it can. The 0.521 line is
worth noting: a
random split over sounds scores ~6 points above ESC-50's curated folds, and
that is not a source-recording leak (grouping the split by parent Freesound
recording gives 0.582 (0.012), the same thing).

Runtime is a few minutes on several cores: 171600 frames, and the scoring
matmul is n_test x n_train x freqbin.

Usage: python esc50_classification.py /path/to/ESC-50-master
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from gasm.rare.classifier import RareClassifier
from gasm.rare.dataset import prepare_clip

FRAME_LENGTH = 512  # param.T
SHIFT_LENGTH = 512  # param.D
N_FREQ_BINS = 257  # param.n_fft
QUANTIZATION = 257  # param.q, the column table:q reports 0.647 for
DECISION_FRAMES = 39  # main_monophonic.m l.128: m = 39
TEST_RATIO = 0.2  # split_dataset_folds(idx_classes, 0.8)
WINDOWS_PER_BATCH = 5  # caps the scoring matmul at ~200 queries x n_train
N_DRAWS = 3  # the split is random, so one draw is not a measurement
THESIS_GCR = 0.647


def _load(esc50_root: Path, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Per-frame (spectrum, class index, clip index) plus the energy-gate mask."""
    with open(esc50_root / "meta" / "esc50.csv", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["esc10"] == "True"]
    classes = sorted({row["category"] for row in rows})

    spectra, keep, labels, clips = [], [], [], []
    for clip_index, row in enumerate(rows):
        signal, _fs = sf.read(esc50_root / "audio" / row["filename"], dtype="float64")
        if signal.ndim == 2:  # create_dataset.m l.33-35
            signal = signal.mean(axis=1)
        clip_spectra, clip_keep = prepare_clip(
            signal, FRAME_LENGTH, SHIFT_LENGTH, N_FREQ_BINS, rng
        )
        spectra.append(clip_spectra)
        keep.append(clip_keep)
        labels.append(np.full(len(clip_spectra), classes.index(row["category"])))
        clips.append(np.full(len(clip_spectra), clip_index))
    return tuple(np.concatenate(array) for array in (spectra, keep, labels, clips))


def _holdout_mask(clips: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stratified TEST_RATIO holdout over whole sounds, per class."""
    is_test = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        class_clips = np.unique(clips[labels == label])
        n_held = max(1, round(TEST_RATIO * len(class_clips)))
        is_test |= np.isin(clips, rng.permutation(class_clips)[:n_held])
    return is_test


def _predict_windows(classifier: RareClassifier, spectra: np.ndarray) -> np.ndarray:
    """One label per DECISION_FRAMES consecutive frames, by summing the per-frame
    log-scores. decision_function is only defined up to a per-frame constant, but
    that constant is common to every class so it cannot change the argmax."""
    n_windows = len(spectra) // DECISION_FRAMES
    windows = spectra[: n_windows * DECISION_FRAMES].reshape(n_windows, DECISION_FRAMES, -1)
    predictions = []
    for start in range(0, n_windows, WINDOWS_PER_BATCH):
        batch = windows[start : start + WINDOWS_PER_BATCH]
        scores = classifier.decision_function(batch.reshape(-1, batch.shape[-1]))
        predictions.append(scores.reshape(len(batch), DECISION_FRAMES, -1).sum(axis=1).argmax(1))
    return np.concatenate(predictions)


def _one_draw(
    spectra: np.ndarray,
    keep: np.ndarray,
    labels: np.ndarray,
    clips: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> float:
    is_test = _holdout_mask(clips, labels, rng)
    train = np.flatnonzero(keep & ~is_test)
    test = np.flatnonzero(keep & is_test)
    classifier = RareClassifier(quantization=QUANTIZATION, kernel="cross_entropy")
    classifier.fit(spectra[train], labels[train])

    # Windows must not straddle two classes, so score one class at a time.
    accuracies = []
    for label in range(n_classes):
        class_test = test[labels[test] == label]
        if len(class_test) < DECISION_FRAMES:
            continue
        predictions = _predict_windows(classifier, spectra[class_test])
        accuracies.append(float(np.mean(predictions == label)))
    return float(np.mean(accuracies))


def main() -> None:
    rng = np.random.default_rng(0)
    spectra, keep, labels, clips = _load(Path(sys.argv[1]), rng)
    n_classes = int(labels.max()) + 1
    print(f"{len(labels)} frames, {100 * (1 - keep.mean()):.1f}% below the energy gate")

    # The noise realization is drawn once, so this spread is the split's alone.
    scores = []
    for draw in range(N_DRAWS):
        scores.append(_one_draw(spectra, keep, labels, clips, n_classes, rng))
        print(f"draw {draw + 1}/{N_DRAWS}: {scores[-1]:.3f}", flush=True)

    gcr, spread = float(np.mean(scores)), float(np.std(scores))
    chance = 1.0 / n_classes
    print(
        f"correct classification: {gcr:.3f} ({spread:.3f}) "
        f"(chance {chance:.3f}, thesis {THESIS_GCR:.3f})"
    )
    assert gcr > chance, "RARE should beat chance on held-out ESC-10 sounds"


if __name__ == "__main__":
    main()
