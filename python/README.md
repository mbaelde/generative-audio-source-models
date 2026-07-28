# gasm

Python rewrite of RARE (ch. 2) and RASE (ch. 3) from the thesis' equations,
not a translation of `matlab/`. See the [repo README](../README.md) for
context and the [thesis](https://theses.fr/2019LILUI058) for the equations
referenced in each module's docstring.

## Install

```
pip install -e .[dev]
```

## Tests

```
pytest
mypy --strict
```

## Demos

`examples/esc50_classification.py` and `examples/musdb18_separation.py`
run on public audio, since the thesis' own A-Volute training data no
longer exists. The classification demo does reproduce the published score
to within its uncertainty (see below); the separation demo only checks that
`DM-GMM` and `Def-MAP` beat a naive "mixture as estimate" baseline.

## Reproducing the thesis' RARE score

`chapter2RTASC.tex`, `table:q` reports **64.7 (2.9)** correct classification
on ESC-10 at `quantization=257`. `esc50_classification.py` measures
**0.592 (0.023)** over its 3 draws, about 1.5 combined standard deviations
away.

Getting there needs three things that the manuscript does not spell out, each
measured separately on ESC-10 at the thesis' parameters:

| | correct classification |
|---|---|
| `kernel="multinomial"`, ESC-50's curated folds, no additive noise | 0.290 |
| `kernel="cross_entropy"` (what the MATLAB ran) | 0.521 (0.031) |
| \+ random 80/20 split over sounds (what `split_dataset_folds.m` does) | 0.579 (0.028) |
| \+ `create_dataset.m`'s additive noise (`gasm.rare.dataset`) | **0.592 (0.023)** |
| thesis, `table:q` | 0.647 (0.029) |

1. **The kernel.** The published code never quantizes: the rounding is
   commented out in `compute_feature.m` l.16-18 and `identification.m` l.51,
   so it scores a cross-entropy KDE rather than the multinomial PMF of
   eq. def_xi. The two agree in the `quantization -> infinity` limit, which is
   the convergence the manuscript itself invokes for `table:q`'s plateau, and
   `table:q` reports the same 64.7 at 1e4 and 1e5. `RareClassifier` defaults to
   the manuscript's kernel and takes `kernel="cross_entropy"` for the other.
2. **The split.** A random stratified 80/20 over sounds scores ~6 points above
   ESC-50's curated 5 folds. This is *not* a parent-recording leak: grouping the
   split by parent Freesound recording gives 0.582 (0.012), the same thing.
3. **The additive noise.** `create_dataset.m` l.44-45 adds white noise at 1% of
   each signal's peak to every signal, train and test. It lifts the noise floor
   ~34 dB above the -60 dB energy gate (1.5% of frames gated instead of 23.8%)
   and leaves no zero bin in the spectra. `gasm.rare.dataset.prepare_clip`
   reproduces it, quirks included.

Two things the MATLAB gets away with, checked rather than assumed:
`main_monophonic.m` l.120 uses an unnormalized frame count as `prior_g` and adds
it raw to log-likelihoods, but every 5 s clip yields the same frame count
(3432 per class here) so the constant cancels exactly; and the energy gate NaNs
its rows without removing them, so it never enters that count either.

Datasets are large enough that they're best run outside a local machine
(the datasets themselves are **not** committed to this repo):

```
pip install -e .[demos]

# ESC-50: https://github.com/karolpiczak/ESC-50 (zip download, ~600 MB)
python examples/esc50_classification.py /path/to/ESC-50-master

# MUSDB18 7s preview: auto-downloaded by musdb on first run (~140 MB)
MUSDB_ROOT=/path/to/musdb18-7s python examples/musdb18_separation.py
```

`musdb` depends on `ffmpeg`/`ffprobe` (not a Python dependency, install it
separately, e.g. `apt-get install ffmpeg`).
