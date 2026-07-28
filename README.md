# generative-audio-source-models

Code from the PhD thesis *"Modèles génératifs pour la classification et la
séparation de sources sonores en temps-réel"* (Generative models for
real-time audio source classification and separation), Maxime Baelde,
Université de Lille, 2019 (dir. Christophe Biernacki, Raphaël Greff /
A-Volute).

- Thesis: [theses.fr/2019LILUI058](https://theses.fr/2019LILUI058) · DOI
  [10.70675/57a0a0d8ze25cz4b79z9bf4z04dde72f8de1](https://doi.org/10.70675/57a0a0d8ze25cz4b79z9bf4z04dde72f8de1)

## What runs and what doesn't

`matlab/` is the code as written in 2019, on a since-expired MATLAB license.
It has not been re-executed since, and the original industrial datasets
(A-Volute) that produced the thesis' published numbers no longer exist. This
is a reference archive, not a runnable pipeline: expect no data files, no
guarantee the toolboxes it once depended on still resolve, and no compiled
MEX binaries (they were dropped: platform-specific artifacts of a licence
nobody here still has).

`python/` is a rewrite of both contributions from the thesis' equations, not
a translation of the MATLAB. It is tested (`pytest`, `mypy --strict`) and
runs on public audio. The two contributions land differently on
reproduction, and the difference is worth stating plainly:

- **RARE reproduces.** On ESC-10 at the thesis' parameters, the demo measures
  0.592 (0.023) correct classification against the published 0.647 (0.029),
  about 1.5 combined standard deviations. Getting there required recovering
  three protocol details the manuscript does not spell out; they are
  documented in [`python/README.md`](python/README.md).
- **RASE does not, and will not.** The thesis' separation numbers were
  measured on A-Volute data with a different task, corpus and metric. The
  MUSDB18 demo only shows that DM-GMM and Def-MAP beat a naive "mixture as
  estimate" baseline. No number it prints is a reproduction of chapter 3,
  and none should be read as one.

## The two contributions

| Method | Thesis chapter | What it does |
|---|---|---|
| **RARE** | ch. 2 | Real-time mono/polyphonic audio classification: normalized power-spectrum features, a Bhattacharyya-kernel-based classifier, and a prototype hierarchy + thresholding scheme to keep inference real-time. |
| **RASE** | ch. 3 | Audio source separation: **DM-GMM** (missing-data GMM) and **Def-MAP** (optimal deformation MAP) proposals. |

## Repository layout

```
matlab/
├── classification/   # RARE (ch. 2)
│   ├── Core/, Dictionary_creation/, Identification_procedure/
│   ├── pipeline/, benchmarks/, scripts/, tests/
│   └── demo/rare_program/   # standalone demo application
├── separation/        # RASE (ch. 3)
│   ├── Benchmark/, toolbox plca/
│   └── *.m             # separation experiments (NMF, PLCA/PLCS, masking, deformation)
├── common/             # shared code used by both
│   ├── Statistics/, Signal_models/, Utils/
│   ├── AudioDescriptors/   # feature extraction
│   └── PLCA/               # PLCA/PLCS implementation used by both classification and separation
├── experiments/        # figure/table generation scripts (analysis, plotting, misc)
└── startup.m

python/                 # rewrite from the equations (see python/README.md)
├── src/gasm/
│   ├── common/         # power spectra, multinomial kernel, Hellinger
│   ├── rare/           # classifier (mono + polyphonic), prototype reduction, dataset prep
│   └── rase/           # dmgmm.py (DM-GMM), defmap.py (Def-MAP)
├── examples/           # ESC-50 classification, MUSDB18 separation
└── tests/
```

## Publications

- M. Baelde, C. Biernacki, R. Greff. ["A mixture model-based real-time audio
  sources classification method"](https://doi.org/10.1109/ICASSP.2017.7952592),
  ICASSP 2017, pp. 2427-2431.
- M. Baelde, C. Biernacki, R. Greff. "Classification de signaux audio en
  temps-réel par un modèle de mélanges d'histogrammes", 49èmes Journées de
  Statistique (JdS), Avignon, 2017.
  ([HAL](https://hal.science/hal-01592496))
- M. Baelde, C. Biernacki. ["Real-Time monophonic and polyphonic audio
  classification from power spectra"](https://doi.org/10.1016/j.patcog.2019.03.017),
  Pattern Recognition 92 (2019), pp. 82-92.

## License

All code in this repo is Maxime Baelde's own and released under
[BSD-3-Clause](LICENSE). No third-party code is vendored here; see
[THIRD_PARTY.md](THIRD_PARTY.md) for the two files that were considered and
dropped over unclear/incompatible license terms.

## Companion package

The mixture models of the thesis' appendices are maintained separately, with a
scikit-learn-compatible API, in
[**nongaussian-mixtures**](https://github.com/mbaelde/nongaussian-mixtures)
([PyPI](https://pypi.org/project/nongaussian-mixtures/)):
`DirichletMixture` (appendix B.2, EM + damped Newton),
`BayesianDirichletMixture` (variational), `BetaMixture`, and
`BinnedGaussianMixture` (appendix B.1, and diagonal covariances in place of the
2-D numerical quadrature of `gmm2d_binned.m`). All four pass `check_estimator`.

`python/` depends on it rather than carrying its own copies; the 2019 originals
stay in `matlab/common/Statistics/`.
