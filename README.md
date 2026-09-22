# CaloChallengeMetrics

The CaloChallenge evaluation code is maintained in [TwoSampleTests/CaloChallenge](https://github.com/TwoSampleTests/CaloChallenge), a modified fork of [CaloChallenge/homepage](https://github.com/CaloChallenge/homepage) used in the CaloChallenge **“re-evaluation”** project.

This repository includes it as the `CaloChallenge` submodule. The evaluation scripts and notebooks are in `CaloChallenge/code/`.

Clone with the submodule:

```bash
git clone --recurse-submodules https://github.com/TwoSampleTests/CaloChallengeMetrics.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Local datasets, reference caches, and evaluation results belong in `CaloChallenge/code/dataset_copy/`, `CaloChallenge/code/source/`, and `CaloChallenge/code/evaluation_results/`. They are excluded from version control and are not downloaded by cloning the repository. See [the evaluation instructions](CaloChallenge/code/LOCAL_CHANGES.md) for configuration and usage.
