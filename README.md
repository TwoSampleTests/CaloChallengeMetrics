# CaloChallengeMetrics

The CaloChallenge evaluation code is maintained in [TwoSampleTests/CaloChallenge](https://github.com/TwoSampleTests/CaloChallenge), a modified fork of [CaloChallenge/homepage](https://github.com/CaloChallenge/homepage) used in the CaloChallenge **“re-evaluation”** project.

The local directories `CaloChallenge/`, `GMetrics/`, `JetNetMetrics/`, and `NysMMD/` are excluded from this repository by `.gitignore`. Their contents are not tracked here, and none is registered as a Git submodule. Any independent Git repositories in these directories retain their own remotes and must be updated separately.

To clone this repository and obtain the evaluation code:

```bash
git clone https://github.com/TwoSampleTests/CaloChallengeMetrics.git
cd CaloChallengeMetrics
git clone https://github.com/TwoSampleTests/CaloChallenge.git CaloChallenge
```

If the local `CaloChallenge/` clone already exists, keep it and manage it directly with `git -C CaloChallenge ...`. Obtain any other required local dependencies separately.

The evaluation scripts and notebooks are in `CaloChallenge/code/`. Local datasets, reference caches, and evaluation results belong in `CaloChallenge/code/dataset_copy/`, `CaloChallenge/code/source/`, and `CaloChallenge/code/evaluation_results/`. They are excluded from version control and are not downloaded by cloning the repository. See [the evaluation instructions](https://github.com/TwoSampleTests/CaloChallenge/blob/main/code/LOCAL_CHANGES.md) for configuration and usage.
