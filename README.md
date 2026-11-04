# Language models as medical world models

Reproducibility materials for **Action fidelity in compact language models used as medical world models**.

This study tests whether frozen compact general-purpose language models can simulate physiological responses to specified treatments. It uses resettable glucose and anesthesia simulators, paired alternative actions and twelve-step self-fed rollouts, with a separate factual VitalDB evaluation. It does not evaluate treatment selection or establish clinical benefit.

## Main result

Across all 54 held-out glucose dose pairs, Qwen3-8B's paired-effect error exceeded zero-effect persistence by **16.1 mg/dL (95% cluster CI 10.4–22.4)**. Better absolute forecasts, valid outputs and verbal pharmacological knowledge did not by themselves establish treatment-response fidelity. The separately specified output-interface and history-retrieval analyses are exploratory and reuse the original cases.

## Access and reproduce

- [Reproduction instructions](repo/README.md): environments, model downloads, benchmarks, inference and analysis.
- [Frozen study protocol](repo/docs/protocol.md): endpoints, feedback modes, statistical units and scope.
- [Public results check](repo/scripts/41_verify_public_results.py): verify artifacts and independently recompute the primary result without model inference or clinical downloads.
- [Aggregate results](data/results/) and [synthetic benchmarks](data/simulated/).
- [Simulator-only outputs](data/releases/synthetic_outputs.zip): raw responses, parsed predictions and numerical controls.
- [Original provenance](data/provenance/original/): internal freezes, model revisions and amendments.
- [Data dictionary](repo/docs/data_dictionary.md) and [license notice](repo/LICENSE_NOTICE.md).

The directory layout is intentional: run commands from this repository's top level, with source under repo/, artifacts under data/, optional environments under env/, and regenerated figures/tables under paper/. Manuscript files are supplied separately to the journal.

## Quick verification

Use Python 3.12. For a full analysis environment, follow repo/README.md. A lightweight primary-result check needs only NumPy:

~~~bash
python3.12 -m venv env/.venv
env/.venv/bin/python -m pip install numpy==1.26.4
env/.venv/bin/python repo/scripts/41_verify_public_results.py
~~~

No raw VitalDB recordings, patient-level clinical inputs/outputs, model weights or fonts are included. Clinical replay can be reconstructed using the acquisition scripts under VitalDB's terms. The published synthetic-output archive is intentionally not a complete clinical run. Perform new inference in a separate checkout to preserve the original aggregates.

Authors: Zaifu Zhan, Min Zeng, Kai Yu, Xiaoyi Chen, Mengyuan Cui and Rui Zhang. Correspondence: Rui Zhang (ruizhang@umn.edu). Original evaluation code is MIT licensed; third-party assets retain their own terms.

## Manuscript revision analyses

Post hoc analyses of the original saved outputs assess reference-effect magnitude, all twelve rollout horizons, and first-object point readability. They do not replace original outcomes or regenerate missing continuations. See [analysis definitions](repo/docs/revision_analysis.md), [revision results](data/results/revision_v1/) and the reproduction commands in [repo/README.md](repo/README.md).

Author contributions: Zaifu Zhan and Rui Zhang conceived the study. Zaifu Zhan led the study and prepared the datasets. Zaifu Zhan, Min Zeng, Kai Yu, Xiaoyi Chen and Mengyuan Cui performed experimental validation. Rui Zhang supervised the research.
