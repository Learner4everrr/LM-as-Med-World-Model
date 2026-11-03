# Medical world model evaluation

This project evaluates frozen general-purpose LLMs as patient-dynamics engines. It does not train a new medical world model or evaluate treatment selection. Controlled simulator A/B truth and observational factual replay are reported separately.

The public repository is https://github.com/Learner4everrr/LM-as-Med-World-Model.

The project root contains `env/`, `repo/`, `data/`, and `paper/`. Run all commands from the project root. Preserve existing data and environments.

## Study
- Frozen general LLMs: Qwen3-8B (non-thinking), Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct.
- Mechanistic environments: simglucose 0.2.11 and Python Anesthesia Simulator 1.0.0.
- Observational environment: public deidentified VitalDB, local candidate-model inference.
- Patient/profile-level development/test separation; 12-step true self-fed and teacher-forced protocols; state/action information ablation.
- Primary endpoint and contrast: [protocol](docs/protocol.md). Clinical review is AI-assisted only, no clinician validation is claimed.

## Environments
Python 3.12. Analysis dependencies are pinned in `requirements-analysis.lock`; GPU inference dependencies are in `requirements-inference.lock`. Installed environments are separate because the simulator requires NumPy 1.x while the inference environment has independent dependencies.

```bash
python3.12 -m venv env/.venv
env/.venv/bin/python -m pip install -r repo/requirements-analysis.lock
python3.12 -m venv env/inference
env/inference/bin/python -m pip install -r repo/requirements-inference.lock
```

The recorded inference used four NVIDIA A100 40 GB GPUs, one model per GPU. Set all Hugging Face/model caches inside `data/`. The downloader and runner do this explicitly. No remote model code is trusted and no paid API is used.

## Reproduction order
```bash
env/.venv/bin/python repo/scripts/verify.py
env/.venv/bin/python repo/scripts/02_generate_benchmark.py
env/.venv/bin/python repo/scripts/08_vitaldb.py
env/.venv/bin/python repo/scripts/download_models.py
env/.venv/bin/python repo/scripts/19_verify_models.py
env/.venv/bin/python repo/scripts/04_baselines.py
env/.venv/bin/python repo/scripts/10_audit.py
env/.venv/bin/python repo/scripts/03_freeze.py
env/.venv/bin/python repo/scripts/13_launch.py
env/.venv/bin/python repo/scripts/12_verify_study.py
env/.venv/bin/python repo/scripts/06_evaluate.py
env/.venv/bin/python repo/src/medwm_eval/figure_style.py
env/.venv/bin/python repo/scripts/07_figures.py
env/.venv/bin/python repo/scripts/11_tables.py
```

A fresh reproduction creates a new timestamped freeze after running development checks; the existing study freeze is never overwritten. The inference runner refuses changed frozen inputs/code. Raw text, exact prompts, model revisions and predictions remain in `data/runs/study_v1/test/`. The local protocol is not a publicly registered preregistration.

The VitalDB loader retrieves public research data under the recorded dataset conditions. Raw records and weights are not part of a manuscript/code release. Its legacy CSV API could change; the original downloaded files and hashes are retained locally. Read `data/results/vitaldb_audit.json`.

## Artifacts
- `data/manifests/study_freeze.json`: pre-test LLM inference specification and hashes.
- `data/results/final_data_audit.json`: split, pairing, replay and quantization checks.
- `data/results/study_v1/`: analysis tables, point/case metrics and final integrity checks.
- `paper/`: output directories for regenerated figures and tables; manuscript files are supplied separately to the journal.
- `repo/docs/novelty_matrix.md`: targeted primary-source comparison; no priority claim.
- `repo/docs/ai_review.md`: review scope and limitations.

The target journal is npj Digital Medicine.

## Separate exploratory extension
The falling-glucose extension was added after the main freeze, without changing the primary study. Read docs/stress_extension.md. To regenerate in a fresh workspace, run 14_stress_data.py, then 16_stress_baselines.py after the original development-trained controls exist. Use 17_queue_stress.py to run the frozen models once each corresponding main run has completed. Verify with 12_verify_study.py --stress and analyze with 06_evaluate.py --stress. Run 20_additional_reports.py after both main and stress analyses are final.

## Annotation-free interface and treatment-history extension
This post-primary sensitivity analysis reuses all 660 original held-out cases. The named-variable interface runs 12 self-fed steps with unchanged visible information; treatment-history retrieval has two observed-history anchors and deterministic labels. No human annotation or model-as-judge scoring is used. Read docs/interface_memory_extension.md. Run 33_freeze_extension.py, 32_launch_extension.py, 30_verify_extension.py and 31_analyze_extension.py in that order in a fresh reproduction. The original run is protected by data/manifests/interface_memory_freeze.json. Never replace original study outcomes with the extension.

## Figure typography
All figure text uses embedded Times New Roman (regular, bold and italic), registered from project-local files in data/assets/fonts/microsoft_corefonts. Before regeneration in a fresh workspace, run env/.venv/bin/python repo/src/medwm_eval/figure_style.py; the setup downloads the original Microsoft core-font archive and extracts it with 7-Zip, verifying archive and font hashes. No system fonts are changed. The final figures are exported as embedded-font PDF and 300-dpi PNG.

## Analysis reports
After verified final analyses:
~~~bash
env/.venv/bin/python repo/scripts/18_describe.py
env/.venv/bin/python repo/scripts/27_auxiliary_content.py
env/.venv/bin/python repo/scripts/20_additional_reports.py
env/.venv/bin/python repo/scripts/23_failure_audit.py
env/.venv/bin/python repo/scripts/24_control_identity.py
env/.venv/bin/python repo/scripts/36_audit_extension.py
env/.venv/bin/python repo/scripts/38_audit_interface_failures.py
env/.venv/bin/python repo/scripts/39_audit_retrieval_time.py
~~~
The public repository supplies scientific analysis and figure/table scripts. It does not include manuscript source, the user-provided schematic, author administrative records, or journal packaging scripts. Generated assets are written to `paper/figures/` and `paper/tables/`.

## Reproduction caveats
Use a fresh project workspace when re-running generation; do not overwrite the archived study artifacts. The current data loader adds missing-index download support; its exact original version is retained under repo/archive/. Core prompt and execution files remain the pre-test frozen versions. The numerical baseline sex-encoding amendment is documented separately. Public data APIs and computational kernels may change; exact original raw clinical data and model hashes are retained locally, while release archives exclude raw clinical tracks and weights. A successful fresh run creates its own freeze and must not be described as the original run.

## Published results versus fresh reproduction

`data/results/` contains the original aggregate analysis. `data/simulated/` contains the development/test synthetic inputs and simulated references. `data/releases/synthetic_outputs.zip` contains simulator-only model responses, predictions, diagnostics and numerical controls. It contains no VitalDB patient-level records. Inspect and independently recalculate the primary result without a GPU or clinical-data download:

~~~bash
env/.venv/bin/python repo/scripts/41_verify_public_results.py
~~~

This check reads the ZIP directly, verifies hashes and synthetic-only scope, and recomputes the primary paired errors and cluster confidence intervals. It writes its audit to `data/results/public_verification.json`.

Original freeze records and model checksums are archived under `data/provenance/original/`. Their hashes describe the original run, including clinical files retained locally; they are not active freezes for a new checkout. Do not copy them into `data/manifests/`. Fresh generation/inference produces its own manifests and overwrites generated aggregate files, so perform a new full run in a separate checkout. The archived synthetic outputs are not a complete clinical-inference run and should not be passed to scripts expecting all 660 cases.

Original code is supplied under the MIT license. Third-party assets and data retain their terms; see LICENSE_NOTICE.md. No clinical treatment recommendations or patient benefit are established.
