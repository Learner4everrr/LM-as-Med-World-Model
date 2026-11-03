# Data dictionary and execution contract

All paths are relative to PROJECT_ROOT. Code belongs in repo/, data and computed artifacts in data/, and manuscript assets in paper/.

## Inputs and references

Original simulator cases: data/simulated/benchmark_v1/{development,test}.inputs.jsonl.
Factual cases: data/processed/vitaldb_v1/{development,test}.inputs.jsonl.
Separate reference files replace inputs with oracles in the filename. The exploratory extension is data/simulated/hypoglycemia_stress_v1/test.{inputs,oracles}.jsonl.

Each input row contains:
- id: branch/window identifier; group: shared simulator branch group or factual window.
- patient: virtual identity, PAS demographic-profile cluster, or actual VitalDB subject key.
- domain: simglucose, anesthesia, or vitaldb.
- split: development or test; kind: dose/timing/action_change/factual; branch: A/R/B or factual.
- scenario: original scenario index or factual anchor grid index.
- payload: variables, units, action_names, step_min, demographics, observation_definition, action_information, history and actions.

History rows contain time (minutes relative to anchor), values (ordered observations), past_action (preceding-interval rates) and past_meal_g (preceding-interval announced carbohydrate). Actions contain start time, ordered rates and announced meal_g. A model-visible serializer allowlists fields and excludes identities, group/split/scenario labels, seeds, hidden parameters and references. It never passes the entire saved case directly. In VitalDB, the stored retrospective action sequence is revealed only one current action at a time.

Each reference row has id/group/patient and targets, a 12-by-variable array. Simulator metadata records generation seeds/recipes/profiles, anchor-state hashes and replay checks. VitalDB metadata records source track IDs and the track-relative anchor. These reference-only fields are not model inputs.

## Units and timing

CGM: mg/dL; macro transition 30 minutes; 12 targets (360 minutes). The simulator integrates each minute and its Dexcom environment samples every three minutes. Returned CGM is the mean of held per-minute sensor values. Initial observation uses the cached final reset reading because upstream reset measures twice but its history stores the first reading. Pump basal and the API's bolus argument are rates in U/min; this study varies basal only. Applied rates are pump-quantized. Announced meals are not absorbed carbohydrate.

PAS: BIS (unitless), MAP (mmHg); propofol mg/s, remifentanil ug/s. Macro transition two minutes; 12 targets (24 minutes); internal integration ten seconds. Its binary sex argument is 0=female, 1=male. Profiles are synthetic and scenario PK/PD seeds differ.

VitalDB: BIS and MAP; macro transition one minute, 12 targets. Observations are as-of samples no older than 15 seconds; pump samples no older than 90 seconds. PPF20 and RFTN20 rate tracks in mL/hour are converted with 20 mg/mL and 20 ug/mL respectively. Only within-case relative times are used. Current action is not known before its replay step.

## Execution and outputs

Frozen main runner: repo/scripts/05_run_local.py; frozen prompt/parser: repo/src/medwm_eval/prompts.py.
Each model has data/runs/study_v1/test/<org>--<model>/ containing raw.jsonl, predictions.jsonl, auxiliary.json, diagnostics.json, manifest.json and launch.log. Stress runs are under stress_v1 and omit auxiliary/robustness tasks.

raw.jsonl stores the exact user prompt, SHA-256, decoded answer, model/version, input-token count, padded generated-token slots (an upper bound), batch duration/size and UTC timestamp. Chat-template and weights versions are separately fixed. Prompt caching is within a model/run.

Prediction rows identify case/model/condition/mode and contain attempted steps. Each step records validity, point values, interval validity/bounds when valid, scalar-normalization flag and prompt hash. The permitted one-variable scalar normalization was frozen before testing. No missing variable is supplied. Invalid self-fed output terminates the branch; later scheduled points remain missing. Teacher forcing continues with reference state. No retry or repaired downstream trajectory is used.

SA/free appends the preceding model prediction; SA/teacher appends the preceding reference; S/free removes all action/meal fields but retains self-feedback. A single multi-horizon response is a separate diagnostic, not a rollout.

## Analysis artifacts

data/results/study_v1 and stress_v1 contain final integrity.json, analysis_manifest.json, summary.json/csv, per-point and per-trajectory/pair Parquet files, and generated figures/tables. Interim analyses are explicitly marked partial and stored in an interim/ subfolder.

summary rows identify type/model/domain/variable/condition/mode and any pair family, step, subgroup or event. mean/lo/hi are patient/profile-weighted estimates and percentile 95% cluster bootstrap limits; clusters is the contributing cluster count; n is the finite contributing row count, not the independent sample size. Null means undefined. One-cluster bootstrap intervals are degenerate and are omitted in manuscript displays.

Complete-trajectory errors exclude incomplete paths, with completion reported separately. Per-step errors use available predictions. Event misses include unavailable predictions as no detection. All-point interval coverage counts missing intervals as uncovered. Strict auxiliary scores and the post hoc leading-JSON content audit are separate; the latter does not modify transition scores.

## Provenance and review

The pre-test internal freeze is data/manifests/study_freeze.json. A baseline sex-encoding correction, separate stress freeze and post hoc auxiliary content audit have their own records. Original baseline outputs are preserved. Source/model hashes, exact prompt reconstruction and branch replay support auditability. There was AI-assisted methods review, not independent clinical validation; thresholds are descriptive and do not define clinical adequacy.

## Annotation-free interface/history extension
The separately frozen interface_memory_v1 study reuses all original held-out inputs. Its transition condition is state_action_named/free: the original visible input object, with a variable-keyed value/lower/upper response. It has no scalar normalization, numerical content rescue, retry or teacher-forced condition. Failed points terminate a branch. The standalone retrieval task appends observed reference states only to its separate anchor-6 histories; it cannot alter rollout feedback.

Each model's extension folder has predictions.jsonl, raw.jsonl, memory.json and manifest.json. Retrieval rows identify case/domain/patient and anchor_step (0 or 6), raw answer, prompt hash and a deterministic target copied from displayed history. The target has current_time, current_values, last_action, query_time, query_action and last_action_interval_start. The last_action refers to the interval ending at the current observation, not a recommendation or future prescription.

Extension summary types are trajectory, step, interval, paired, direction, interface_trajectory, interface_paired and retrieval. Retrieval is stratified by anchor/parser (strict or first-object content), with every scheduled task in the denominator. Error contrasts are named minus original on matched complete cases/pairs; completion differences retain all cases/pairs. The paired sign_all auxiliary statistic averages eligible points within each case before clustering; the manuscript dose-direction table instead uses type=direction/metric=correct, preserving the original point-within-patient estimand.

The extension is post-primary and reuses subjects; it adds no independent clinical sample. Its freeze, full prompt reconstruction and independent numerical audit are saved under data/manifests/interface_memory_freeze.json and data/results/interface_memory_v1/. Original primary, summary and integrity files must remain byte-identical.

Post hoc extension audits: failure_summary.json contains category counts and fixed-order synthetic examples only; failure_audit_private.json additionally holds clinical examples and is excluded from release. time_components.json decomposes the original temporal conjunction without changing it. independent_audit.json independently verifies 408 original extension statistics. All three public aggregate audit files are available in the repository under data/results/interface_memory_v1/.
