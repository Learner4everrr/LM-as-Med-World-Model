# Post hoc revision sensitivity analysis

Recorded before computing the revision summaries. This is a post hoc analysis of the already inspected, original held-out results; it is not preregistration or independent replication. No model inference, prompt selection, output repair or primary-result replacement is performed.

Scope: original state-plus-action, self-fed dose pairs in glucose and anesthesia; all four LLMs and original numerical controls (history-only ridge retains its original input condition). Pairing, native units, and patient/profile clustering follow the original protocol.

Descriptive magnitude bins use absolute simulator-reference A-minus-B effects at individual horizons. CGM: [0,5), [5,20), [20,50), [50,infinity) mg/dL. BIS and MAP: [0,1), [1,5), [5,10), [10,infinity) native units. These round, unit-specific bins are not clinical adequacy thresholds and will not be changed after observing results. Reference distributions are reported over all 648 dose-pair/time points per outcome.

Each stratum reports scheduled and evaluable points, distinct pairs/clusters, raw and cluster-weighted point availability, and matched LLM/control error differences. Available-point results include a point only if both branches produced valid values; a separate analysis restricts to pairs complete at all twelve steps. Within each stratum, errors are averaged over points within pair, then scenarios within identity/profile, then equally across identities/profiles. Zero-effect error is evaluated on exactly the same points. Percentile cluster intervals use 2,000 resamples, seed 20261102; a single cluster has no inferential interval.

Horizon contrasts use all twelve horizons without selecting favorable time points, both among available pairs at each step and among fully completed pairs. Counts accompany every stratum. Multiple exploratory intervals are descriptive; no multiplicity-adjusted significance claims are made. The all-horizon complete-pair estimates must reproduce the original saved paired-effect summaries.

The separate existing failure-taxonomy and leading-object auxiliary audits are retained as post hoc descriptive evidence. No repaired partial response is treated as a completed self-fed trajectory. Downstream clinical strategy selection and patient outcomes are not evaluated.

## Saved-response parseability audit

The revision additionally inspects issued state-plus-action, self-fed responses for the two simulated domains under the original and named-variable interfaces. The alternative reads only the first JSON object beginning at the first opening brace, then applies the same point-value numeric/type/cardinality rules. It does not coerce strings, merge objects, or fill missing fields. Report strict-valid and first-object-valid issued response positions, first failures with a readable point, and categories of first failure. Valid points with invalid intervals and finite out-of-range points are recorded separately. The denominator is issued positions, not all scheduled positions or unique generation calls. No missing continuation is regenerated; the audit cannot estimate repaired trajectory completion. Padded batch token-slot counts are insufficient to identify individual truncation, which is left unclassified.
