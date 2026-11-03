# AI-assisted scientific and implementation review

Date: 2026-11-02. The researcher explicitly authorized the assistant to conduct the requested review ("你帮我审阅即可"). This replaces the planned external review workflow for this computational study, not the qualifications of an independent clinician. No clinical expert review or safety certification is claimed.

Reviewed: installed simulator action semantics, reset and branch invariance, measurement timing, input allowlists, chronological replay, actual-subject splits, raw-data licensing, event definitions, failure handling, and patient-level statistical units. Automated contracts and saved generated-branch checks supplement this review.

Simglucose varies basal infusion within 0.75–1.25 times its patient-specific equilibrium reference. All rates are accepted by the installed pump; that is a software constraint, not a clinical safety guarantee. One development child scenario reached the simulator termination boundary; all five branches for that scenario were excluded and the full error is retained. Held-out scenarios were not replaced.

PAS varies propofol relative to numerical test schedules, with remifentanil fixed. It uses randomized PK/PD parameters and noiseless readouts. No surgical disturbance, changing fluid status, vasopressors, or measurement noise is modeled. BIS and MAP are simulator outputs, not validated patient outcomes. Doses are research perturbations, not treatment recommendations.

Glucose <70 or >180 mg/dL, BIS <40 or >60, and MAP <65 mmHg are descriptive event labels. They neither identify a complete dangerous-state taxonomy nor define clinically adequate simulation. MAP recommendations depend on setting and patient; the 65 cutoff is not asserted as a universal treatment target. Direction evaluation uses a numerical resolution of one native unit, with 0 and 5 units as sensitivity analyses. We do not label any horizon clinically usable.

VitalDB provides factual conditional replay only. Current pump rates are revealed one step at a time. Other treatments and operative disturbances remain unobserved. Track-unit conversions and no-future-resampling are audited. Future-dependent signal-quality eligibility may select a cleaner, less representative sample. The selected public research data remain local and are not sent to external inference providers.

Remaining limitations: no independent clinical review; simulator-specific truth; small virtual-identity counts; limited four-model/open-weight scale; short observation vectors; potential public-data pretraining exposure; no prospective clinical validation. These limitations must appear in the paper.
