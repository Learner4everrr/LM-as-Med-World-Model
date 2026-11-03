# Post hoc auxiliary content audit

Added 2 November 2026 after completed primary-condition endpoint inspection. Motivation: raw examples showed correct explicit direction answers followed by explanatory prose; full-response JSON scoring conflated content and response compliance. This is exploratory, not a newly confirmatory endpoint.

Decode only the first complete JSON object beginning at the first opening brace. Do not scan later objects, repair truncation, infer answers from prose, change numeric values or canonicalize omitted fields. Score copied state/time and the explicit direction array separately. Keep every scheduled auxiliary case in the denominator and cluster by the original patient/profile. Preserve and report the original strict auxiliary scores alongside this audit.

This does not change any transition parser, prediction, failure, interval, paired-effect result, self-fed history or primary conclusion. It tests content in a narrow two-question probe, not comprehensive medical knowledge.

The rule and script are saved before aggregate content-audit scores are computed.
