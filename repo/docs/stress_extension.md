# Targeted hypoglycemia extension

Added after the primary specification freeze, before examining held-out LLM endpoint summaries. Trigger: a target-distribution audit found no CGM <70 mg/dL among original test endpoints. The original primary data, prompts, endpoint and comparison remain unchanged. This is a targeted secondary stress analysis, not an independently confirmatory replication.

Attempt all 18 original held-out simglucose identities. New fixed seeds: 20262000 + category index ×100 + identity index. No meals. Deliver twice the simulator reference basal during warmup, using the installed pump bounds. Examine every 30 minutes from 180 to 720 minutes. Select the first visible CGM between 80 and 130 mg/dL that fell by at least 3 mg/dL in the preceding interval. Retain the latest seven observations and their actions. Branch from exactly that anchor: A=0.5 reference basal, B=1.0 reference basal, for 12 ×30-minute steps. These numerical simulator perturbations are not clinical dosing advice.

Both branches must complete without simulator termination and exactly replay. Preserve all exclusions. Do not replace failures or select based on LLM performance. This may exclude the most severe trajectories and is explicitly conditional on simulator survival.

Run the same frozen four models, prompt, parser, decoding and feedback protocols (state/action free and teacher, and state-only free). Keep the extension separate in data/runs/stress_v1. Report low-event counts, failure coverage, low-event miss rates, complete-pair effect errors and trajectory errors, clustering by original virtual identity. Reusing these identities does not increase the independent sample count of the main study. No combined confirmatory primary statistic.

AI-assisted scenario review only. No independent clinical validation. Event labels and their limitations match the main protocol. Save an extension manifest before model inference.
