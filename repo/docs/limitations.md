# Study limitations

The study evaluates four frozen, relatively small open-weight models from three developers, with Qwen3 thinking disabled. It does not assess all current models or dedicated trained clinical world models.

The two simulator implementations supply reproducible in-simulator action references, not validated individual clinical treatment effects. Glucose has 30 fixed library identities; PAS uses synthetic demographic profiles and scenario-specific random PK/PD parameters, without observation noise or operative disturbances. Partial visible histories need not identify latent dynamics.

The VitalDB arm is factual replay on a selected single-centre subset. Its quality rules use future signal availability/artifact bounds; actions are revealed as time unfolds. Other drugs and clinical decisions are not fully observed. It is neither a prospective anchor-time forecast nor evidence of individual clinical counterfactuals. Public-data pretraining exposure cannot be ruled out.

Errors conditional on complete outputs can select easy cases. Format compliance, numeric forecasting and medical content are distinct. State extraction and two repeated pharmacology questions are narrow probes. A separately declared post hoc auxiliary content audit helps diagnose format confounding but does not revise transition scores.

Numerical controls have in-domain supervised training and fixed limited features. Their comparison with zero-shot LLMs does not isolate model architecture. Bootstrap intervals reflect the finite observed clusters, not broad human-population uncertainty. The post-freeze hypoglycemia extension reuses the test identities and has positive low events in only six.

The internal protocol freeze is not public preregistration. The baseline categorical correction, stress extension and post hoc audits are disclosed. No independent clinician review, comprehensive clinical safety specification, study-specific institutional ethics determination or prospective validation is claimed. The researcher authorized AI-assisted review. No clinically adequate horizon or whole-medicine world-model qualification is assigned.

Target publication is npj Digital Medicine; use of its article structure and the requested official NeurIPS draft style is not evidence of submission, review or acceptance.
