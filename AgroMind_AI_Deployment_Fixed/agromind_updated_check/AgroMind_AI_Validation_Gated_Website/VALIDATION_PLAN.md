# AgroMind AI — Evidence Required Before Validated Agricultural Recommendations

## Current release status
`research_prototype` / `experimental_not_validated`.

The model output must not be represented as a validated recommendation, a guarantee of yield, or a guarantee of profitability. A high random-split test score is not independent field validation.

## Required validation program
1. **Leakage and feature-availability audit**
   - Remove post-decision or target-proxy features from prediction-time inputs, including harvest outcomes, crop duration if not known before planting, crop-specific fields, and any label-derived upper/lower bounds.
   - Freeze a documented feature contract that can actually be collected before sowing.
   - Fit all encoders/scalers on training data only. Keep a locked test set untouched until final evaluation.
2. **Independent geographic and temporal evaluation**
   - Evaluate on independent seasons, districts/regions, soil types, and farms not represented in training.
   - Report per-crop precision/recall/F1, macro-F1, top-3/top-5 hit rate, confusion matrix, class support, calibration, and abstention/coverage.
   - Include confidence intervals and compare with agronomist recommendations and simple agronomic baselines.
3. **Field trials**
   - Prospective, locally approved trials across representative farms and seasons.
   - Record recommended crop, farmer/agronomist decision, soil tests, weather, irrigation, input practices, crop establishment, yield, costs, adverse outcomes, and harvest outcome.
   - Use an appropriate sample-size/power plan with agricultural researchers/statisticians; pre-register endpoints and analysis.
   - Compare with farmer practice and local extension guidance. A crop-classification match alone is not enough; evaluate yield, economic outcomes, and downside risk.
4. **Agronomist review**
   - Qualified local agronomists/extension specialists review recommendations and exclusions for each launch region and crop set.
   - Document approved geography, season, soil conditions, irrigation assumptions, input ranges, and contraindications.
5. **Operational controls**
   - Out-of-distribution detection, missing-input validation, calibrated uncertainty, abstention when uncertain, model/data versioning, monitoring, incident reporting, and rollback.
   - Display local soil testing and extension-service contact guidance. Never issue pesticide or fertilizer dosages without separately validated, jurisdiction-specific guidance.

## Evidence record required to enable a validated status
Store a signed/reviewed validation report with:
- report identifier, version, date, responsible research institution and agronomist reviewers;
- model artifact hash, code/data versions, frozen feature contract;
- prospective study protocol, sample counts, locations/seasons, inclusion/exclusion criteria;
- independent test metrics and confidence intervals, per-crop results, calibration and coverage;
- field outcomes and adverse events;
- explicit approved scope and limitations;
- approval signatures / institutional review as applicable.

Do not set `validated_for_recommendations` to true until this evidence exists and is reviewed. A configuration flag is not proof of validation; it only records a documented review decision.
