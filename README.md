# Prediction of Individual Speech Outcomes with Cochlear Implants

Analysis code for the manuscript *Prediction of Individual Speech Outcomes with
Cochlear Implants: Limitations and Opportunities for Clinical Translation*.

## What the study asks

The paper addresses three questions using the Swiss National Cochlear Implant
Database (473 adults with bilateral hearing loss):

1. **Can pre-operative data predict individual post-operative speech scores, and
   if not, why?** Seven algorithms, from linear regression to a tabular
   foundation model, across four feature-set sizes, five imputation strategies
   and six validation procedures.
2. **Can pre-operative data identify patients at risk of a poor outcome?**
   Binary classification of word-recognition scores below 40%.
3. **Do early post-operative scores carry prognostic information?** The
   six-month score added to the pre-operative set, and clustering of serial
   scores into trajectory types.

The main finding is that cross-validated R² does not exceed 0.22 in any
configuration, and that this reflects the pre-operative data rather than the
models: the same features predict the *pre*-operative score well (R² = 0.48
from audiometry alone). Implantation selectively removes the predictive value
of audiometric and speech measures while developmental and communicative
variables retain or gain theirs.

## Repository layout

```
analysis/
  prediction/        RQ1 — regression across algorithms and feature sets
  equalizer/         RQ1 — how the predictor-outcome relationship changes (W1)
  risk/              RQ2 — binary classification of poor outcome
  trajectories/      RQ3 — six-month prognostic value and trajectory types
  validation/        temporal splits, conformal intervals, learning curves
figures/             scripts that produce the manuscript figures
results/             derived tables (CSV) reported in the paper and Supplement
```

## Data

The Swiss National Cochlear Implant Database is managed by the participating
centre. Data access requires ethics approval and is subject to Swiss data
protection regulations, so the patient-level data are **not** included here.
The scripts expect a file named `CI_UNIFIED_DATASET.xlsx` in a `data/`
directory, which is not tracked.

`results/` contains the derived tables that underlie the published figures and
supplementary tables; these are aggregate and contain no patient-level
information.

## Requirements

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

TabPFN is fine-tuned on the present dataset; the fine-tuning runs were
performed on an NVIDIA DGX workstation and are not reproducible on CPU in
reasonable time. All other analyses run on a laptop.

## Citation

Campi M, Huber A, Goehring T. *Prediction of Individual Speech Outcomes with
Cochlear Implants: Limitations and Opportunities for Clinical Translation.*
(under review)

## License

MIT — see `LICENSE`.
