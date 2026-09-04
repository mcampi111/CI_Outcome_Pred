#!/bin/bash
# Run this from inside the new repository directory.
# It copies the analysis scripts for the paper out of the two working folders.

LAST=~/Desktop/Postdoc_Zurich/DATACI/CI_Data_Package/LAST_ANALYSIS
EQ=~/Desktop/Postdoc_Zurich/Paper_CI_Outcome_Pred/Analysis_CI_Equaliser_New_TabPFN_and_more/Equalizer_Quant

# RQ1 — regression across algorithms and feature sets
cp "$LAST/run_tabpfn_FINAL.py"                              analysis/prediction/01_regression_grid.py
cp "$LAST/run_tabpfn_ci.py"                                 analysis/prediction/02_conformal_intervals.py

# RQ1 — how the predictor-outcome relationship changes
cp "$EQ/w1_all.py"                                          analysis/equalizer/01_wasserstein_all_features.py

# RQ2 — risk classification
cp "$LAST/run_tabpfn_classifier_subsets.py"                 analysis/risk/01_classifier_by_feature_subset.py
cp "$LAST/rerun_auc_by_category_n378.py"                    analysis/risk/02_auc_by_category.py
cp "$LAST/run_tabpfn_varimp_over_time.py"                   analysis/risk/03_permutation_importance.py

# RQ3 — six-month prognostic value and trajectories
cp "$LAST/run_classification_imp_and_longitudinal_check.py" analysis/trajectories/01_six_month_prognosis.py

# validation
cp "$LAST/run_tabpfn_over_time.py"                          analysis/validation/01_temporal_split.py
cp "$LAST/run_tabpfn_finetune_over_time.py"                 analysis/validation/02_finetune_over_time.py
cp "$LAST/run_tabpfn_classifier_over_time.py"               analysis/validation/03_classifier_over_time.py

# derived tables reported in the paper
cp "$EQ/W1_all49.csv"                                       results/
cp "$EQ/SI_table_COMPLETE_OT.csv"                           results/
cp "$LAST/SI_table_full_predictors_REGEN.csv"               results/
cp "$LAST/auc_by_category_n378.csv"                         results/
cp "$LAST/tabpfn_regression_grid_FINAL.csv"                 results/
cp "$LAST/tabpfn_importance_F60_FINAL.csv"                  results/

echo "Copied. Now check every script for absolute paths:"
echo "  grep -rn '/Users/martacampi' analysis/"
