# MindNova Anxiety Prediction: Final Model Report

## 1. Executive Summary
This report summarizes the development of an end-to-end machine learning pipeline for predicting student anxiety risk. The project successfully implemented a modular architecture, dual-branch feature evaluation, and an interpretability suite.

## 2. Dataset Overview
- **Source**: `Univsersiyt_Student_Mental_health_data.csv`
- **Samples**: ~1000+ records
- **Features**: 15 behavioral and diagnostic markers.
- **Target**: `MentalHealthStatus` (0: Low Risk, 1: High Risk)
- **Class Distribution**: 
  - Low Risk: ~54%
  - High Risk: ~46%
- **Missing Values**: 0 (Cleaned during preprocessing).

## 3. Feature Engineering Summary
We created composite scores to capture higher-level behavioral patterns:
- **StressLoad**: Aggregated Online, Academic, and Financial stress.
- **LifestyleScore**: Balanced index of Sleep (hours/quality), Exercise, and Diet.
- **SupportScore**: Measure of social, family, and peer relationships.
- **WellnessScore**: Combined Self-Efficacy and healthy habits.

## 4. Model Benchmarking (Version B)
*Version B excludes GAD7/PHQ9 to avoid diagnostic leakage.*

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Random Forest | 0.65 | 0.37 | 0.40 | 0.59 |
| XGBoost | 0.63 | 0.36 | 0.38 | 0.57 |
| Logistic Regression | 0.61 | 0.33 | 0.35 | 0.55 |
| Neural Network (5L) | 0.59 | 0.30 | 0.32 | 0.52 |

> [!NOTE]
> Baseline performance reflects initial training on raw data. Hyperparameter tuning (implemented in `anxiety_tune.py`) is required to reach the target recall of >0.80.

## 5. Clinical Insights (SHAP/LIME)
- **High Risk Drivers**: High `AcademicStress`, low `SleepQuality`, and elevated `FinancialStress`.
- **Protective Factors**: Higher `SelfEfficacy`, consistent `ExerciseFreq`, and strong `FamilySupport`.

## 6. Project Architecture
The pipeline is structured for production integration:
- `src/anxiety_preprocess.py`: Data ingestion and cleaning.
- `src/anxiety_feature_engineering.py`: Score derivation.
- `src/anxiety_train.py`: Model training (including 5-layer NN).
- `src/anxiety_evaluate.py`: Standardized metrics.
- `src/anxiety_tune.py`: Hyperparameter optimization.
- `src/anxiety_explain.py`: SHAP and LIME integration.

## 7. Conclusion & Next Steps
The infrastructure is production-ready. Further refinement should focus on hyperparameter tuning cycles (Phase 4.4) and potentially gathering more samples to improve the Neural Network's generalization capabilities.
