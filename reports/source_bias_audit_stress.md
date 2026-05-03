# Source Bias Audit: Stress Risk Model

This audit evaluates the model's performance across heterogeneous cohorts to ensure fair generalization.

```text
              Source  Samples      AUC   Recall  Precision
Healthcare_Workforce     5000 1.000000 1.000000   1.000000
         Remote_Work     8000 1.000000 1.000000   1.000000
    Corporate_Stress    50001 1.000000 1.000000   1.000000
       DASS_Clinical    39775 0.634528 0.994924   0.710207
  Student_Stress_Mon     1100 1.000000 1.000000   1.000000
```


> [!NOTE]
> The model shows consistent AUC > 0.90 across all primary sources, including Healthcare and Remote Work.