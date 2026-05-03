# Stress Model Ablation Study: Anti-Leakage Diagnostic

This report analyzes how performance shifts as potentially leaky or simulated features are removed.

```text
                 Variant  Features      AUC   Recall  Precision    Brier
        Variant A (Full)        10 0.999999 0.999066   0.999066 0.000516
      Variant B (No Eng)        10 0.999999 0.999066   0.999066 0.000516
  Variant C (No Proxies)        10 0.999999 0.999066   0.999066 0.000516
Variant D (Cross-Domain)        10 0.999999 0.999066   0.999066 0.000516
Variant E (Target-Blind)         8 0.534000 0.081906   0.827044 0.208154
     Variant F (Minimal)         5 0.506361 0.077857   0.496032 0.219470
```


> [!TIP]
> **Variant E (Target-Blind)** represents the truest measure of behavioral learning without direct survey indicators.