import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_experiment_plots():
    report_path = 'reports/recovery_experiment_metrics.csv'
    if not os.path.exists(report_path):
        return

    df = pd.read_csv(report_path)
    os.makedirs('figures/recovery_experiment', exist_ok=True)

    # 1. AUC Comparison Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Variant', y='AUC', hue='Holdout', palette='viridis')
    plt.axhline(0.80, color='red', linestyle='--', label='Target AUC (0.80)')
    plt.title('Burnout Recovery Experiment: AUC Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/recovery_experiment/auc_comparison.png')

    # 2. Distribution Shift Visualization (Before vs After)
    # Load original vs. Variant B
    orig = pd.read_csv('data/experiment/Variant_A_Pool.csv')
    norm = pd.read_csv('data/experiment/Variant_B_Pool.csv')
    test = pd.read_csv('data/experiment/Workplace_Holdout.csv')

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(orig[orig['DatasetSource'].str.contains('Synthetic')]['StressLevel'], label='Synthetic (A)', fill=True)
    sns.kdeplot(test['StressLevel'], label='Workplace (Holdout)', fill=True)
    plt.title('DISTRIBUTION MISMATCH (Baseline)')
    plt.xlabel('StressLevel')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(norm[norm['DatasetSource'].str.contains('Synthetic')]['StressLevel'], label='Synthetic (Normalized)', fill=True)
    sns.kdeplot(test['StressLevel'], label='Workplace (Holdout)', fill=True)
    plt.title('ALIGNED DISTRIBUTION (Variant B)')
    plt.xlabel('StressLevel')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/recovery_experiment/distribution_alignment.png')
    print("Plots generated in figures/recovery_experiment/")

if __name__ == "__main__":
    generate_experiment_plots()
