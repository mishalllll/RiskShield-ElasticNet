import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Imbalance handling library (Make sure to run: pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE

# --- Configuration & Mock Data Setup ---
# Replacing with actual file path if working locally
INPUT_CSV = "financial_ledger.csv" 

def generate_mock_fintech_data(output_path):
    """Generates a heavily imbalanced fintech ledger dataset for testing."""
    np.random.seed(42)
    n_samples = 50000
    
    # 95% approved/good loans (0), 5% defaults (1) -> Severe class imbalance
    is_default = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Generate some standard credit risk features
    credit_score = np.random.normal(700, 50, n_samples) - (is_default * 80)
    debt_to_income = np.random.uniform(0.1, 0.6, n_samples) + (is_default * 0.2)
    missed_payments = np.random.poisson(0.2, n_samples) + (is_default * 2)
    
    df = pd.DataFrame({
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'missed_payments': missed_payments,
        'is_default': is_default
    })
    df.to_csv(output_path, index=False)
    print(f"Generated mock highly imbalanced dataset at: {output_path}")

# --- Step 1: Ingest & Preprocess Dataset ---
# Generate mock dataset automatically if it doesn't exist
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    generate_mock_fintech_data(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

X = df.drop(columns=['is_default'])
y = df['is_default']

print("\n--- Class Distribution Before SMOTE ---")
print(y.value_counts(normalize=True))

# Train-Test Split (Stratified to maintain baseline proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (Critical for regularized generalized linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 2: Handle Imbalanced Classes using SMOTE ---
print("\nApplying Synthetic Minority Over-sampling Technique (SMOTE)...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("--- Class Distribution After SMOTE ---")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# --- Step 3: Train an ElasticNet Regularized Classification Model ---
print("\nTraining ElasticNet Logistic Regression Model...")
# In scikit-learn, LogisticRegression handles ElasticNet via:
# penalty='elasticnet', solver='saga', and l1_ratio (0.5 balances L1 & L2)
model = LogisticRegression(
    penalty='elasticnet', 
    solver='saga', 
    l1_ratio=0.5, 
    C=1.0, 
    random_state=42, 
    max_iter=5000
)
model.fit(X_train_resampled, y_train_resampled)

# --- Step 4: Evaluate ROC-AUC and Optimize Threshold ---
# Get predicted probabilities for the default class (1)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Calculate false positive rate, true positive rate, and operational thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Operational Business Logic Matrix:
# Let's say a False Negative (approving a defaulting client) costs the bank $10,000.
# A False Positive (wrongfully rejecting a good client) costs $500 in lost processing fees.
cost_fn = 10000
cost_fp = 500
net_profits = []

for th in thresholds:
    y_pred_th = (y_probs >= th).astype(int)
    cm = confusion_matrix(y_test, y_pred_th)
    tn, fp, fn, tp = cm.ravel()
    
    # Net financial layout evaluation
    total_loss = (fn * cost_fn) + (fp * cost_fp)
    # Profit metric represents minimizing total operational risks/losses
    net_profits.append(-total_loss) 

# Select the exact threshold maximizing our business cost equation
best_index = np.argmax(net_profits)
optimal_threshold = thresholds[best_index]

print(f"\nOptimization Results:")
print(f"-> ROC-AUC Score: {roc_auc:.4f}")
print(f"-> Selected Profit-Maximizing Threshold: {optimal_threshold:.4f}")

# --- Step 5: Visualizations ---
plt.figure(figsize=(14, 5))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[best_index], tpr[best_index], color='red', s=100, label=f'Optimal Thresh ({optimal_threshold:.2f})', zorder=5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Type I Error)')
plt.ylabel('True Positive Rate (Recall / Power)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 2: Financial Threshold Optimization Curve
plt.subplot(1, 2, 2)
plt.plot(thresholds, net_profits, color='teal', lw=2, label='Profit Profile')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.xlabel('Classification Threshold Probability')
plt.ylabel('Relative Financial Output (Negated Cost)')
plt.title('Profit Maximization vs. Threshold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Step 6: Final Performance Verification at Selected Threshold ---
final_preds = (y_probs >= optimal_threshold).astype(int)
print("\n--- Final Operational Classification Report ---")
print(classification_report(y_test, final_preds))