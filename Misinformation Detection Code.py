import pandas as pd  s
import matplotlib.pyplot as plt 
import seaborn as sns  
import numpy as np
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler  
from sklearn.impute import SimpleImputer  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score  
from sklearn.metrics import accuracy_score

# Import Data
data = pd.read_csv(r'D:\桌面\NTU\BC2406\CBA Question Paper\CBA Question Paper\misinformation2.csv')

# Q1 Data Exploration

# First Notable Finding
print(data.head())
data.info() 
sns.countplot(x='is_misinformation', data=data)
plt.title('Number of Misinformation (1) vs. Not Misinformation (0) ')
plt.show()

#Second Notable Finding
data.describe(include="all").to_csv('data_describe.csv')
subset = data[data["is_misinformation"] == 1]  
subset.describe(include="all").to_csv("data_describe_Y=1.csv")
print(data['is_misinformation'].value_counts(normalize=True))

#Third Notable Finding
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6)) 
ax = sns.boxplot(x='is_misinformation', y='source_domain_reliability', data=data)
plt.title('Source Domain Reliability by Misinformation Status', fontsize=16)
plt.ylabel('Source Domain Reliability Score', fontsize=12)
plt.xlabel('Post Status', fontsize=12)
ax.set_xticklabels(['Not Misinformation (0)', 'Misinformation (1)'])
plt.show()

#Q2 Data Preparation & Data Cleaning
#Search for missing value
data.shape
data.columns.tolist()
data.head
missing = data.isnull().sum().sort_values(ascending=False)
missing_percent = (data.isnull().mean()*100).sort_values(ascending=False)
pd.concat([missing,missing_percent],axis = 1, keys = ["missing_count","missiong_percent"]).to_csv("missing_data.csv")

#Duplication Check
duplicates_all = data.duplicated().sum()
duplicates_id = data['id'].duplicated().sum()
print(f"Total duplicate rows: {duplicates_all}")
print(f"Duplicate IDs: {duplicates_id}")

#Data Cleaning
X = data.drop(['is_misinformation'], axis=1)
y = data['is_misinformation']
X = X.drop(['id', 'timestamp', 'date', 'time'], axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


#Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Before
X_encoded_sample = X_encoded.iloc[:, :5].describe()
axes[0].bar(range(len(X_encoded_sample.columns)), X_encoded_sample.loc['mean'], 
            label='Mean (before)', alpha=0.7)
axes[0].set_title('Before Standardization')
axes[0].set_ylabel('Mean Value')
axes[0].set_xlabel('Features')
# After
X_scaled_sample = X_scaled_df.iloc[:, :5].describe()
axes[1].bar(range(len(X_scaled_sample.columns)), X_scaled_sample.loc['mean'], 
            label='Mean (after)', alpha=0.7, color='orange')
axes[1].set_title('After Standardization')
axes[1].set_ylabel('Mean Value')
axes[1].set_xlabel('Features')
plt.tight_layout()
plt.show()

#Q3 Logistic Regression and CART
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

#Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn_lr = cm_lr[0, 0]  # True Negatives
fp_lr = cm_lr[0, 1]  # False Positives
fn_lr = cm_lr[1, 0]  # False Negatives
tp_lr = cm_lr[1, 1]  # True Positives
fpr_lr = fp_lr / (fp_lr + tn_lr) if (fp_lr + tn_lr) > 0 else 0  # False Positive Rate
fnr_lr = fn_lr / (fn_lr + tp_lr) if (fn_lr + tp_lr) > 0 else 0  # False Negative Rate
overall_error_lr = (fp_lr + fn_lr) / len(y_test)  # Overall Error Rate
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Logistics Regression Model Complexity
num_features_lr = X_train.shape[1]

#CART 
cart_model = DecisionTreeClassifier(random_state=42)  
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

#Confusion Matrix for CART
cm_cart = confusion_matrix(y_test, y_pred_cart)
tn_cart = cm_cart[0, 0]
fp_cart = cm_cart[0, 1]
fn_cart = cm_cart[1, 0]
tp_cart = cm_cart[1, 1]
fpr_cart = fp_cart / (fp_cart + tn_cart) if (fp_cart + tn_cart) > 0 else 0
fnr_cart = fn_cart / (fn_cart + tp_cart) if (fn_cart + tp_cart) > 0 else 0
overall_error_cart = (fp_cart + fn_cart) / len(y_test)
accuracy_cart = accuracy_score(y_test, y_pred_cart)

# Model Complexity of CART
num_terminal_nodes_cart = cart_model.get_n_leaves()
num_internal_nodes_cart = cart_model.tree_.node_count
tree_depth = cart_model.get_depth()

# Model Comparison Resluts
comparison_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'CART'],
    'Model Complexity': [f'{num_features_lr} X variables', 
                        f'{num_terminal_nodes_cart} terminal nodes'],
    'False Positive Rate': [f'{fpr_lr:.4f}', f'{fpr_cart:.4f}'],
    'False Negative Rate': [f'{fnr_lr:.4f}', f'{fnr_cart:.4f}'],
    'Overall Error': [f'{overall_error_lr:.4f}', f'{overall_error_cart:.4f}'],
    'Accuracy': [f'{accuracy_lr:.4f}', f'{accuracy_cart:.4f}']
})
comparison_results.to_csv('model_comparison_results.csv', index=False)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# FPR
models = ['Logistic Regression', 'CART']
fpr_values = [fpr_lr, fpr_cart]
axes[0, 0].bar(models, fpr_values, color=['blue', 'red'], alpha=0.7)
axes[0, 0].set_title('False Positive Rate (FPR) Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('FPR')
axes[0, 0].set_ylim(0, max(fpr_values) * 1.2)
for i, v in enumerate(fpr_values):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
# FNR
fnr_values = [fnr_lr, fnr_cart]
axes[0, 1].bar(models, fnr_values, color=['blue', 'red'], alpha=0.7)
axes[0, 1].set_title('False Negative Rate (FNR) Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('FNR')
axes[0, 1].set_ylim(0, max(fnr_values) * 1.2)
for i, v in enumerate(fnr_values):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
# Error Rate
error_values = [overall_error_lr, overall_error_cart]
axes[1, 0].bar(models, error_values, color=['blue', 'red'], alpha=0.7)
axes[1, 0].set_title('Overall Error Rate Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Error Rate')
axes[1, 0].set_ylim(0, max(error_values) * 1.2)
for i, v in enumerate(error_values):
    axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
# Acuuracy
accuracy_values = [accuracy_lr, accuracy_cart]
axes[1, 1].bar(models, accuracy_values, color=['blue', 'red'], alpha=0.7)
axes[1, 1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim(0, 1)
for i, v in enumerate(accuracy_values):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Misinformation', 'Misinformation'],
            yticklabels=['Not Misinformation', 'Misinformation'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
sns.heatmap(cm_cart, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=['Not Misinformation', 'Misinformation'],
            yticklabels=['Not Misinformation', 'Misinformation'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('CART - Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.show()


#Q4 Importance Analysis
X_numerical_only = data.drop(['is_misinformation', 'id', 'timestamp', 'date', 'time'], axis=1)
X_numerical_only = X_numerical_only.select_dtypes(include=[np.number])
numeric_feature_names = X_numerical_only.columns.tolist()
# importance
lr_all_coefficients = lr_model.coef_[0]
all_feature_names = X_encoded.columns.tolist()
numeric_indices = []
for feat in numeric_feature_names:
    if feat in all_feature_names:
        numeric_indices.append(all_feature_names.index(feat))

lr_numeric_coefficients = lr_all_coefficients[numeric_indices]
feature_importance_lr_numeric = pd.DataFrame({
    'Feature': numeric_feature_names,
    'Coefficient': lr_numeric_coefficients,
    'Abs_Coefficient': np.abs(lr_numeric_coefficients)
}).sort_values('Abs_Coefficient', ascending=False)
top5_lr = feature_importance_lr_numeric.head(5)
cart_all_importances = cart_model.feature_importances_
cart_numeric_importances = cart_all_importances[numeric_indices]

feature_importance_cart_numeric = pd.DataFrame({
    'Feature': numeric_feature_names,
    'Importance': cart_numeric_importances,
    'Importance_Percent': cart_numeric_importances * 100
}).sort_values('Importance', ascending=False)

top5_cart = feature_importance_cart_numeric.head(5)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LR Top 5
colors_lr = ['red' if x < 0 else 'blue' for x in top5_lr['Coefficient']]
axes[0].barh(range(len(top5_lr)), top5_lr['Coefficient'], color=colors_lr, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(top5_lr)))
axes[0].set_yticklabels(top5_lr['Feature'], fontsize=12, fontweight='bold')
axes[0].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
axes[0].set_title('Logistic Regression: Top 5', fontsize=13, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[0].grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top5_lr.iterrows()):
    axes[0].text(row['Coefficient'] + 0.01 if row['Coefficient'] > 0 else row['Coefficient'] - 0.01, 
                i, f"{row['Coefficient']:.4f}", va='center', fontweight='bold', fontsize=10)

# CART Top 5
axes[1].barh(range(len(top5_cart)), top5_cart['Importance_Percent'], color='green', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_yticks(range(len(top5_cart)))
axes[1].set_yticklabels(top5_cart['Feature'], fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
axes[1].set_title('CART Decision Tree: Top 5', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top5_cart.iterrows()):
    axes[1].text(row['Importance_Percent'] + 0.5, i, f"{row['Importance_Percent']:.2f}%", 
                va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

