# model.py - Model Training & Evaluation
# Run EDA.py first before running this file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data from EDA.py
df = pd.read_csv('../Data/data_cleaned.csv')
print("Cleaned data loaded!")
print("Shape:", df.shape)

# Step 5: Train/Test Split
X = df.drop(columns=['price'])
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_test_actual = np.expm1(y_test)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("Preprocessing complete!")

# ============================================================
# Step 6: Random Forest (PRIMARY MODEL)
# ============================================================
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("\nRandom Forest trained!")

rf_pred = rf_model.predict(X_test)
rf_pred_actual = np.expm1(rf_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test_actual, rf_pred_actual))
rf_mae = mean_absolute_error(y_test_actual, rf_pred_actual)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print(f"R²  Score : {rf_r2:.4f}")
print(f"RMSE      : ${rf_rmse:,.2f}")
print(f"MAE       : ${rf_mae:,.2f}")

rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\nCross Validation R² (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# Feature Importance Chart
feature_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='steelblue')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Img/feature_importance.png', dpi=150)
plt.close()
print("Feature importance chart saved!")

# RF Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, rf_pred_actual,
            alpha=0.3, color='steelblue')
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Price - Random Forest')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Img/rf_actual_vs_predicted.png', dpi=150)
plt.close()
print("RF chart saved!")

# ============================================================
# Step 7: Decision Tree (COMPARISON MODEL)
# ============================================================
dt_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)
dt_model.fit(X_train, y_train)
print("\nDecision Tree trained!")

dt_pred = dt_model.predict(X_test)
dt_pred_actual = np.expm1(dt_pred)

dt_rmse = np.sqrt(mean_squared_error(y_test_actual, dt_pred_actual))
dt_mae = mean_absolute_error(y_test_actual, dt_pred_actual)
dt_r2 = r2_score(y_test, dt_pred)

print("\n--- Decision Tree Results ---")
print(f"R²  Score : {dt_r2:.4f}")
print(f"RMSE      : ${dt_rmse:,.2f}")
print(f"MAE       : ${dt_mae:,.2f}")

dt_cv = cross_val_score(dt_model, X, y, cv=5, scoring='r2')
print(f"\nCross Validation R² (5-fold): {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")

# DT Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, dt_pred_actual,
            alpha=0.3, color='purple')
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Price - Decision Tree')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Img/dt_actual_vs_predicted.png', dpi=150)
plt.close()
print("Decision Tree chart saved!")

# ============================================================
# Step 8: KNN (REJECTED MODEL)
# ============================================================
knn_model = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)
knn_model.fit(X_train_scaled, y_train)
print("\nKNN trained!")

knn_pred = knn_model.predict(X_test_scaled)
knn_pred_actual = np.expm1(knn_pred)
knn_pred_actual = np.clip(knn_pred_actual, 0, None)

knn_rmse = np.sqrt(mean_squared_error(y_test_actual, knn_pred_actual))
knn_mae = mean_absolute_error(y_test_actual, knn_pred_actual)
knn_r2 = r2_score(y_test, knn_pred)

print("\n--- KNN Results ---")
print(f"R²  Score : {knn_r2:.4f}")
print(f"RMSE      : ${knn_rmse:,.2f}")
print(f"MAE       : ${knn_mae:,.2f}")

knn_cv = cross_val_score(knn_model, X_train_scaled,
                          y_train, cv=5, scoring='r2')
print(f"\nCross Validation R² (5-fold): {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")

# KNN Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, knn_pred_actual,
            alpha=0.3, color='green')
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Price - KNN')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Img/knn_actual_vs_predicted.png', dpi=150)
plt.close()
print("KNN chart saved!")

# ============================================================
# Final Comparison — All 4 Models
# ============================================================
print("\n========== FINAL MODEL COMPARISON ==========")
print(f"{'Model':<20} {'R²':>8} {'RMSE':>15} {'MAE':>15}")
print("-" * 62)
print(f"{'Random Forest':<20} {rf_r2:>8.4f} ${rf_rmse:>14,.2f} ${rf_mae:>14,.2f}")
print(f"{'Decision Tree':<20} {dt_r2:>8.4f} ${dt_rmse:>14,.2f} ${dt_mae:>14,.2f}")
print(f"{'KNN':<20} {knn_r2:>8.4f} ${knn_rmse:>14,.2f} ${knn_mae:>14,.2f}")
print("=" * 62)
print("\n Best Model: Random Forest (lowest RMSE)")

# Final Comparison Chart
models = ['Random Forest', 'Decision Tree', 'KNN']
r2_scores = [rf_r2, dt_r2, knn_r2]
rmse_scores = [rf_rmse, dt_rmse, knn_rmse]
mae_scores = [rf_mae, dt_mae, knn_mae]
colors = ['steelblue', 'purple', 'green']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(models, r2_scores, color=colors)
axes[0].set_title('R² Score Comparison')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.01, f'{v:.4f}',
                 ha='center', fontsize=9)

axes[1].bar(models, rmse_scores, color=colors)
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('RMSE ($)')
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 1000, f'${v:,.0f}',
                 ha='center', fontsize=8)

axes[2].bar(models, mae_scores, color=colors)
axes[2].set_title('MAE Comparison')
axes[2].set_ylabel('MAE ($)')
for i, v in enumerate(mae_scores):
    axes[2].text(i, v + 1000, f'${v:,.0f}',
                 ha='center', fontsize=8)

plt.suptitle('Model Performance Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Img/final_model_comparison.png', dpi=150)
plt.close()
print("Final comparison chart saved!")
print("\n All models complete!")