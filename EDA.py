# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# Step 2: Load & Explore Dataset
df = pd.read_csv('data.csv')

# Basic info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 3a: Price Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=50, color='steelblue')
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), bins=50, color='coral')
plt.title('Log Price Distribution')
plt.xlabel('Log Price')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('Img/price_distribution.png', dpi=150)
plt.close()  
print("Price distribution saved!")

# Step 3b: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', 
            cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('Img/correlation_heatmap.png', dpi=150)
plt.close()  
print("Correlation heatmap saved!")

# Step 3c: Price vs Key Features
plt.figure(figsize=(12, 10))

# Price vs sqft_living
plt.subplot(2, 2, 1)
plt.scatter(df['sqft_living'], df['price'], 
            alpha=0.3, color='steelblue')
plt.title('Price vs Living Area')
plt.xlabel('sqft_living')
plt.ylabel('Price ($)')

# Price vs bedrooms
plt.subplot(2, 2, 2)
df.boxplot(column='price', by='bedrooms', ax=plt.gca())
plt.title('Price vs Bedrooms')
plt.suptitle('')

# Price vs condition
plt.subplot(2, 2, 3)
df.boxplot(column='price', by='condition', ax=plt.gca())
plt.title('Price vs Condition')
plt.suptitle('')

# Price vs waterfront
plt.subplot(2, 2, 4)
df.boxplot(column='price', by='waterfront', ax=plt.gca())
plt.title('Price vs Waterfront')
plt.suptitle('')

plt.tight_layout()
plt.savefig('Img/price_vs_features.png', dpi=150)
plt.close()  
print("Price vs features saved!")

# Step 3d: Average Price by City (Top 15)
city_price = df.groupby('city')['price'].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 6))
city_price.plot(kind='bar', color='steelblue')
plt.title('Average House Price by City (Top 15)')
plt.xlabel('City')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Img/price_by_city.png', dpi=150)
plt.close() 
print("City price chart saved!")

# 4a: Remove invalid prices first
df = df[df['price'] > 50000]  # Remove unrealistic prices
df = df[df['price'] < 5000000]  # Remove extreme outliers
print("Valid rows after price cleaning:", len(df))

# 4b: Drop irrelevant features
df = df.drop(columns=['date', 'street', 'statezip', 'country'])
print("Dropped irrelevant columns")

# 4c: Engineer yr_renovated → was_renovated
df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['yr_renovated'])
print("Engineered was_renovated feature")

# 4d: Remove outliers in sqft_living
df = df[df['sqft_living'] > 0]
df = df[df['bedrooms'] > 0]
df = df[df['bedrooms'] < 15]  # Remove unrealistic bedroom counts
print("Removed outliers")

# 4e: One-hot encode city
df = pd.get_dummies(df, columns=['city'], drop_first=True)
print("Encoded city column")

print("\nFinal Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())

# Step 5: Train/Test Split & Preprocessing

# 5a: Define features and target
X = df.drop(columns=['price'])
y = df['price']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# 5b: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# 5c: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing complete!")

# Step 6: Random Forest Model

# 6a: Train
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("\nRandom Forest trained!")

# 6b: Predictions
rf_pred = rf_model.predict(X_test)

# 6c: Evaluate
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print(f"R²  Score : {rf_r2:.4f}")
print(f"RMSE      : ${rf_rmse:,.2f}")
print(f"MAE       : ${rf_mae:,.2f}")

# 6d: Cross Validation
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\nCross Validation R² (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# 6e: Feature Importance Chart
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

# 6f: Actual vs Predicted Chart
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred, alpha=0.3, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Price - Random Forest')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Img/rf_actual_vs_predicted.png', dpi=150)
plt.close()
print("Actual vs Predicted chart saved!")
