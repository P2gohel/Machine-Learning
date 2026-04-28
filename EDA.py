# EDA.py - Exploratory Data Analysis & Feature Engineering
# Run this file first

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.makedirs('Img', exist_ok=True)
print("All libraries imported successfully!")

# Step 2: Load & Explore Dataset
df = pd.read_csv('data.csv')
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
plt.subplot(2, 2, 1)
plt.scatter(df['sqft_living'], df['price'],
            alpha=0.3, color='steelblue')
plt.title('Price vs Living Area')
plt.xlabel('sqft_living')
plt.ylabel('Price ($)')
plt.subplot(2, 2, 2)
df.boxplot(column='price', by='bedrooms', ax=plt.gca())
plt.title('Price vs Bedrooms')
plt.suptitle('')
plt.subplot(2, 2, 3)
df.boxplot(column='price', by='condition', ax=plt.gca())
plt.title('Price vs Condition')
plt.suptitle('')
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

# Step 4: Feature Engineering & Cleaning
df = df[df['price'] > 50000]
df = df[df['price'] < 5000000]
print("Valid rows after price cleaning:", len(df))

df = df.drop(columns=['date', 'street', 'statezip', 'country'])
print("Dropped irrelevant columns")

df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['yr_renovated'])
print("Engineered was_renovated feature")

df = df[df['sqft_living'] > 0]
df = df[df['bedrooms'] > 0]
df = df[df['bedrooms'] < 15]
print("Removed outliers")

df = pd.get_dummies(df, columns=['city'], drop_first=True)
print("Encoded city column")

df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
df['sqft_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
df['property_age'] = 2015 - df['yr_built']
df['rooms_total'] = df['bedrooms'] + df['bathrooms']
print("New features engineered!")

print("\nFinal Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())

# Save cleaned data for model.py
df.to_csv('data_cleaned.csv', index=False)
print("\n✅ Cleaned data saved as data_cleaned.csv")
print("✅ EDA Complete! Now run model.py")