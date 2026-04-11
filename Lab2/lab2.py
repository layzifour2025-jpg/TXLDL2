import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
sender_password = "vakyllievezqudra"


warnings.filterwarnings('ignore')

print("="*50)
print("BÀI 1: KHÁM PHÁ DỮ LIỆU HOUSING")
print("="*50)

df_housing = pd.read_csv('ITA105_Lab_2_Housing.csv')
print("Shape của dữ liệu Housing:", df_housing.shape)
print("Số lượng Missing values:\n", df_housing.isnull().sum())

print("\nThống kê mô tả Housing:\n", df_housing.describe())

numeric_cols_housing = ['dien_tich', 'gia', 'so_phong']
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols_housing, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df_housing[col])
    plt.title(f'Housing Boxplot - {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_housing, x='dien_tich', y='gia')
plt.title('Housing: Diện tích vs Giá')
plt.show()

print("\nSo sánh số lượng ngoại lệ (Housing):")
for col in numeric_cols_housing:
    # IQR
    Q1 = df_housing[col].quantile(0.25)
    Q3 = df_housing[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = df_housing[(df_housing[col] < lower_bound) | (df_housing[col] > upper_bound)].shape[0]
    
    # Z-score
    z_scores = np.abs(stats.zscore(df_housing[col].dropna()))
    outliers_z = (z_scores > 3).sum()
    
    print(f"- [{col}] Theo IQR: {outliers_iqr} | Theo Z-score (>3): {outliers_z}")

df_housing_clean = df_housing.copy()
for col in numeric_cols_housing:
    lower = df_housing_clean[col].quantile(0.01)
    upper = df_housing_clean[col].quantile(0.99)
    df_housing_clean[col] = np.clip(df_housing_clean[col], lower, upper)

plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols_housing, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df_housing_clean[col], color='lightgreen')
    plt.title(f'Housing Cleaned - {col}')
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("BÀI 2: PHÁT HIỆN NGOẠI LỆ TRONG DỮ LIỆU IOT")
print("="*50)

df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv', parse_dates=['timestamp'])
df_iot.set_index('timestamp', inplace=True)
print("Missing values in IoT:\n", df_iot.isnull().sum())

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_iot, x=df_iot.index, y='temperature', hue='sensor_id')
plt.title('IoT: Temperature over Time by Sensor')
plt.show()

df_iot['temp_roll_mean'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
df_iot['temp_roll_std'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(window=10, min_periods=1).std())

df_iot['is_outlier_rolling'] = (df_iot['temperature'] > df_iot['temp_roll_mean'] + 3 * df_iot['temp_roll_std']) | \
                               (df_iot['temperature'] < df_iot['temp_roll_mean'] - 3 * df_iot['temp_roll_std'])

df_iot['temp_zscore'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: np.abs(stats.zscore(x, nan_policy='omit')))
df_iot['is_outlier_z'] = df_iot['temp_zscore'] > 3

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_iot, x='temperature', y='pressure', hue='is_outlier_z', palette={False: 'blue', True: 'red'})
plt.title('IoT: Temperature vs Pressure (Z-score Outliers Highlighted)')
plt.show()

print(f"Số ngoại lệ IoT theo Rolling Mean: {df_iot['is_outlier_rolling'].sum()}")
print(f"Số ngoại lệ IoT theo Z-score: {df_iot['is_outlier_z'].sum()}")

df_iot_clean = df_iot.copy()
df_iot_clean.loc[df_iot_clean['is_outlier_z'], 'temperature'] = np.nan
df_iot_clean['temperature'] = df_iot_clean.groupby('sensor_id')['temperature'].transform(lambda x: x.interpolate())


print("\n" + "="*50)
print("BÀI 3: NGOẠI LỆ TRONG GIAO DỊCH E-COMMERCE")
print("="*50)

df_ecom = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')
print("\nThống kê mô tả E-commerce:\n", df_ecom.describe())

plt.figure(figsize=(12, 4))
for i, col in enumerate(['price', 'quantity', 'rating'], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df_ecom[col])
    plt.title(f'E-com Boxplot - {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_ecom, x='price', y='quantity')
plt.title('E-com: Price vs Quantity')
plt.show()


df_ecom_clean = df_ecom[(df_ecom['price'] > 0) & (df_ecom['rating'] <= 5.0) & (df_ecom['rating'] >= 0)].copy()

df_ecom_clean['price_log'] = np.log1p(df_ecom_clean['price'])

plt.figure(figsize=(8, 4))
sns.boxplot(y=df_ecom_clean['price_log'], color='orange')
plt.title('E-com: Boxplot of Price (Log Transformed)')
plt.show()


print("\n" + "="*50)
print("BÀI 4: MULTIVARIATE OUTLIER (NGOẠI LỆ ĐA BIẾN)")
print("="*50)

z_price = np.abs(stats.zscore(df_ecom_clean['price']))
z_qty = np.abs(stats.zscore(df_ecom_clean['quantity']))

df_ecom_clean['multivariate_outlier'] = (z_price > 2.5) & (z_qty > 2.5)

print(f"Số lượng ngoại lệ đa biến (Price & Quantity Z>2.5): {df_ecom_clean['multivariate_outlier'].sum()}")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_ecom_clean, x='price', y='quantity', 
                hue='multivariate_outlier', palette={False: 'gray', True: 'red'})
plt.title('E-com Multivariate Outliers: Price vs Quantity')
plt.show()

print("\nHoàn tất chạy mã cho toàn bộ Lab 2!")