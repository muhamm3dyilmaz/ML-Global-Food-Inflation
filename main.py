import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri kümesini okur
df = pd.read_csv('WLD_RTFP_country_2023-10-02.csv')

# Tarih sütununu datetime türüne dönüştürür
df['date'] = pd.to_datetime(df['date'])

# Eksik değerleri kontrol eder
print("Eksik Değer Sayısı: ", df['Inflation'].isnull().sum())

# Eksik değerleri ortalama değer ile doldurur
df['Inflation'] = df['Inflation'].fillna(df['Inflation'].mean())

# 'Inflation' sütununu sınıflandırmak için yeni bir sütun ekler
df['Inflation_Class'] = pd.cut(df['Inflation'], bins=[-np.inf, -0.01, 0.01, np.inf], labels=['Decrease', 'Stable', 'Increase'])

# Eğitim ve test veri setlerini ayırır
features = ['Open', 'High', 'Low', 'Close']
X = df[features]
y = df['Inflation_Class']

# Eksik değerleri doldurur
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturur ve eğitir
model = RandomForestClassifier(n_estimators=125, max_depth=10, min_samples_split=2, min_samples_leaf=1)

model.fit(X_train, y_train)

# Test verileri üzerinde tahminler yapar
y_pred_test = model.predict(X_test)



# Tahminlerin dağılımını görselleştirir
plt.figure(figsize=(10, 6))
plt.hist(y_pred_test, bins=np.arange(-0.5, 2.5, 1), color='green', alpha=0.7, rwidth=0.8, label='Tahminler')
plt.xticks([0, 1, 2], ['Increase','Decrease',  'Stable'])
plt.title('Tahminlerin Dağılımı')
plt.xlabel('Enflasyon Sınıfı')
plt.ylabel('Frekans')
plt.legend()
plt.show()

# Hata metriklerini hesaplar
accuracy = accuracy_score(y_test, y_pred_test)
#conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

print(f'Accuracy: {accuracy}')
#print('Confusion Matrix:')
#print(conf_matrix)
print('Classification Report:')
print(class_report)

# Test seti için scatter plot ile tahminleri görselleştirir
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, features.index('Close')], y_test, color='blue', label='Gerçek Değer')
plt.scatter(X_test[:, features.index('Close')], y_pred_test, color='orange', alpha=0.5, label='Tahmin (Test Seti)')
plt.title('Inflation Tahminleri (Test Seti)')
plt.xlabel('Close Value')
plt.ylabel('Inflation')
plt.legend()
plt.show()
