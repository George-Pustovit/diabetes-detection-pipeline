import pandas as pd 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('data/raw/diabetes.csv')

features  = df.columns
features =  features.drop("Outcome")
target  = 'Outcome'

for feature in features:
    df[feature] = df[feature].replace(0, df[feature].median())
    if df[feature].isna().sum() != 0:
        df[feature].fillna(df[feature].median(), inplace=True)

# Категориальные признаки

#Индекс инсулинорезистентности
df['HOMA_IR_Approx'] = (df['Glucose'] * df['Insulin']) / 405

# Соотношение глюкозы к ИМТ 
df['Glucose_to_BMI_Ratio'] = df['Glucose'] / df['BMI']

# Категоризация уровня глюкозы 
bins = [-np.inf, 100, 126, np.inf]
labels = ['Normal', 'Prediabetes', 'Diabetes']
df['Glucose_Category'] = pd.cut(df['Glucose'], bins=bins, labels=labels)

# Категоризация генетического риска
bins_pedigree = [-np.inf, 0.5, 1.0, np.inf]
labels_pedigree = ['Low Risk', 'Medium Risk', 'High Risk']
df['DiabetesPedigreeFunction_Category'] = pd.cut(df['DiabetesPedigreeFunction'], bins=bins_pedigree, labels=labels_pedigree)  # Генетический фактор риска

# Бинарный флаг гипертонии
df['Hypertension_Flag'] = (df['BloodPressure'] > 130).astype(int)  # Порог 130

# Бинарный флаг гиперинсулинемии
df['Hyperinsulinemia_Above_50'] = (df['Insulin'] > 50).astype(int)  # Порог 50

# Категоризация возраста
bins_age = [-np.inf, 35, 60, np.inf]
labels_age = ['Young', 'Middle-aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age)

# One-Hot Encoding для всех категориальных признаков
categorical_cols = ['Age_Group', 'Glucose_Category', 'DiabetesPedigreeFunction_Category','Age_Group']
df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)


numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                    'DiabetesPedigreeFunction', 'Age', 'HOMA_IR_Approx', 'Glucose_to_BMI_Ratio']  # Только непрерывные числовые

categorical_features = [col for col in df.columns if col not in numeric_features + [target]]  # Бинарные флаги + one-hot



X_numeric = df[numeric_features]
X_categorical = df[categorical_features]
y = df[target]

# Нормализация только числовых признаков
transformer = Normalizer().fit(X_numeric)
X_numeric_normalized = transformer.transform(X_numeric)
X_numeric_normalized = pd.DataFrame(X_numeric_normalized, columns=numeric_features)

# Объединение нормализованных числовых с категориальными (без нормализации)
X = pd.concat([X_numeric_normalized, X_categorical.reset_index(drop=True)], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

pd.concat([X_train, pd.DataFrame(y_train)], axis=1).to_csv('data/processed/train.csv', index=False)
pd.concat([X_test, pd.DataFrame(y_test)], axis=1).to_csv('data/processed/test.csv', index=False)