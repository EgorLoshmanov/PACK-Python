import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# --- 1 ---
df = pd.read_csv('wells_info_with_prod.csv')

# Преобразуем все даты в datetime (на всякий случай)
date_cols = ['PermitDate', 'SpudDate', 'CompletionDate', 'FirstProductionDate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Целевая переменная
Y = df['Prod1Year']
df = df.drop(['Prod1Year'], axis=1)

# Хотим оставить:
#   - одну дату: CompletionDate
#   - один категориальный признак: formation
date_keep = ['CompletionDate']
cat_keep = ['formation']

# Остальные категориальные/датовые будем кодировать one-hot'ом
cat_ohe_cols = ['FirstProductionDate', 'operatorNameIHS',
                'BasinName', 'StateName', 'CountyName']

# Удалим лишние даты, которые вообще не используем
drop_dates = ['PermitDate', 'SpudDate']
df = df.drop(columns=[c for c in drop_dates if c in df.columns])

# Отдельно берём данные для one-hot кодирования
ohe_input_cols = [c for c in cat_ohe_cols if c in df.columns]
cat_data_for_ohe = df[ohe_input_cols]

# Уберём их из df, чтобы там остались:
# - числовые признаки,
# - CompletionDate,
# - formation
df = df.drop(columns=ohe_input_cols, errors='ignore')

# One-hot кодирование
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_array = ohe.fit_transform(cat_data_for_ohe)
one_hot_cols = ohe.get_feature_names_out(ohe_input_cols)
one_hot_data = pd.DataFrame(one_hot_array, index=cat_data_for_ohe.index, columns=one_hot_cols)

# Итоговая матрица признаков X:
# числовые + CompletionDate + formation + one-hot
X = pd.concat([df, one_hot_data], axis=1)

# Проверим, что CompletionDate и formation остались в X
# print(X[['CompletionDate', 'formation']].head())
print(X)

# =========================
# --- 2 --- 
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =========================
# --- 3 --- 
# =========================
scaler_X = StandardScaler()

# Выбираем числовые столбцы (int/float, включая one-hot)
num_cols = X.select_dtypes(include=[np.number]).columns

# Отдельно берём числовую часть train/test
X_train_num = X_train[num_cols]
X_test_num = X_test[num_cols]

# Масштабируем только числовые признаки
X_train_num_scaled = scaler_X.fit_transform(X_train_num)
X_test_num_scaled = scaler_X.transform(X_test_num)

# Собираем обратно DataFrame, оставляя немасштабированные дату и категориальный признак
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = X_train_num_scaled
X_test_scaled[num_cols] = X_test_num_scaled

# Теперь в X_train_scaled / X_test_scaled:
# - CompletionDate и formation остались как есть
# - остальные числовые (включая one-hot) отмасштабированы

# =========================
# --- 4 --- 
# =========================
scaler_Y = StandardScaler()

Y_train_2d = Y_train.values.reshape(-1, 1)
Y_test_2d = Y_test.values.reshape(-1, 1)

Y_train_scaled = scaler_Y.fit_transform(Y_train_2d)
Y_test_scaled = scaler_Y.transform(Y_test_2d)

# Превращаем обратно в вектор (1D)
Y_train_scaled = Y_train_scaled.ravel()
Y_test_scaled = Y_test_scaled.ravel()

# Можно для контроля вывести формы:
# print(X_train_scaled.shape, X_test_scaled.shape)
# print(Y_train_scaled.shape, Y_test_scaled.shape)