import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import r2_score, make_scorer
import time
import warnings
warnings.filterwarnings('ignore')


def calculate_rmsle(y_true, y_pred):
    """
    Расчет Root Mean Squared Logarithmic Error (RMSLE)
    """
    y_pred = np.maximum(y_pred, 0)
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    squared_log_error = (log_pred - log_true) ** 2
    rmsle = np.sqrt(np.mean(squared_log_error))
    return rmsle


def rmsle_scorer(y_true, y_pred):
    """Scorer для использования RandomizedSearchCV"""
    return -calculate_rmsle(y_true, y_pred)  # Негативное значение, т.к. scorer максимизирует


def preprocess_data(df, is_train=True):
    """
    Предобработка данных
    """
    df_processed = df.copy()

    # 1. Создание новых признаков
    df_processed['Volume'] = (df_processed['Length'] *
                               df_processed['Diameter'] *
                               df_processed['Height'])

    df_processed['Surface_Area'] = 2 * (
        df_processed['Length'] * df_processed['Diameter'] +
        df_processed['Length'] * df_processed['Height'] +
        df_processed['Diameter'] * df_processed['Height']
    )

    df_processed['Length_to_Diameter'] = df_processed['Length'] / (df_processed['Diameter'] + 1e-6)
    df_processed['Height_to_Length'] = df_processed['Height'] / (df_processed['Length'] + 1e-6)
    df_processed['Height_to_Diameter'] = df_processed['Height'] / (df_processed['Diameter'] + 1e-6)

    df_processed['Meat_ratio'] = df_processed['Whole weight.1'] / (df_processed['Whole weight'] + 1e-6)
    df_processed['Viscera_ratio'] = df_processed['Whole weight.2'] / (df_processed['Whole weight'] + 1e-6)
    df_processed['Shell_ratio'] = df_processed['Shell weight'] / (df_processed['Whole weight'] + 1e-6)

    df_processed['Density'] = df_processed['Whole weight'] / (df_processed['Volume'] + 1e-6)

    df_processed['Meat_total'] = (df_processed['Whole weight.1'] +
                                   df_processed['Whole weight.2'])

    # 2. One-Hot кодирование категориального признака Sex
    sex_dummies = pd.get_dummies(df_processed['Sex'], prefix='Sex', drop_first=False)
    df_processed = pd.concat([df_processed, sex_dummies], axis=1)
    df_processed.drop('Sex', axis=1, inplace=True)

    # 3. Удаление коррелирующих признаков
    features_to_drop = ['Length', 'Whole weight.1', 'Whole weight.2']
    df_processed.drop(features_to_drop, axis=1, inplace=True)

    # 4. Удаление id
    if 'id' in df_processed.columns:
        df_processed.drop('id', axis=1, inplace=True)

    return df_processed


def evaluate_model(model, X_train, X_val, y_train, y_val):
    """Оценка модели на train и validation"""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_rmsle = calculate_rmsle(y_train, y_train_pred)
    val_rmsle = calculate_rmsle(y_val, y_val_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    return {
        'train_rmsle': train_rmsle,
        'val_rmsle': val_rmsle,
        'train_r2': train_r2,
        'val_r2': val_r2
    }


def random_search_tuning(X_train, y_train, X_val, y_val, n_iter=50, cv=3):
    """
    Подбор гиперпараметров с использованием RandomizedSearchCV
    """

    # Определяем пространство гиперпараметров для XGBoost
    param_distributions = {
        'n_estimators': [500, 750, 1000, 1250, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        'max_depth': [4, 6, 8, 10, 12],
        'min_child_weight': [1, 3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    print(f"\nПространство поиска:")
    for param, values in param_distributions.items():
        print(f"  {param}: {values}")

    print(f"\nПараметры поиска:")
    print(f"  - Количество итераций: {n_iter}")
    print(f"  - Cross-validation folds: {cv}")
    print(f"  - Метрика: RMSLE (негативная для максимизации)")

    # Создаем базовую модель XGBoost
    xgboost = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='rmse'
    )

    # Создаем scorer для RMSLE
    rmsle_score = make_scorer(rmsle_scorer, greater_is_better=True)

    # Запускаем RandomizedSearchCV
    start_time = time.time()

    random_search = RandomizedSearchCV(
        estimator=xgboost,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=rmsle_score,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nВремя выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")

    # Лучшие параметры
    print("\n" + "="*70)
    print("ЛУЧШИЕ ПАРАМЕТРЫ (Random Search):")
    print("="*70)
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nЛучший RMSLE (CV): {-random_search.best_score_:.4f}")

    # Оценка лучшей модели на валидации
    best_model = random_search.best_estimator_
    metrics = evaluate_model(best_model, X_train, X_val, y_train, y_val)

    print("\n" + "="*70)
    print("МЕТРИКИ ЛУЧШЕЙ МОДЕЛИ:")
    print("="*70)
    print(f"Train RMSLE: {metrics['train_rmsle']:.4f}")
    print(f"Val RMSLE:   {metrics['val_rmsle']:.4f}")
    print(f"Train R²:    {metrics['train_r2']:.4f}")
    print(f"Val R²:      {metrics['val_r2']:.4f}")

    # Топ-10 лучших комбинаций
    print("\n" + "="*70)
    print("ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ:")
    print("="*70)
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df['mean_test_rmsle'] = -results_df['mean_test_score']
    top_10 = results_df.nsmallest(10, 'mean_test_rmsle')[
        ['params', 'mean_test_rmsle', 'std_test_score', 'rank_test_score']
    ]

    for idx, row in top_10.iterrows():
        print(f"\nРанг {row['rank_test_score']}:")
        print(f"  RMSLE: {row['mean_test_rmsle']:.4f} (±{row['std_test_score']:.4f})")
        print(f"  Параметры: {row['params']}")

    return random_search, best_model, metrics


def main():
    # 1. Загрузка данных
    train_df = pd.read_csv('data/train.csv')

    # 2. Предобработка
    train_processed = preprocess_data(train_df, is_train=True)

    # 3. Разделение на X и y
    X = train_processed.drop('Rings', axis=1)
    y = train_processed['Rings']

    # 4. Разделение на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Random Search
    random_search, best_random_model, random_metrics = random_search_tuning(
        X_train, y_train, X_val, y_val,
        n_iter=50,  #
        cv=3
    )

    # 6. Сохранение результатов
    results = {
        'random_search_best_params': random_search.best_params_,
        'random_search_metrics': random_metrics,
        'random_search_cv_score': -random_search.best_score_
    }

    # Сохраняем результаты в файл
    results_df = pd.DataFrame([results])
    results_df.to_csv('hyperparameter_tuning_xgboost_results.csv', index=False)

    # Сохраняем все результаты CV в детальный файл
    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.to_csv('random_search_xgboost_cv_results.csv', index=False)

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)
    print("\nИспользуйте лучшие параметры в вашей модели:")
    print("="*70)
    for param, value in random_search.best_params_.items():
        print(f"  {param}={value},")
    print("="*70)

if __name__ == "__main__":
    main()