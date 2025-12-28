import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")


# =========================
# Метрики
# =========================
def calculate_rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error (RMSLE)

    Важно:
    - прогнозы < 0 приравниваем к 0, чтобы log1p был определён
    """
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)

    squared_log_error = (log_pred - log_true) ** 2
    return float(np.sqrt(np.mean(squared_log_error)))


RMSLE_SCORER = make_scorer(calculate_rmsle, greater_is_better=False)


# =========================
# Предобработка
# =========================
def preprocess_data(df, is_train=True):
    """
    Предобработка данных:
    - добавление признаков
    - удаление коррелирующих признаков
    - удаление id 
    """
    df_processed = df.copy()

    # 1) Новые признаки
    df_processed["Volume"] = (
        df_processed["Length"] * df_processed["Diameter"] * df_processed["Height"]
    )

    df_processed["Surface_Area"] = 2 * (
        df_processed["Length"] * df_processed["Diameter"]
        + df_processed["Length"] * df_processed["Height"]
        + df_processed["Diameter"] * df_processed["Height"]
    )

    df_processed["Length_to_Diameter"] = df_processed["Length"] / (
        df_processed["Diameter"] + 1e-6
    )
    df_processed["Height_to_Length"] = df_processed["Height"] / (
        df_processed["Length"] + 1e-6
    )
    df_processed["Height_to_Diameter"] = df_processed["Height"] / (
        df_processed["Diameter"] + 1e-6
    )

    df_processed["Meat_ratio"] = df_processed["Whole weight.1"] / (
        df_processed["Whole weight"] + 1e-6
    )
    df_processed["Viscera_ratio"] = df_processed["Whole weight.2"] / (
        df_processed["Whole weight"] + 1e-6
    )
    df_processed["Shell_ratio"] = df_processed["Shell weight"] / (
        df_processed["Whole weight"] + 1e-6
    )

    df_processed["Density"] = df_processed["Whole weight"] / (
        df_processed["Volume"] + 1e-6
    )

    df_processed["Meat_total"] = df_processed["Whole weight.1"] + df_processed["Whole weight.2"]

    # 2) Удаление коррелирующих/лишних признаков
    features_to_drop = ["Length", "Whole weight.1", "Whole weight.2"]
    for col in features_to_drop:
        if col in df_processed.columns:
            df_processed.drop(col, axis=1, inplace=True)

    # 3) Удаление id
    if "id" in df_processed.columns:
        df_processed.drop("id", axis=1, inplace=True)

    return df_processed


# =========================
# Оценка модели
# =========================
def evaluate_model(model, X_train, X_val, y_train, y_val):
    """Оценка модели на train и validation."""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_rmsle = calculate_rmsle(y_train, y_train_pred)
    val_rmsle = calculate_rmsle(y_val, y_val_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    return {
        "train_rmsle": train_rmsle,
        "val_rmsle": val_rmsle,
        "train_r2": train_r2,
        "val_r2": val_r2,
    }

def random_search_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_features,
    n_iter=50,
    cv=3,
):
    """
    Подбор гиперпараметров через RandomizedSearchCV.
    """

    param_distributions = {
        "iterations": [500, 750, 1000, 1250, 1500],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        "depth": [4, 6, 8, 10, 12],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "min_data_in_leaf": [1, 5, 10, 20],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bylevel": [0.5, 0.7, 0.9, 1.0],
        "border_count": [32, 64, 128, 254],
    }

    print("\nПространство поиска:")
    for param, values in param_distributions.items():
        print(f"  {param}: {values}")

    print("\nПараметры поиска:")
    print(f"  - Количество итераций: {n_iter}")
    print(f"  - Cross-validation folds: {cv}")
    print(f"  - Метрика: RMSLE (меньше = лучше)")
    print(f"  - cat_features (indices): {cat_features}")

    # Базовая модель для поиска
    base_model_for_search = CatBoostRegressor(
        random_seed=42,
        verbose=0,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_features,
    )

    random_search = RandomizedSearchCV(
        estimator=base_model_for_search,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=RMSLE_SCORER,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True,
    )

    start_time = time.time()

    random_search.fit(X_train, y_train, cat_features=cat_features)

    elapsed_time = time.time() - start_time
    print(f"\nВремя выполнения: {elapsed_time:.2f} секунд ({elapsed_time / 60:.2f} минут)")

    print("\n" + "=" * 70)
    print("ЛУЧШИЕ ПАРАМЕТРЫ (Random Search):")
    print("=" * 70)
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    best_cv_rmsle = -random_search.best_score_
    print(f"\nЛучший RMSLE (CV): {best_cv_rmsle:.4f}")


    best_params = dict(random_search.best_params_)

    final_model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        verbose=200,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_features,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    final_model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
    )

    metrics = evaluate_model(final_model, X_train, X_val, y_train, y_val)

    print("\n" + "=" * 70)
    print("МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ (на hold-out validation):")
    print("=" * 70)
    print(f"Train RMSLE: {metrics['train_rmsle']:.4f}")
    print(f"Val RMSLE:   {metrics['val_rmsle']:.4f}")
    print(f"Train R²:    {metrics['train_r2']:.4f}")
    print(f"Val R²:      {metrics['val_r2']:.4f}")

    # Топ-10 лучших комбинаций
    print("\n" + "=" * 70)
    print("ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ (CV):")
    print("=" * 70)

    results_df = pd.DataFrame(random_search.cv_results_)
    # mean_test_score = negative RMSLE
    results_df["mean_test_rmsle"] = -results_df["mean_test_score"]

    top_10 = results_df.nsmallest(10, "mean_test_rmsle")[
        ["params", "mean_test_rmsle", "std_test_score", "rank_test_score"]
    ]

    for _, row in top_10.iterrows():
        print(f"\nРанг {int(row['rank_test_score'])}:")
        print(f"  RMSLE: {row['mean_test_rmsle']:.4f} (std(score)={row['std_test_score']:.4f})")
        print(f"  Параметры: {row['params']}")

    return random_search, final_model, metrics


def main():
    # 1) Загрузка данных
    train_df = pd.read_csv("data/train.csv")

    # 2) Предобработка
    train_processed = preprocess_data(train_df, is_train=True)

    # 3) Разделение на X и y
    X = train_processed.drop("Rings", axis=1)
    y = train_processed["Rings"]

    # 4) Категориальные признаки
    cat_cols = ["Sex"]
    cat_features_indices = [X.columns.get_loc(col) for col in cat_cols]

    # 5) Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Random Search
    random_search, best_final_model, best_metrics = random_search_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_features=cat_features_indices,
        n_iter=50,
        cv=3,
    )

    # 7) Сохранение результатов
    results = {
        "random_search_best_params": random_search.best_params_,
        "random_search_metrics": best_metrics,
        "random_search_cv_rmsle": float(-random_search.best_score_),
    }

    pd.DataFrame([results]).to_csv("hyperparameter_tuning_catboost_results.csv", index=False)

    pd.DataFrame(random_search.cv_results_).to_csv("random_search_catboost_cv_results.csv", index=False)

    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)

    print("\nЛучшие параметры для CatBoost:")
    print("=" * 70)
    for param, value in random_search.best_params_.items():
        print(f"  {param}={value},")
    print("=" * 70)


if __name__ == "__main__":
    main()
