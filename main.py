import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

from graphs import plot_roc_curve, plot_feature_importances

FILE_PATH = "dataset/covtype.data.gz"


def process_data(file_path):
    """
    Выполняет полный цикл обработки данных:
    - загрузка данных,
    - исследование данных,
    - предварительная обработка данных.

    :param file_path: Путь к файлу с данными.
    :return: Кортеж из четырех элементов: X_train, X_test, y_train, y_test.
    """
    data = pd.read_csv(file_path, header=None)

    print("Первые 5 строк датасета:")
    print(data.head())

    print("\nИнформация о датасете:")
    print(data.info())

    print("\nСтатистика по числовым признакам:")
    print(data.describe())

    # Предварительная обработка данных
    X = data.iloc[:, :-1]  # Все колонки, кроме последней, являются признаками
    y = data.iloc[:, -1]  # Последняя колонка - это целевая переменная

    # Преобразуем целевую переменную в бинарную (например, разделим по значению медианы)
    y = (y > y.median()).astype(int)

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    print("\nРазмеры обучающей и тестовой выборок:")
    print(f"Обучающая выборка: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Тестовая выборка: X_test: {X_test.shape}, y_test: {y_test.shape}\n")

    return X_train, X_test, y_train, y_test


def grid_search_model(model, param_grid, X_train, y_train):
    """
    Выполняет GridSearchCV для переданной модели.

    :param model: Модель классификатора.
    :param param_grid: Словарь с параметрами для GridSearch.
    :param X_train: Обучающие данные.
    :param y_train: Метки классов для обучающих данных.
    :return: Лучшая модель после поиска и результаты поиска.
    """
    start_time = time.time()
    print(f"Начало GridSearch для {model.__class__.__name__}...")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"GridSearch для {model.__class__.__name__} завершен за {elapsed_time:.2f} секунд.")
    print(f"Лучшие параметры для {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Лучший результат точности для {model.__class__.__name__}: {grid_search.best_score_}\n")

    return grid_search.best_estimator_, grid_search


def train_classifiers(X_train, y_train):
    """
    Строит классифицирующие модели с использованием GridSearch для алгоритмов RandomForest, LogisticRegression и XGBoost.

    :param X_train: Обучающие данные.
    :param y_train: Метки классов для обучающих данных.
    :return: Словарь с лучшими моделями.
    """
    models = {}

    print("Начало обучения моделей с регуляризацией...")

    # RandomForest
    print("Обучение RandomForest...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_rf, _ = grid_search_model(rf_model, rf_param_grid, X_train, y_train)
    models['RandomForest'] = best_rf

    # LogisticRegression
    print("Обучение LogisticRegression...")
    lr_model = LogisticRegression(max_iter=5000, random_state=42)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    best_lr, _ = grid_search_model(lr_model, lr_param_grid, X_train, y_train)
    models['LogisticRegression'] = best_lr

    # XGBoost
    print("Обучение XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'lambda': [0, 1, 10],
        'alpha': [0, 0.1, 1]
    }
    best_xgb, _ = grid_search_model(xgb_model, xgb_param_grid, X_train, y_train)
    models['XGBoost'] = best_xgb

    print("Обучение моделей завершено.\n")

    return models


def balance_dataset(X, y):
    """
    Балансировка датасета с помощью метода oversampling для меньшего класса.

    :param X: Признаки.
    :param y: Целевая переменная.
    :return: Сбалансированные X и y.
    """
    y = y.values

    # Соединяем признаки и целевую переменную
    X_y = np.hstack((X, y.reshape(-1, 1)))

    # Разделяем на два класса
    X_y_class_0 = X_y[y == 0]
    X_y_class_1 = X_y[y == 1]

    # Определяем количество элементов в большем классе
    n_samples = max(len(X_y_class_0), len(X_y_class_1))

    # Увеличиваем меньший класс до размера большего класса
    X_y_class_0_resampled = resample(X_y_class_0, replace=True, n_samples=n_samples, random_state=42)
    X_y_class_1_resampled = resample(X_y_class_1, replace=True, n_samples=n_samples, random_state=42)

    # Соединяем классы обратно
    X_y_balanced = np.vstack((X_y_class_0_resampled, X_y_class_1_resampled))
    np.random.shuffle(X_y_balanced)

    return X_y_balanced[:, :-1], X_y_balanced[:, -1].astype(int)


def calculate_roc_auc(models, X_test, y_test):
    """
    Рассчитывает значения ROC-кривой и AUC для каждой модели.

    :param models: Словарь с обученными моделями.
    :param X_test: Тестовые данные.
    :param y_test: Метки классов для тестовых данных.
    :return: Словарь с именами моделей и соответствующими ROC-кривыми и значениями AUC.
    """
    roc_data = {}
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    return roc_data


def calculate_feature_importances(model, X_train):
    """
    Рассчитывает значимость признаков для модели.

    :param model: Обученная модель.
    :param X_train: Признаки обучающей выборки.
    :return: Сортированный массив значимостей признаков и соответствующие имена признаков.
    """
    if hasattr(model, "coef_"):  # Для LogisticRegression
        importances = np.abs(model.coef_[0])
    else:  # Для моделей на деревьях
        importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    return importances[indices], [feature_names[i] for i in indices]


def perform_stratified_kfold(model, X, y, k=5):
    """
    Выполняет K-блочную стратифицированную проверку для указанной модели.

    :param model: Модель для обучения.
    :param X: Признаки.
    :param y: Целевая переменная.
    :param k: Количество фолдов для проверки.
    :return: Средние значения метрик точности, precision, recall и F1-score.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    print("-" * 30)
    print(f"Модель: {model.__class__.__name__}")
    print(f"Средняя точность: {np.mean(accuracy_scores):.4f}")
    print(f"Средняя Precision: {np.mean(precision_scores):.4f}")
    print(f"Средняя Recall: {np.mean(recall_scores):.4f}")
    print(f"Средняя F1-score: {np.mean(f1_scores):.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_data(FILE_PATH)

    # Балансировка выборки
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Обучение моделей на обычной выборке
    best_models_original = train_classifiers(X_train, y_train)

    # Обучение моделей на сбалансированной выборке
    best_models_balanced = train_classifiers(X_train_balanced, y_train_balanced)

    # Оценка на тестовых данных
    for model_name, model in best_models_original.items():
        print(f"\nОценка модели {model_name} на тестовых данных:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

    # Расчет ROC-кривых и AUC
    roc_data = calculate_roc_auc(best_models_original, X_test, y_test)

    # Построение и визуализация ROC-кривых
    plot_roc_curve(roc_data)

    # Расчет и сохранение значимости признаков для исходной выборки
    for model_name, model in best_models_original.items():
        importances, feature_names = calculate_feature_importances(model, X_train)
        plot_feature_importances(importances, feature_names, f"{model_name} Важность характеристик (Original Data)",
                                 f"graphs_img/{model_name}_feature_importances_original.png")

    # Расчет и сохранение значимости признаков для сбалансированной выборки
    for model_name, model in best_models_balanced.items():
        importances, feature_names = calculate_feature_importances(model, X_train_balanced)
        plot_feature_importances(importances, feature_names, f"{model_name} Важность характеристик (Balanced Data)",
                                 f"graphs_img/{model_name}_feature_importances_balanced.png")

    # Выполняем K-блочную стратифицированную проверку для каждой модели
    print("K-блочная стратифицированная проверка:")
    for model_name, model in best_models_original.items():
        perform_stratified_kfold(model, X_train, y_train, k=5)
    print("-" * 30)
