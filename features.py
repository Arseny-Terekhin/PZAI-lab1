"""Подготовка признаков для задачи оттока клиентов Telco (ЛР1, вариант 1).

Функции вынесены в отдельный модуль, чтобы сохранённый пайплайн
(model.joblib) загружался в другом процессе (ЛР2).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

SERVICES = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PROTECTION = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']

# все новые признаки; список NEW_FEATURES используется в абляции
NEW_FEATURES = ['avg_monthly_charge', 'charge_increase', 'n_services',
                'contract_payment', 'fiber_no_protection', 'auto_payment',
                'tenure_bucket']


def clean(X):
    """Очистка сырых данных: приведение типов, удаление идентификатора."""
    X = X.copy()
    X = X.drop(columns=['customerID', 'Churn'], errors='ignore')
    # ловушка варианта: у новых клиентов (tenure = 0) TotalCharges — пустая строка
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X.loc[X['TotalCharges'].isna() & (X['tenure'] == 0), 'TotalCharges'] = 0.0
    X['SeniorCitizen'] = X['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).fillna(
        X['SeniorCitizen'].astype(str))
    return X


def add_features(X, drop=()):
    """Новые признаки; drop — имена признаков, которые не создавать (абляция)."""
    X = clean(X)
    tenure_safe = X['tenure'].replace(0, np.nan)
    # средний фактический платёж за месяц за всё время обслуживания
    X['avg_monthly_charge'] = (X['TotalCharges'] / tenure_safe).fillna(X['MonthlyCharges'])
    # насколько текущий тариф выше исторического среднего (рост цены)
    X['charge_increase'] = X['MonthlyCharges'] - X['avg_monthly_charge']
    # число подключённых услуг
    X['n_services'] = (X[SERVICES] == 'Yes').sum(axis=1) + (X['InternetService'] != 'No')
    # сочетание типа договора и способа оплаты
    X['contract_payment'] = X['Contract'] + ' | ' + X['PaymentMethod']
    # оптоволокно без единой услуги защиты/поддержки
    X['fiber_no_protection'] = ((X['InternetService'] == 'Fiber optic')
                                & (X[PROTECTION] != 'Yes').all(axis=1)).astype(int)
    # автоматический платёж
    X['auto_payment'] = X['PaymentMethod'].str.contains('automatic').astype(int)
    # стаж в годах (0–6), первый год — самый рискованный
    X['tenure_bucket'] = np.minimum(X['tenure'] // 12, 6).astype(int)
    return X.drop(columns=list(drop))


def to_category(X):
    """Строковые столбцы — в тип category (для LightGBM)."""
    X = X.copy()
    cols = X.select_dtypes(exclude='number').columns
    X[cols] = X[cols].astype(str).astype('category')
    return X


def to_catboost(X):
    """Строковые столбцы — в str без пропусков (для CatBoost)."""
    X = X.copy()
    cols = X.select_dtypes(exclude='number').columns
    for c in cols:
        X[c] = X[c].astype(object).where(X[c].notna(), 'missing').astype(str)
    return X


def cat_columns(X):
    return X.select_dtypes(exclude='number').columns.tolist()


class CategoryAligner(TransformerMixin, BaseEstimator):
    """Переводит строковые столбцы в category с категориями, запомненными на fit.

    Гарантирует одинаковое кодирование категорий на обучении и на новых данных;
    неизвестные значения становятся NaN (LightGBM обрабатывает пропуски сам).
    """

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique())
                            for c in cat_columns(X)}
        return self

    def transform(self, X):
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return X


class CatBoostCat(ClassifierMixin, BaseEstimator):
    """Обёртка CatBoostClassifier для sklearn: категориальные столбцы определяются при fit.

    Нужна, потому что CatBoostClassifier с параметром cat_features не клонируется
    в sklearn (CalibratedClassifierCV, cross_val_score).
    """

    def __init__(self, params=None):
        self.params = params

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.cat_features_ = cat_columns(X)
        self.model_ = CatBoostClassifier(cat_features=self.cat_features_, **(self.params or {}))
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
