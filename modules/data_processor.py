import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose

from modules.common import TimeSeries, DecompositionPart


class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            data_column: str = None,
            missing_method: str = None,
            rolling_window_size: int = None,
            target_frequency: str = None,
            frequency_method: str = None,
            anomaly_method: str = None,
            z_threshold: float = None
    ):
        self.data_column: str = data_column
        self.missing_method: str = missing_method
        self.rolling_window_size: int = rolling_window_size
        self.target_frequency: str = target_frequency
        self.frequency_method: str = frequency_method
        self.anomaly_method: str = anomaly_method
        self.z_threshold: float = z_threshold

    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.missing_method == 'mean':
            data[[self.data_column]] = data[[self.data_column]].fillna(data[self.data_column].mean())
        elif self.missing_method == 'ffill':
            data[[self.data_column]] = data[[self.data_column]].ffill()
        elif self.missing_method == 'rolling' and self.rolling_window_size:
            data[[self.data_column]] = data[[self.data_column]].fillna(
                data[[self.data_column]].rolling(self.rolling_window_size, min_periods=1).mean()
            )
        return data

    def resample_frequency(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.target_frequency and not data.empty:
            # Автоопределение текущей частоты
            current_frequency = pd.infer_freq(data.index)
            if current_frequency is None:
                # Частота не определена, можно попробовать найти "моду" разниц между датами
                deltas = data.index.to_series().diff().dropna()
                mode_delta = deltas.mode()[0]
                current_frequency = pd.tseries.frequencies.to_offset(mode_delta).freqstr
            current_offset = pd.tseries.frequencies.to_offset(current_frequency)
            target_offset = pd.tseries.frequencies.to_offset(self.target_frequency)

            if current_offset and target_offset:
                if target_offset > current_offset:
                    # снижение частоты
                    agg_func = self.frequency_method if self.frequency_method in ['mean', 'sum'] else 'mean'
                    data = data[[self.data_column]].resample(self.target_frequency).agg(agg_func)
                else:
                    # повышение частоты
                    data = data[[self.data_column]].resample(self.target_frequency).ffill().bfill()
        return data

    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.anomaly_method:
            return data

        if self.anomaly_method == 'zscore':
            zscore = (data[self.data_column] - data[self.data_column].mean()) / data[self.data_column].std()
            mask = zscore.abs() > self.z_threshold
            data.loc[mask, self.data_column] = np.nan
            # Заполняем пропуски после выбросов
            data[self.data_column] = data[self.data_column].ffill().bfill()

        elif self.anomaly_method == 'rolling' and self.rolling_window_size:
            rolling_mean = data[self.data_column].rolling(self.rolling_window_size, min_periods=1).mean()
            mask = ((data[self.data_column] - rolling_mean).abs() / data[self.data_column].std()) > self.z_threshold
            data.loc[mask, self.data_column] = rolling_mean[mask]

        elif self.anomaly_method == 'last':
            mask = (
                    ((data[self.data_column] - data[self.data_column].mean()).abs() / data[self.data_column].std())
                    > self.z_threshold
            )
            data.loc[mask, self.data_column] = data[self.data_column].ffill()[mask]
        return data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data = self.handle_missing(data)
        data = self.resample_frequency(data)
        data = self.handle_outliers(data)
        return data


class TimeSeriesProcessor:
    def __init__(
            self,
            time_series: TimeSeries = None,
            data_column_name: str = None,
            missing_method: str = None,
            rolling_window_size: int = None,
            target_frequency: str = None,
            frequency_method: str = None,
            anomaly_method: str = None,
            z_threshold: float = None,
            decompose_model: str = None,
            decompose_period: int = None,
            correlation_method: str = None,
            correlation_threshold: float = None,
            is_feature_selection: bool = None
    ):
        self.time_series: TimeSeries = time_series
        self.data_column_name: str = data_column_name
        self.missing_method: str = missing_method
        self.rolling_window_size: int = rolling_window_size
        self.target_frequency: str = target_frequency
        self.frequency_method: str = frequency_method
        self.anomaly_method: str = anomaly_method
        self.z_threshold: float = z_threshold
        self.decompose_model: str = decompose_model
        self.decompose_period: int = decompose_period
        self.correlation_method: str = correlation_method
        self.correlation_threshold: float = correlation_threshold
        self.is_feature_selection: bool = is_feature_selection

    def transform(self) -> None:
        transformer = TimeSeriesTransformer(
            data_column=self.data_column_name,
            missing_method=self.missing_method,
            rolling_window_size=self.rolling_window_size,
            target_frequency=self.target_frequency,
            frequency_method=self.frequency_method,
            anomaly_method=self.anomaly_method,
            z_threshold=self.z_threshold
        )
        self.time_series.data_processed = transformer.transform(self.time_series.data)

    def decompose(self) -> None:
        data_decomposed = seasonal_decompose(
            self.time_series.data_processed,
            model=self.decompose_model,
            period=self.decompose_period
        )
        self.time_series.data_processed[DecompositionPart.observed] = data_decomposed.observed
        self.time_series.data_processed[DecompositionPart.trend] = data_decomposed.trend
        self.time_series.data_processed[DecompositionPart.seasonal] = data_decomposed.seasonal
        self.time_series.data_processed[DecompositionPart.resid] = data_decomposed.resid
        self.time_series.data_processed.drop(self.data_column_name, axis=1, inplace=True)

        self.time_series.data_decomposed = self.time_series.data_processed[
            [
                DecompositionPart.observed,
                DecompositionPart.trend,
                DecompositionPart.seasonal,
                DecompositionPart.resid
            ]
        ].copy()

    def features_generation(self) -> None:
        # Календарные признаки
        self.time_series.data_processed['day'] = self.time_series.data_processed.index.day
        self.time_series.data_processed['dayofweek'] = self.time_series.data_processed.index.dayofweek
        self.time_series.data_processed['weekofyear'] = (
            self.time_series.data_processed.index.isocalendar().week.astype(int)
        )
        self.time_series.data_processed['month'] = self.time_series.data_processed.index.month
        self.time_series.data_processed['quarter'] = self.time_series.data_processed.index.quarter
        # Циклические признаки
        self.time_series.data_processed['dow_sin'] = (
            np.sin(2 * np.pi * self.time_series.data_processed.index.dayofweek / 7)
        )
        self.time_series.data_processed['dow_cos'] = (
            np.cos(2 * np.pi * self.time_series.data_processed.index.dayofweek / 7)
        )
        self.time_series.data_processed['month_sin'] = (
            np.sin(2 * np.pi * self.time_series.data_processed.index.month / 12)
        )
        self.time_series.data_processed['month_cos'] = (
            np.cos(2 * np.pi * self.time_series.data_processed.index.month / 12)
        )
        self.time_series.data_processed['dayofyear_sin'] = (
            np.sin(2 * np.pi * self.time_series.data_processed.index.dayofyear / 365)
        )
        self.time_series.data_processed['dayofyear_cos'] = (
            np.cos(2 * np.pi * self.time_series.data_processed.index.dayofyear / 365)
        )
        self.time_series.data_processed['hour_sin'] = (
            np.sin(2 * np.pi * self.time_series.data_processed.index.hour / 24)
        )
        self.time_series.data_processed['hour_cos'] = (
            np.cos(2 * np.pi * self.time_series.data_processed.index.hour / 24)
        )
        # Признаки сезонности
        # сила сезонного эффекта
        self.time_series.data_processed[f'{DecompositionPart.seasonal}__abs'] = (
            self.time_series.data_processed[DecompositionPart.seasonal].abs())
        # фаза сезонности
        self.time_series.data_processed[f'{DecompositionPart.seasonal}__phase'] = (
            self.time_series.data_processed[DecompositionPart.seasonal].rank(pct=True)
        )
        # дельта сезонности
        self.time_series.data_processed[f'{DecompositionPart.seasonal}__diff'] = (
            self.time_series.data_processed[DecompositionPart.seasonal].diff())
        # нормализованная сезонность
        self.time_series.data_processed[f'{DecompositionPart.seasonal}__scaled'] = (
                (
                        self.time_series.data_processed[DecompositionPart.seasonal]
                        - self.time_series.data_processed[DecompositionPart.seasonal].mean()
                )
                / self.time_series.data_processed[DecompositionPart.seasonal].std()
        )
        # Добавление нормализованного тренда
        trend = self.time_series.data_processed[DecompositionPart.trend]
        self.time_series.data_processed[f'{DecompositionPart.trend}__normalize'] = (
                (trend - trend.mean()) / trend.std(ddof=0)
        )

    def decompose_important_features_find(self) -> None:
        decompose_parts = [DecompositionPart.observed, DecompositionPart.trend, DecompositionPart.seasonal]
        for part in decompose_parts:
            # Корреляция признаков с данными одной из частей временного ряда
            correlation = self.time_series.data_processed.corr(method=self.correlation_method)[part]

            # Поиск значимых признаков
            selected_features = correlation.drop(labels=part) \
                .loc[lambda x: x.abs() > self.correlation_threshold].index.tolist()
            # Убираем данные декомпозиции
            selected_features = [
                feature for feature in selected_features
                if feature not in decompose_parts and feature != f'{DecompositionPart.trend}__normalize'
            ]

            self.time_series.important_features[part] = selected_features

    def features_processing(self) -> None:
        self.features_generation()
        self.decompose_important_features_find()

    def process(self) -> None:
        self.transform()
        self.decompose()
        self.features_processing()
