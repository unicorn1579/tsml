from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor


class ModelType(NamedTuple):
    linear_regression = 'linear_regression'
    xgb_regressor = 'xgb_regressor'
    decompose = 'decompose'

@dataclass
class ModelParams:
    max_depth: int = None

class DecompositionPart(NamedTuple):
    observed = 'observed'
    trend = 'trend'
    seasonal = 'seasonal'
    resid = 'resid'


class MetricName(Enum):
    mae = 'mae'
    mape = 'mape'
    wape = 'wape'
    fa_mape = 'fa_mape'
    fa_wape = 'fa_wape'


@dataclass
class Metrics:
    observation_data: pd.Series = None
    prediction_data: pd.Series = None
    values: Dict[Optional[Union[MetricName, str]], float] = None

    def __post_init__(self):
        self.values = {
            MetricName.mae.value: None,
            MetricName.mape.value: None,
            MetricName.wape.value: None,
            MetricName.fa_mape.value: None,
            MetricName.fa_wape.value: None
        }
        if self.observation_data is not None and self.prediction_data is not None:
            self.values = {
                MetricName.mae.value: mean_absolute_error(self.observation_data, self.prediction_data),
                MetricName.mape.value: mean_absolute_percentage_error(self.observation_data, self.prediction_data),
                MetricName.wape.value: np.sum(np.abs(self.observation_data - self.prediction_data)) / np.sum(
                    self.observation_data),
                MetricName.fa_mape.value: 1 - mean_absolute_percentage_error(self.observation_data, self.prediction_data),
                MetricName.fa_wape.value: 1 - np.sum(np.abs(self.observation_data - self.prediction_data)) / np.sum(
                    self.observation_data)
            }

    def get_metrics_names(self) -> List[str]:
         return list(self.values.keys())

@dataclass
class ModelData:
    model_type: ModelType = None
    model: Optional[Union[LinearRegression, XGBRegressor]] = None
    x_columns_names: Optional[List[str]] = None
    y_column_name: Optional[Union[str, DecompositionPart]] = None
    metrics: Metrics = None

    def predict(self, x_data: pd.DataFrame) -> Union[pd.Series, None]:
        return self.model.predict(x_data[self.x_columns_names])


@dataclass
class DecomposeModelData(ModelData):
    models_data: Dict[DecompositionPart, ModelData] = None
    model_type: ModelType = ModelType.decompose
    model = None
    y_column_name = None

    def predict(self, x_data: pd.DataFrame) -> Union[pd.Series, None]:
        y_prediction = None
        for model_data in self.models_data.values():
            if model_data.model is None:
                continue
            prediction = model_data.predict(x_data)
            if y_prediction is None:
                y_prediction = prediction
            else:
                y_prediction = y_prediction + prediction
        return y_prediction


@dataclass
class TimeSeries:
    ticket_name: str = None
    data: pd.DataFrame = None
    data_column_name: str = None
    data_processed: pd.DataFrame = None
    data_decomposed: pd.DataFrame = None
    models_data: Dict[ModelType, ModelData] = None
    important_features: Dict[DecompositionPart, List[str]] = field(default_factory=dict)


class TimeSeriesPlot:

    @staticmethod
    def plot_source_data(time_series: TimeSeries) -> None:
        if time_series.data is None or time_series.data.empty:
            return

        plt.figure(figsize=(15, 5))
        plt.plot(
            time_series.data.index,
            time_series.data[time_series.data_column_name],
            label="Data",
            color="red"
        )
        plt.title(f"{time_series.ticket_name} — Data Time Series")
        plt.xlabel("Time")
        plt.ylabel(time_series.data_column_name)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_processed_data(time_series: TimeSeries, column_name: str = None) -> None:
        if time_series.data_processed is None or time_series.data_processed.empty:
            return

        if column_name:
            columns = [column_name]
        else:
            columns = list(time_series.data_processed.columns)

        n = len(columns)

        fig, axes = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, column in zip(axes, columns):
            ax.plot(
                time_series.data_processed.index,
                time_series.data_processed[column],
                label=column
            )
            ax.set_title(column)
            ax.grid(True)
            ax.legend()

        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_decomposed_data(time_series: TimeSeries) -> None:
        if time_series.data_decomposed is None or time_series.data_decomposed.empty:
            return

        columns = list(time_series.data_decomposed.columns)

        n = len(columns)

        fig, axes = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, column in zip(axes, columns):
            ax.plot(
                time_series.data_decomposed.index,
                time_series.data_decomposed[column],
                label=column
            )
            ax.set_title(column)
            ax.grid(True)
            ax.legend()

        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_processed_prediction_data(time_series: TimeSeries) -> None:
        if time_series.data_processed is None or time_series.data_processed.empty or not time_series.models_data:
            return
        y_column = DecompositionPart.observed

        if y_column not in time_series.data_processed.columns:
            return
        data = time_series.data_processed.dropna()

        y_observation = data[y_column]
        x_data = data.drop(columns=[y_column])

        plt.figure(figsize=(15, 5))
        plt.plot(
            data.index,
            y_observation,
            label="Observed",
            color="black",
            linewidth=2,
            alpha=0.7
        )

        for model_type, model_data in time_series.models_data.items():
            if model_data is None:
                continue
            y_prediction = model_data.predict(x_data)

            if y_prediction is None:
                continue

            metrics = Metrics(observation_data=y_observation, prediction_data=y_prediction)

            metrics_str = ", ".join(
                f"{name}={value:.4f}"
                for name, value in metrics.values.items()
                if value is not None
            )

            plt.plot(
                data.index,
                y_prediction,
                label=f"{model_type} ({metrics_str})",
                alpha=0.8
            )

        plt.title(f"{time_series.ticket_name} — Model Predictions vs Observed")
        plt.xlabel("Time")
        plt.ylabel(y_column)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_validate_data(time_series: TimeSeries, time_series_validate: TimeSeries) -> None:
        if (
                time_series_validate.data_processed
                is None or time_series_validate.data_processed.empty
                or not time_series.models_data
        ):
            return
        y_column = DecompositionPart.observed
        if y_column not in time_series_validate.data_processed.columns:
            return
        data = time_series_validate.data_processed.dropna()

        y_observation = data[y_column]
        x_data = data.drop(columns=[y_column])
        plt.figure(figsize=(15, 5))
        plt.plot(
            data.index,
            y_observation,
            label="Observed",
            color="black",
            linewidth=2,
            alpha=0.7
        )

        for model_type, model_data in time_series.models_data.items():
            if model_data is None:
                continue
            y_prediction = model_data.predict(x_data)

            if y_prediction is None:
                continue

            metrics = Metrics(observation_data=y_observation, prediction_data=y_prediction)

            metrics_str = ", ".join(
                f"{name}={value:.4f}"
                for name, value in metrics.values.items()
                if value is not None
            )

            plt.plot(
                data.index,
                y_prediction,
                label=f"{model_type} ({metrics_str})",
                alpha=0.8
            )

        plt.title(f"{time_series.ticket_name} — Model Predictions vs Observed")
        plt.xlabel("Time")
        plt.ylabel(y_column)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
