from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from modules.common import (TimeSeries, ModelData, ModelType, DecompositionPart,
                            MetricName, ModelParams, Metrics, DecomposeModelData)


class TimeSeriesModelGenerator:
    def __init__(
            self,
            time_series: TimeSeries = None,
            cv_method: str = None,
            cv_frequency: str = None,
            models_types: List[str] = None,
            validation_ratio: float = None,
            n_estimators: int = None,
            max_depth: List[int] = None,
            learning_rate: float = None,
            eval_metric: str = None,
            early_stopping_rounds: int = None
    ):
        self.time_series: TimeSeries = time_series
        self.cv_method: str = cv_method
        self.cv_frequency: str = cv_frequency
        self.models_types: List[str] = models_types
        # xgb_regressor
        self.validation_ratio: float = validation_ratio
        self.n_estimators: int = n_estimators
        self.max_depth: List[int] = max_depth
        self.learning_rate: float = learning_rate
        self.eval_metric: str = eval_metric
        self.early_stopping_rounds: int = early_stopping_rounds

    def generate_splits(self, data: pd.DataFrame):
        data.index = data.index.tz_localize(None)  # убираем временной пояс
        periods = data.index.to_period(self.cv_frequency)
        unique_periods = periods.unique()

        for i in range(1, len(unique_periods)):

            if self.cv_method == "expanding":
                train_mask = periods < unique_periods[i]
                test_mask = periods == unique_periods[i]

            elif self.cv_method == "sliding":
                train_mask = periods == unique_periods[i - 1]
                test_mask = periods == unique_periods[i]

            yield data.loc[train_mask], data.loc[test_mask]

    def linear_regression_fit_prediction(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame
    ) -> pd.Series:
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model.predict(x_test)

    def xgb_regressor_fit_prediction(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame,
            params: ModelParams
    ) -> pd.Series:
        split_idx = int(len(x_train) * self.validation_ratio)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=params.max_depth,
            learning_rate=self.learning_rate,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds
        )

        model.fit(
            x_train.iloc[:split_idx],
            y_train.iloc[:split_idx],
            eval_set=[(x_train.iloc[split_idx:], y_train.iloc[split_idx:])],
            verbose=False
        )

        return model.predict(x_test)

    def xgb_regressor_get_param_grid(self):
        for depth in self.max_depth:
            yield ModelParams(max_depth=depth)

    def linear_regression_generate(
            self,
            y_column_name: Optional[Union[str, DecompositionPart]],
            data: pd.DataFrame
    ) -> ModelData:
        scores = []
        for train, test in self.generate_splits(data):
            x_train = train.drop(columns=[y_column_name])
            y_train = train[y_column_name]
            x_test = test.drop(columns=[y_column_name])
            y_test = test[y_column_name]

            y_prediction = self.linear_regression_fit_prediction(x_train, y_train, x_test)
            scores.append(Metrics(observation_data=y_test, prediction_data=y_prediction))

        metrics = Metrics()
        metrics_values = {metric_name: [] for metric_name in metrics.get_metrics_names()}
        for metrics in scores:
            for metric_name in metrics.get_metrics_names():
                metrics_values[metric_name].append(metrics.values[metric_name])
        metrics.values = {metric_name: np.mean(values) for metric_name, values in metrics_values.items()}

        x_all = data.drop(columns=[y_column_name])
        y_all = data[y_column_name]

        model = LinearRegression()
        model.fit(x_all, y_all)

        return ModelData(
            model_type=ModelType.linear_regression,
            model=model,
            x_columns_names=list(x_all.columns),
            y_column_name=y_column_name,
            metrics=metrics
        )

    def xgb_regressor_generate(
            self,
            y_column_name: Optional[Union[str, DecompositionPart]],
            data: pd.DataFrame
    ) -> ModelData:

        best_eval_metric: Optional[float] = None
        best_metrics: Optional[Metrics] = None
        best_params: Optional[ModelParams] = None
        for params in self.xgb_regressor_get_param_grid():
            scores = []
            for train, test in self.generate_splits(data):
                x_train = train.drop(columns=[y_column_name])
                y_train = train[y_column_name]
                x_test = test.drop(columns=[y_column_name])
                y_test = test[y_column_name]

                y_prediction = self.xgb_regressor_fit_prediction(x_train, y_train, x_test, params)
                scores.append(Metrics(observation_data=y_test, prediction_data=y_prediction))
            metrics = Metrics()
            metrics_values = {metric_name: [] for metric_name in metrics.get_metrics_names()}
            for metrics in scores:
                for metric_name in metrics.get_metrics_names():
                    metrics_values[metric_name].append(metrics.values[metric_name])
            metrics.values = {metric_name: np.mean(values) for metric_name, values in metrics_values.items()}

            if best_eval_metric is None or (
                    self.eval_metric in [MetricName.mae, MetricName.mape, MetricName.wape]
                    and metrics.values[self.eval_metric] < best_eval_metric
            ) or (
                    self.eval_metric not in [MetricName.mae, MetricName.mape, MetricName.wape]
                    and metrics.values[self.eval_metric] > best_eval_metric
            ):
                best_eval_metric = metrics.values[self.eval_metric]
                best_params = params
                best_metrics = metrics

        # финальное обучение
        x_all = data.drop(columns=[y_column_name])
        y_all = data[y_column_name]

        split_idx = int(len(x_all) * self.validation_ratio)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=best_params.max_depth,
            learning_rate=self.learning_rate,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds
        )

        model.fit(
            x_all.iloc[:split_idx],
            y_all.iloc[:split_idx],
            eval_set=[(x_all.iloc[split_idx:], y_all.iloc[split_idx:])],
            verbose=False
        )

        return ModelData(
            model_type=ModelType.xgb_regressor,
            model=model,
            x_columns_names=list(x_all.columns),
            y_column_name=y_column_name,
            metrics=best_metrics
        )

    def generate_model(self, y_column_name: Optional[Union[str, DecompositionPart]]) -> List[ModelData]:
        features = self.time_series.important_features[y_column_name] + [y_column_name]
        data = self.time_series.data_processed[features].dropna()

        results: List[ModelData] = []

        if ModelType.linear_regression in self.models_types:
            results.append(self.linear_regression_generate(y_column_name=y_column_name, data=data))

        if ModelType.xgb_regressor in self.models_types:
            results.append(self.xgb_regressor_generate(y_column_name=y_column_name, data=data))

        return results

    def get_best_model(self, models_data: List[ModelData]) -> ModelData:
        best_model_data = None
        for model_data in models_data:
            if not best_model_data:
                best_model_data = model_data
            else:
                if self.eval_metric in [MetricName.mae, MetricName.mape, MetricName.wape]:
                    if model_data.metrics.values[self.eval_metric] < best_model_data.metrics.values[self.eval_metric]:
                        best_model_data = model_data
                elif self.eval_metric in [MetricName.fa_mape, MetricName.fa_wape]:
                    if model_data.metrics.values[self.eval_metric] > best_model_data.metrics.values[self.eval_metric]:
                        best_model_data = model_data

        return best_model_data

    def generate_decompose_model(self) -> DecomposeModelData:
        models_by_part = {}

        for part in [
            DecompositionPart.trend,
            DecompositionPart.seasonal
        ]:
            models = self.generate_model(y_column_name=part)
            best_model = self.get_best_model(models)
            models_by_part[part] = best_model

        return DecomposeModelData(models_data=models_by_part)

    def generate(self) -> None:
        models_dict = {}

        # модели на исходный ряд
        models = self.generate_model(DecompositionPart.observed)

        for model in models:
            models_dict[model.model_type] = model

        # модель на данных декомпозиции
        if ModelType.decompose in self.models_types:
            decompose_model = self.generate_decompose_model()
            models_dict[ModelType.decompose] = decompose_model

        self.time_series.models_data = models_dict
