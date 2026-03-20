from itertools import chain
from typing import List

from joblib import Parallel, delayed

import modules.data_processor as dp
import modules.data_reader as dr
import modules.model_generator as mg
from modules.common import TimeSeries


class TimeSeriesPipeline:

    @staticmethod
    def time_series_read(params: dr.InputParams, ticket_name: str) -> List[TimeSeries]:
        # Инициализация временных рядов
        time_series: List[TimeSeries] = list()
        for data_column in params.data_columns:
            ts_reader = dr.TimeSeriesReader(
                data_directory=params.data_directory,
                file_name=ticket_name,
                file_extension=params.file_extension,
                time_column=params.time_column,
                data_column=data_column
            )
            ts_reader.read()
            time_series.append(ts_reader.time_series)
        return time_series

    @staticmethod
    def time_series_read_multiple(params: dr.InputParams) -> List[TimeSeries]:
        # Инициализация временных рядов. Цикл
        time_series_read: List[TimeSeries] = list()
        for ticket_name in params.get_tickets_names():
            time_series_read.extend(TimeSeriesPipeline.time_series_read(params=params, ticket_name=ticket_name))
        return time_series_read

    @staticmethod
    def time_series_read_parallel(params: dr.InputParams) -> List[TimeSeries]:
        # Инициализация временных рядов. Параллелизм
        time_series_read: List[List[TimeSeries]] = list()
        time_series_read.extend(
            Parallel(n_jobs=-1, backend='threading', verbose=10)(
                delayed(TimeSeriesPipeline.time_series_read)(params=params, ticket_name=ticket_name)
                for ticket_name in params.get_tickets_names()
            )
        )
        return list(chain.from_iterable(time_series_read))

    @staticmethod
    def time_series_process(params: dr.InputParams, time_series: TimeSeries) -> TimeSeries:
        # Подготовка данных временных рядов
        ts_processor = dp.TimeSeriesProcessor(
            time_series=time_series,
            data_column_name=time_series.data_column_name,
            missing_method=params.missing_method,
            rolling_window_size=params.rolling_window_size,
            target_frequency=params.target_frequency,
            frequency_method=params.frequency_method,
            anomaly_method=params.anomaly_method,
            z_threshold=params.z_threshold,
            decompose_model=params.decompose_model,
            decompose_period=params.decompose_period,
            correlation_method=params.correlation_method,
            correlation_threshold=params.correlation_threshold,
            is_feature_selection=params.is_feature_selection
        )
        ts_processor.process()
        return time_series

    @staticmethod
    def time_series_process_multiple(params: dr.InputParams, time_series: List[TimeSeries]) -> List[TimeSeries]:
        # Подготовка данных временных рядов. Цикл
        time_series_processed: List[TimeSeries] = list()
        for ts in time_series:
            time_series_processed.append(TimeSeriesPipeline.time_series_process(params=params, time_series=ts))
        return time_series_processed

    @staticmethod
    def time_series_process_parallel(params: dr.InputParams, time_series: List[TimeSeries]) -> List[TimeSeries]:
        # Подготовка данных временных рядов. Параллелизм
        time_series_processed: List[TimeSeries] = Parallel(n_jobs=-1, backend='threading', verbose=10)(
            delayed(TimeSeriesPipeline.time_series_process)(params=params, time_series=ts)
            for ts in time_series
        )
        return time_series_processed

    @staticmethod
    def model_generate(params: dr.InputParams, time_series: TimeSeries) -> TimeSeries:
        # Генерация групп моделей для каждого временного ряда
        ts_generator = mg.TimeSeriesModelGenerator(
            time_series=time_series,
            cv_method=params.cv_method,
            cv_frequency=params.cv_frequency,
            models_types=params.models_types,
            validation_ratio=params.validation_ratio,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            eval_metric=params.eval_metric,
            early_stopping_rounds=params.early_stopping_rounds
        )
        ts_generator.generate()
        return time_series

    @staticmethod
    def model_generate_multiple(params: dr.InputParams, time_series: List[TimeSeries]) -> List[TimeSeries]:
        # Генерация групп моделей для каждого временного ряда. Цикл
        time_series_models: List[TimeSeries] = list()
        for ts in time_series:
            time_series_models.append(TimeSeriesPipeline.model_generate(params=params, time_series=ts))
        return time_series_models

    @staticmethod
    def model_generate_parallel(params: dr.InputParams, time_series: List[TimeSeries]) -> List[TimeSeries]:
        # Генерация групп моделей для каждого временного ряда. Параллелизм
        time_series_models: List[TimeSeries] = Parallel(n_jobs=-1, backend='threading', verbose=10)(
            delayed(TimeSeriesPipeline.model_generate)(params=params, time_series=ts)
            for ts in time_series
        )
        return time_series_models
