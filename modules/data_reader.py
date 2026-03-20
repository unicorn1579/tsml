import json
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd

from modules.common import TimeSeries


class InputParams(NamedTuple):
    """
    Входные параметры.

    Attributes:
        data_directory (str): Наименование директории файлов с данными.
        tickets_group (str): Наименование группы тикетов, совпадающих с названием файлов с данными.
        file_extension (str): Расширение файлов с данными.
        time_column (str): Наименование столбца с временными метками.
        data_columns (List[str]): Список обрабатываемых столбцов.
        missing_method (str): Метод заполнения пропусков.
        rolling_window_size (int): Размер окна скользящего среднего при обработки значений.
        target_frequency (str): Целевая гранулярность обрабатываемых столбцов.
        frequency_method (str): Метод обработки значений при изменении гранулярности.
        anomaly_method (str): Метод исправления выбросов.
        z_threshold (float): Порог z-score для выявления выбросов.
        decompose_model (str): Способ разложения временного ряда.
        decompose_period (int): Период временного ряда.
        correlation_method (str): Метод корреляции.
        correlation_threshold (float): Порог корреляции.
        is_feature_selection (bool): Флаг для отбора значимых атрибутов.
        cv_method (str): Метод кросс-валидации.
        cv_frequency (str): Гранулярность блоков данных при кросс-валидации.
        models_types (List[str]): Набор типов формируемых моделей.
        validation_ratio (float): Коэффициент валидационных данных при обучении модели XGBRegressor.
        n_estimators (int): Количество деревьев при обучении модели XGBRegressor.
        max_depth List[int]: Максимальная глубина деревьев при обучении модели XGBRegressor.
        learning_rate (float): Коэффициент скорости обучения модели XGBRegressor.
        eval_metric (str): Метрика оценки моделей и валидационных данных при обучении моделей.
        early_stopping_rounds (int): Итерации ранней остановки обучения моделей.
    """
    # reading
    data_directory: str = None
    tickets_group: str = None
    file_extension: str = None
    time_column: str = None
    data_columns: List[str] = None
    # processing
    missing_method: str = None
    rolling_window_size: int = None
    target_frequency: str = None
    frequency_method: str = None
    anomaly_method: str = None
    z_threshold: float = None
    decompose_model: str = None
    decompose_period: int = None
    correlation_method: str = None
    correlation_threshold: float = None
    is_feature_selection: bool = None
    # generate
    cv_method: str = None
    cv_frequency: str = None
    models_types: List[str] = None
    validation_ratio: float = None
    n_estimators: int = None
    max_depth: List[int] = None
    learning_rate: float = None
    eval_metric: str = None
    early_stopping_rounds: int = None

    def get_tickets_names(self) -> List[str]:
        with open('configuration.json', 'r') as file:
            configuration = json.load(file)
        return configuration['tickets_groups'][self.tickets_group]


class TimeSeriesReader:
    def __init__(
            self,
            data_directory: str,
            file_name: str,
            file_extension: str,
            time_column: str,
            data_column: str
    ) -> None:
        self.data_directory: str = data_directory
        self.file_name: str = file_name
        self.file_extension: str = file_extension
        self.time_column: str = time_column
        self.data_column: str = data_column

        self.file_path: Path = Path(data_directory) / f'{self.file_name}.{self.file_extension}'
        self.data: Optional[pd.DataFrame] = None
        self.time_series: Optional[TimeSeries] = None

    def get_data(self) -> None:
        if self.file_extension == 'csv':
            self.data = pd.read_csv(self.file_path)

    def read(self) -> None:
        self.get_data()

        self.data[self.time_column] = pd.to_datetime(self.data[self.time_column], errors='coerce', utc=True)
        self.data = self.data.sort_values(self.time_column).set_index(self.time_column)

        self.time_series = TimeSeries(
            ticket_name=self.file_name,
            data=self.data[[self.data_column]],
            data_column_name=self.data_column
        )
