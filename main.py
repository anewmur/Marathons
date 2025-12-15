"""
Прогнозирование времени финиша марафонцев.

Класс MarathonModel управляет пайплайном:
загрузка → предобработка → эталоны → возрастная модель → прогноз
"""

import logging
from pathlib import Path
from reporting import dump_references_report
import yaml
import pandas as pd

from data_loader import DataLoader
from preprocessor import Preprocessor
from reference_builder import TraceReferenceBuilder
from age_reference_builder import AgeReferenceBuilder
from time_utils import print_age_references_pretty, analyze_age_group

logger = logging.getLogger(__name__)


class MarathonModel:
    """
    Модель прогнозирования времени финиша марафонцев.

    Attributes:
        data_path: Путь к папке с Excel файлами
        validation_year: Год для валидации
        verbose: Флаг подробного вывода
        config: Параметры из config.yaml
        df: Сырые данные после загрузки
        df_clean: Данные после предобработки
        references: DataFrame с эталонами трасс (race_id, gender, reference_time, ...)
        age_references: dict[str, pd.DataFrame] - возрастные эталоны по трассам
    """

    def __init__(
        self,
        data_path: str,
        validation_year: int = 2025,
        verbose: bool = True
    ):
        """
        Инициализация модели.

        Args:
            data_path: Путь к папке с Excel файлами или к файлу
            validation_year: Год для валидации
            verbose: Выводить подробные сообщения
        """
        self.data_path = Path(data_path)
        self.validation_year = validation_year
        self.verbose = verbose

        # Загрузить конфигурацию
        self.config = self._load_config()

        # Компоненты пайплайна
        self._data_loader = DataLoader()

        # Данные — заполняются по мере выполнения шагов
        self.df: pd.DataFrame | None = None
        self.df_clean: pd.DataFrame | None = None

        # Флаги выполнения шагов
        self._steps_completed = {
            'load_data': False,
            'preprocess': False,
            'build_references': False,
            'fit_age_model': False,
        }

    def _load_config(self) -> dict:
        """
        Загрузить конфигурацию из config.yaml.

        Returns:
            Словарь с параметрами
        """
        config_path = Path(__file__).parent / 'config.yaml'

        if not config_path.exists():
            logger.warning(
                "config.yaml не найден, используются значения по умолчанию"
            )
            return {
                'preprocessing': {'age_center': 35, 'age_scale': 10},
                'validation': {'year': 2025}
            }

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _check_step(self, step_name: str, required_steps: list[str]) -> None:
        """
        Проверить что необходимые шаги выполнены.

        Args:
            step_name: Название текущего шага
            required_steps: Список требуемых шагов

        Raises:
            RuntimeError: Требуемый шаг не выполнен
        """
        for req in required_steps:
            if not self._steps_completed.get(req, False):
                raise RuntimeError(
                    f"Шаг '{step_name}' требует выполнения '{req}'. "
                    f"Сначала вызовите model.{req}()"
                )

    # ==================== ШАГ 1: ЗАГРУЗКА ====================

    def load_data(self) -> 'MarathonModel':
        """
        Загрузить данные из Excel файлов.

        После выполнения доступен self.df с колонками:
        runner_id, race_id, year, gender, age, time_seconds, status, city, bib_number

        Returns:
            self для цепочки вызовов
        """
        logger.debug(f"\n{'='*60}")
        logger.debug("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
        logger.debug(f"{'='*60}")
        logger.debug(f"Источник: {self.data_path}")

        if self.data_path.is_file():
            raise FileNotFoundError(f"Нужно указать путь до папки: {self.data_path}")
        elif self.data_path.is_dir():
            self.df = self._data_loader.load_directory_cached(self.data_path)
        else:
            raise FileNotFoundError(f"Путь не существует: {self.data_path}")

        if self.df.empty:
            raise ValueError("Не удалось загрузить данные")

        self._steps_completed['load_data'] = True
        self._print_load_stats()

        return self

    def _print_load_stats(self) -> None:
        """Вывести статистику после загрузки."""
        if not self.verbose or self.df is None:
            return

        logger.info(f"Всего записей: {len(self.df)}")
        logger.info(f"Финишировали (OK): {(self.df['status'] == 'OK').sum()}")
        logger.info(f"DNF/DNS/DSQ: {(self.df['status'] != 'OK').sum()}")
        logger.info(f"Уникальных участников: {self.df['runner_id'].nunique()}")
        logger.info(f"Трасс: {self.df['race_id'].nunique()}")
        logger.info(f"Годы: {sorted(int(y) for y in self.df['year'].unique())}")

        gender_counts = self.df.groupby('gender')['runner_id'].count()
        for gender, count in gender_counts.items():
            logger.info(f"  {gender}: {count}")

    # ==================== ШАГ 2: ПРЕДОБРАБОТКА ====================

    def preprocess(self) -> 'MarathonModel':
        """
        Предобработка данных.

        Выполняет:
        - Фильтрацию финишировавших (status='OK')
        - Разукрупнение однофамильцев по году рождения
        - Заполнение города внутри кластеров
        - Добавление Y = ln(time_seconds)
        - Добавление x = (age - age_center) / age_scale
        - Удаление дублей

        Returns:
            self для цепочки вызовов
        """
        self._check_step('preprocess', ['load_data'])

        logger.debug(f"\n{'='*60}")
        logger.debug("ШАГ 2: ПРЕДОБРАБОТКА")
        logger.debug(f"{'='*60}")

        preprocessor = Preprocessor(self.config)
        self.df_clean = preprocessor.run(self.df)

        self._steps_completed['preprocess'] = True
        logger.debug(f"После предобработки: {len(self.df_clean)} записей")

        return self

    # ==================== ШАГ 3: ЭТАЛОНЫ ====================

    def build_trace_references(self) -> 'MarathonModel':
        """
        Построить эталоны R_{c,g} для каждой пары (трасса, пол).

        Returns:
            self для цепочки вызовов
        """
        self._check_step("build_trace_references", ["load_data", "preprocess"])

        validation_config = self.config.get("validation", {})
        validation_year = int(validation_config.get("year", 0))

        if validation_year > 0:
            loy_frame = self.df_clean[self.df_clean["year"] != validation_year].copy()
        else:
            loy_frame = self.df_clean.copy()

        builder = TraceReferenceBuilder(self.config)
        self.references = builder.build(loy_frame)

        self._steps_completed["build_trace_references"] = True
        if self.verbose:
            logger.info("Построено эталонов трасс: %d", len(self.references))

        return self

    def build_age_references(self) -> "MarathonModel":
        """
        Построить возрастные медианы отдельно по каждой трассе.

        После выполнения доступен self.age_references: dict[str, pd.DataFrame]
        Ключ - название трассы, значение - DataFrame с колонками:
        gender, age, age_median_time, age_median_log, age_median_var, age_median_std, n_total
        """
        self._check_step("build_age_references", ["load_data", "preprocess"])

        validation_config = self.config.get("validation", {})
        validation_year = int(validation_config.get("year", 0))

        if validation_year > 0:
            loy_frame = self.df_clean[self.df_clean["year"] != validation_year].copy()
        else:
            loy_frame = self.df_clean.copy()

        builder = AgeReferenceBuilder(self.config)

        references_by_race: dict[str, pd.DataFrame] = {}

        if not loy_frame.empty:
            for race_id_value, race_group in loy_frame.groupby("race_id", sort=False):
                race_table = builder.build(race_group)

                if race_table.empty:
                    continue

                race_table = race_table.rename(columns={
                    "median_time": "age_median_time",
                    "median_log": "age_median_log",
                    "median_std": "age_median_std",
                }).reset_index(drop=True)

                race_table["age_median_var"] = race_table["age_median_std"] ** 2

                references_by_race[str(race_id_value)] = race_table

        self.age_references = references_by_race
        self._steps_completed["build_age_references"] = True

        if self.verbose:
            total_rows = sum(len(dataframe) for dataframe in references_by_race.values())
            logger.info(
                "Построено возрастных таблиц: %d трасс, %d строк",
                len(references_by_race),
                total_rows
            )

        return self

    def get_age_references(self, race_id: str) -> pd.DataFrame | None:
        """
        Получить возрастные эталоны для конкретной трассы.

        Args:
            race_id: Название трассы
        Returns:
            DataFrame с колонками: gender, age, age_median_time, age_median_log,
        """
        return self.age_references.get(race_id)

    # ==================== ШАГ 4: ВОЗРАСТНАЯ МОДЕЛЬ ====================

    def fit_age_model(self) -> 'MarathonModel':
        """
        Обучить возрастную модель (B-сплайны с REML).

        Returns:
            self для цепочки вызовов
        """
        self._check_step('fit_age_model', ['load_data', 'preprocess', 'build_trace_references'])

        logger.debug(f"\n{'='*60}")
        logger.debug("ШАГ 4: ВОЗРАСТНАЯ МОДЕЛЬ")
        logger.debug(f"{'='*60}")

        print("TODO: fit_age_model — реализация в age_model.py")

        self._steps_completed['fit_age_model'] = True

        return self

    # ==================== ПОЛНЫЙ ПАЙПЛАЙН ====================

    def run(self) -> 'MarathonModel':
        """
        Выполнить все шаги пайплайна.

        Returns:
            self для цепочки вызовов
        """
        self.load_data()
        self.preprocess()
        self.build_trace_references()
        self.build_age_references()
        self.fit_age_model()

        logger.debug(f"\n{'='*60}")
        logger.debug("ПАЙПЛАЙН ЗАВЕРШЁН")
        logger.debug(f"{'='*60}")

        return self

    # ==================== ПРОГНОЗИРОВАНИЕ ====================

    def predict(
        self,
        runner_id: str,
        race_id: str,
        year: int,
        age: int | None = None
    ) -> dict[str, float] | None:
        """
        Прогноз времени финиша для участника.

        Args:
            runner_id: Идентификатор участника ("Фамилия_Имя")
            race_id: Название трассы
            year: Год забега
            age: Возраст (если None — берётся из истории)

        Returns:
            Словарь с median, q05, q95 или None
        """
        self._check_step('predict', ['load_data', 'preprocess', 'build_trace_references', 'fit_age_model'])

        print("TODO: predict — здесь будет прогноз")

        return None

    # ==================== ВСПОМОГАТЕЛЬНЫЕ ====================

    def summary(self) -> str:
        """
        Текстовое описание состояния модели.

        Returns:
            Строка с описанием
        """
        lines = [
            "MarathonModel Summary",
            "=" * 40,
            f"Data path: {self.data_path}",
            f"Validation year: {self.validation_year}",
            f"Config: age_center={self.config['preprocessing']['age_center']}, "
            f"age_scale={self.config['preprocessing']['age_scale']}",
            "",
            "Steps completed:"
        ]

        for step, done in self._steps_completed.items():
            status = "✓" if done else "✗"
            lines.append(f"  [{status}] {step}")

        if self.df is not None:
            lines.extend([
                "",
                f"Data loaded: {len(self.df)} records",
                f"Unique runners: {self.df['runner_id'].nunique()}",
                f"Races: {self.df['race_id'].nunique()}",
            ])

        if self.df_clean is not None:
            lines.append(f"After preprocessing: {len(self.df_clean)} records")

        if hasattr(self, 'references') and self.references is not None:
            lines.extend([
                "",
                f"Trace references: {len(self.references)} rows",
            ])

        if hasattr(self, 'age_references') and self.age_references:
            total_rows = sum(len(df) for df in self.age_references.values())
            lines.extend([
                "",
                f"Age references: {len(self.age_references)} races, {total_rows} rows",
                f"Available races: {', '.join(self.age_references.keys())}",
            ])

        return "\n".join(lines)



    def __repr__(self) -> str:
        completed = sum(self._steps_completed.values())
        total = len(self._steps_completed)
        return f"MarathonModel(data_path='{self.data_path}', steps={completed}/{total})"


if __name__ == "__main__":
    from logging_setup import easy_logging
    easy_logging(True)

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        validation_year=2025,
        verbose=True
    )
    model.run()

    dump_references_report(model, r"C:\Users\andre\github\Marathons\references_dump.txt")

    print("\n" + "="*60)
    print(model.summary())

    print("\n" + "="*60)
    print("Trace references:")
    print(model.references.head())

    print("\n" + "="*60)
    print("Age references (by race):")
    for race_id, df in model.age_references.items():
        print(f"\n{race_id}:")
        print(df.head())
    df = model.age_references['Белые ночи']
    print_age_references_pretty(df)
    analyze_age_group(df, 'F', 23)

    # Женское время лучшее 2:28:35 Пермски2 2020