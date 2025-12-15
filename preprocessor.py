"""
Предобработка сырых данных марафонов перед построением модели.

Класс Preprocessor выполняет:
  1) проверку структуры данных;
  2) фильтрацию по статусу "OK";
  3) разукрупнение однофамильцев по приблизительному году рождения
     и присвоение стабильных runner_id;
  4) заполнение городов внутри кластеров одного человека;
  5) добавление преобразованных признаков Y и x;
  6) удаление дублей.
"""

from __future__ import annotations

import logging
import time
import numpy as np
import pandas as pd

from utils import split_runner_id
from dictionaries import CLUSTER_CITY_CANONICAL_MAP

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Предобработка сырых данных марафонов.

    Attributes:
        config: Словарь конфигурации с секцией preprocessing
        age_center: Центр шкалы возраста для стандартизации
        age_scale: Масштаб шкалы возраста для стандартизации
    """

    REQUIRED_COLUMNS = [
        "runner_id",
        "race_id",
        "year",
        "gender",
        "age",
        "time_seconds",
        "status",
    ]

    def __init__(self, config: dict[str, object]) -> None:
        """
        Инициализация препроцессора.

        Args:
            config: Словарь с секцией preprocessing (age_center, age_scale)
        """
        self.config = config

        preprocessing_config = config.get("preprocessing", {})
        try:
            self.age_center = float(preprocessing_config["age_center"])
            self.age_scale = float(preprocessing_config["age_scale"])
        except KeyError as error:
            raise KeyError(
                "В config['preprocessing'] должны быть заданы "
                "'age_center' и 'age_scale'"
            ) from error

        if self.age_scale == 0:
            raise ValueError("preprocessing.age_scale не должен быть равен нулю")

    import time

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Полный конвейер предобработки.
        """
        logger.debug("Старт предобработки")

        pipeline_start = time.perf_counter()

        step_start = time.perf_counter()
        validated = self.validate_dataframe(dataframe)
        step_end = time.perf_counter()
        print(f"validate_dataframe: {step_end - step_start:.6f} s")

        step_start = time.perf_counter()
        filtered = self.filter_ok_status(validated)
        step_end = time.perf_counter()
        print(f"filter_ok_status: {step_end - step_start:.6f} s")

        step_start = time.perf_counter()
        processed = self.disambiguate_runners_and_fill_city(filtered)
        step_end = time.perf_counter()
        print(f"disambiguate_runners_and_fill_city: {step_end - step_start:.6f} s")

        step_start = time.perf_counter()
        with_features = self.add_transformed_columns(processed)
        step_end = time.perf_counter()
        print(f"add_transformed_columns: {step_end - step_start:.6f} s")

        step_start = time.perf_counter()
        deduplicated = self.deduplicate_results(with_features)
        step_end = time.perf_counter()
        print(f"deduplicate_results: {step_end - step_start:.6f} s")

        pipeline_end = time.perf_counter()
        print(f"TOTAL preprocessing: {pipeline_end - pipeline_start:.6f} s")

        logger.info(f"Готово: {len(deduplicated)} строк после предобработки")
        return deduplicated

    def validate_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Проверка структуры и базовых типов входного DataFrame.
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("Пустой DataFrame на входе препроцессора")

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"В данных отсутствуют обязательные колонки: {missing_columns}"
            )

        validated = dataframe.copy()

        if "city" not in validated.columns:
            validated["city"] = pd.NA
        if "bib_number" not in validated.columns:
            validated["bib_number"] = pd.NA

        try:
            validated["year"] = validated["year"].astype(int)
            validated["age"] = validated["age"].astype(int)
            validated["time_seconds"] = validated["time_seconds"].astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Не удалось привести year/age/time_seconds к числовым типам"
            ) from error

        validated["runner_id"] = validated["runner_id"].astype(str)

        logger.info(f"После validate: {len(validated)} строк")
        return validated

    def filter_ok_status(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Оставить только записи со статусом 'OK'.
        """
        total_count = len(dataframe)
        filtered = dataframe[dataframe["status"] == "OK"].copy()
        logger.info(
            f"Фильтрация по статусу OK: {len(filtered)} из {total_count} строк"
        )
        return filtered

    def disambiguate_runners_and_fill_city(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Разукрупнить однофамильцев и заполнить города.
        """
        working = dataframe.copy()

        runner_series = working["runner_id"].astype("string").str.strip()

        splitted = runner_series.str.split("_", n=1, expand=True)
        working["surname"] = splitted[0].str.strip()
        working["name"] = splitted[1].str.strip()

        mask_blank = (
            working["surname"].isna() | (working["surname"] == "") |
            working["name"].isna() | (working["name"] == "")
        )
        working.loc[mask_blank, ["surname", "name"]] = pd.NA

        working["approx_birth_year"] = (
            working["year"].astype(int) - working["age"].astype(int)
        )

        mask_valid = working["surname"].notna() & working["name"].notna()
        valid = working[mask_valid].copy()
        invalid = working[~mask_valid].copy()

        if not invalid.empty:
            logger.warning(
                f"{len(invalid)} строк с некорректным runner_id "
                "не участвуют в разукрупнении, но будут возвращены как есть"
            )

        processed_groups: list[pd.DataFrame] = []

        if not valid.empty:
            grouped = valid.groupby(["surname", "name", "gender"], sort=False)
            total_groups = len(grouped)
            logger.debug(
                "disambiguate_runners_and_fill_city: %d групп (surname, name, gender)",
                total_groups,
            )

            for index, ((surname, name, gender), group) in enumerate(grouped, start=1):
                clusters = self._split_into_person_clusters(group)
                base_runner_id = f"{surname}_{name}"

                clusters = self._assign_runner_ids(clusters, base_runner_id)

                # Мы смотрим на весь кластер "одного человека": все годы, все трассы, все забеги этого runner_id, после разукрупнения по году рождения.
                # После применения CLUSTER_CITY_CANONICAL_MAP внутри кластера остаётся несколько разных канонических локаций: Москва, Санкт-Петербург и т.п.
                # география этого кластера неоднородная, заполнение пропусков остановлено.
                # Мы не можем автоматом заполнить пропуски city одной модой в записях ниже
                clusters = self._fill_city_in_clusters(clusters, base_runner_id)
                processed_groups.extend(clusters)

                if index % 1000 == 0:
                    logger.debug(
                        "disambiguate_runners_and_fill_city: обработано %d/%d групп",
                        index,
                        total_groups,
                    )

        if processed_groups:
            processed_valid = pd.concat(processed_groups, ignore_index=True)
        else:
            processed_valid = valid

        combined = pd.concat([processed_valid, invalid], ignore_index=True)

        for column in ["surname", "name", "approx_birth_year"]:
            if column in combined.columns:
                combined = combined.drop(columns=[column])

        original_columns = [
            column for column in dataframe.columns if column in combined.columns
        ]
        extra_columns = [
            column for column in combined.columns if column not in dataframe.columns
        ]
        combined = combined[original_columns + extra_columns]

        logger.info(
            f"После разукрупнения и заполнения городов: {len(combined)} строк"
        )
        return combined

    def add_transformed_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Добавить преобразованные признаки Y и x.
        """
        working = dataframe.copy()

        non_positive_mask = working["time_seconds"] <= 0
        if non_positive_mask.any():
            count_bad = int(non_positive_mask.sum())
            raise ValueError(
                f"Обнаружено {count_bad} строк с неположительным time_seconds "
                "при расчёте Y = ln(time_seconds)"
            )

        working["Y"] = np.log(working["time_seconds"])
        working["x"] = (working["age"] - self.age_center) / self.age_scale

        logger.debug(
            "Добавлены колонки Y (ln(time_seconds)) и x ((age − age_center) / age_scale)"
        )
        return working

    def deduplicate_results(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Удалить строгие дубли и залогировать подозрительные группы.
        """
        working = dataframe.copy()

        strict_keys = [
            "runner_id",
            "race_id",
            "year",
            "gender",
            "age",
            "city",
            "time_seconds",
            "status",
            "bib_number",
        ]
        group_keys = ["runner_id", "year", "race_id"]

        before_count = len(working)
        deduplicated = working.drop_duplicates(
            subset=strict_keys,
            keep="first",
        ).reset_index(drop=True)
        removed = before_count - len(deduplicated)

        if removed > 0:
            logger.warning(
                f"Удалено {removed} строгих дублей по {strict_keys} "
                f"(было {before_count}, осталось {len(deduplicated)})"
            )

        soft_dupes_mask = deduplicated.duplicated(subset=group_keys, keep=False)
        soft_dupes = deduplicated.loc[soft_dupes_mask]

        if not soft_dupes.empty:
            # TODO: когда буду делать модель индивидуального эффекта по человеку # Тогда runner_id важен, и
            #  подозрительные группы нужно исключить только из части, # где используется runner_id
            logger.info(
                f"Найдено {len(soft_dupes)} записей в подозрительных группах "
                f"по {group_keys}. Строки не удаляются, только логируются."
            )
            sample = (
                soft_dupes.sort_values(group_keys)
                .groupby(group_keys)
                .head(5)
            )
            display_columns = [
                "runner_id",
                "race_id",
                "year",
                "age",
                "city",
                "bib_number",
                "time_seconds",
            ]
            display_columns = [
                column for column in display_columns if column in sample.columns
            ]
            logger.warning(
                "Пример подозрительных групп  один runner_id={surname}_{name}, трасса и год, но разные "
                "age/city/bib/time. Вероятно, склеены несколько людей:\n"
                f"{sample[display_columns].to_string(index=False)}"
            )

        return deduplicated

    def _split_into_person_clusters(self, group: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Разбить группу (surname, name, gender) на кластеры по approx_birth_year.
        """
        clusters: list[pd.DataFrame] = []

        if group.empty:
            return clusters

        surname_value = str(group["surname"].iloc[0])
        name_value = str(group["name"].iloc[0])
        full_name = f"{surname_value}_{name_value}"

        # pop(0) это O(n). Берём стек.
        pending_groups: list[pd.DataFrame] = [group]

        while pending_groups:
            current_group = pending_groups.pop()

            birth_year_series = current_group["approx_birth_year"].dropna()
            if birth_year_series.empty:
                clusters.append(current_group)
                continue

            # Быстро, без sorted(unique)
            min_year = int(birth_year_series.min())
            max_year = int(birth_year_series.max())
            if max_year - min_year <= 1:
                clusters.append(current_group)
                continue

            # Быстрое вычисление моды: O(n), а не O(n^2)
            mode_birth_year = int(birth_year_series.value_counts().idxmax())

            first_mask = birth_year_series.between(mode_birth_year - 1, mode_birth_year + 1)
            # first_mask относится только к индексам birth_year_series, приводим к индексу current_group
            first_index = birth_year_series.index[first_mask]

            first_cluster = current_group.loc[first_index].copy()
            rest_cluster = current_group.drop(index=first_index).copy()

            clusters.append(first_cluster)

            if not rest_cluster.empty:
                pending_groups.append(rest_cluster)

        if len(clusters) > 1:
            logger.info(
                f"Группа {full_name} разделена на {len(clusters)} кластера "
                f"(разные люди с одинаковым ФИО)"
            )

        return clusters

    def _assign_runner_ids(
        self,
        clusters: list[pd.DataFrame],
        base_runner_id: str,
    ) -> list[pd.DataFrame]:
        """
        Присвоить каждому кластеру стабильный runner_id.
        """
        updated_clusters: list[pd.DataFrame] = []

        for index, cluster in enumerate(clusters):
            cluster_copy = cluster.copy()
            if index == 0:
                cluster_copy["runner_id"] = base_runner_id
            else:
                cluster_copy["runner_id"] = f"{base_runner_id}_{index}"
            updated_clusters.append(cluster_copy)

        return updated_clusters

    def _fill_city_in_clusters(
            self,
            clusters: list[pd.DataFrame],
            base_runner_id: str,
    ) -> list[pd.DataFrame]:
        """
        Заполнить city внутри каждого кластера.

        Правило:
        - если все ненулевые города после приведения к канону совпадают,
          заполняем пропуски этим каноном и приводим все значения к нему;
        - если после приведения к канонам остаётся более одного значения,
          считаем, что это разные города и ничего не меняем, только логируем.
        """
        updated_clusters: list[pd.DataFrame] = []

        for cluster in clusters:
            if "city" not in cluster.columns:
                updated_clusters.append(cluster)
                continue

            cluster_copy = cluster.copy()

            city_series = cluster_copy["city"]
            non_null_mask = city_series.notna()
            if not bool(non_null_mask.any()):
                updated_clusters.append(cluster_copy)
                continue

            non_null_cities = city_series.loc[non_null_mask].astype("string").str.strip()

            # Векторная канонизация без lambda
            canonical_series = non_null_cities.map(CLUSTER_CITY_CANONICAL_MAP).fillna(non_null_cities)

            unique_values = canonical_series.dropna().unique()
            if len(unique_values) == 1:
                target_city = str(unique_values[0])

                # Векторно: приводим все ненулевые к канону, затем всё к target_city, затем заполняем NaN
                full_city = city_series.astype("string").str.strip()
                full_canonical = full_city.map(CLUSTER_CITY_CANONICAL_MAP).fillna(full_city)

                cluster_copy["city"] = full_canonical.fillna(target_city)
                cluster_copy.loc[:, "city"] = target_city

                updated_clusters.append(cluster_copy)
                continue

            # Конфликт:
            # несколько канонических мест внутри кластера одного человека после приведения
            # городов к каноническим названиям получается более одного значения.
            num_missing = int(city_series.isna().sum())

            if "race_id" in cluster_copy.columns and "year" in cluster_copy.columns:
                non_null_rows = cluster_copy.loc[non_null_mask, ["city", "race_id", "year"]].copy()

                raw_city = non_null_rows["city"].astype("string").str.strip()
                canonical_city = raw_city.map(CLUSTER_CITY_CANONICAL_MAP).fillna(raw_city)
                non_null_rows["canonical_city"] = canonical_city

                # Контекст "race-year" векторно
                race_part = non_null_rows["race_id"].astype("string")
                year_part = pd.to_numeric(non_null_rows["year"], errors="coerce").astype("Int64")
                context_label = race_part + "-" + year_part.astype("string")
                # где year NaN, label будет "race-<NA>", уберём
                non_null_rows["context_label"] = context_label.where(year_part.notna(), pd.NA)

                # Собираем контексты по городу (без iterrows)
                context_rows = non_null_rows.loc[
                    non_null_rows["context_label"].notna(), ["canonical_city", "context_label"]]
                if context_rows.empty:
                    detailed_entries = [f"'{value}'" for value in sorted(non_null_rows["canonical_city"].unique())]
                else:
                    grouped_context = (
                        context_rows.groupby("canonical_city", sort=True)["context_label"]
                        .unique()
                        .to_dict()
                    )

                    detailed_entries: list[str] = []
                    for city_value in sorted(non_null_rows["canonical_city"].unique()):
                        contexts = grouped_context.get(city_value, None)
                        if contexts is None:
                            detailed_entries.append(f"'{city_value}'")
                            continue
                        context_str = ", ".join(sorted(str(item) for item in contexts))
                        detailed_entries.append(f"'{city_value}'({context_str})")

                detailed_text = ", ".join(detailed_entries)
                base_message = (
                    f"Города в кластере {base_runner_id} "
                    f"не сведены к одному значению: {detailed_text}"
                )
            else:
                raw_unique = sorted(str(value).strip() for value in non_null_cities.unique())
                base_message = (
                    f"Города в кластере {base_runner_id} "
                    f"не сведены к одному значению: {raw_unique}"
                )

            if num_missing > 0:
                logger.info(f"{base_message}; {num_missing} пропусков city не заполнен")
            else:
                logger.info(base_message)

            updated_clusters.append(cluster_copy)

        return updated_clusters
