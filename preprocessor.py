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
from dataclasses import dataclass
from utils import split_runner_id
from dictionaries import CLUSTER_CITY_CANONICAL_MAP, REGION_NORMALIZATION_MAP
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
            self.age_center = float(preprocessing_config.get("age_center", 0))
            self.age_scale = float(preprocessing_config.get("age_scale", 0))
            self.debug_timing = bool(preprocessing_config.get("debug_timing", False))
            self.debug_timing_top = int(preprocessing_config.get("debug_timing_top", 15))

        except KeyError as error:
            raise KeyError(
                "В config['preprocessing'] должны быть заданы "
                "'age_center' и 'age_scale'"
            ) from error

        if self.age_scale == 0:
            raise ValueError("preprocessing.age_scale не должен быть равен нулю")

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

        processed = self.disambiguate_same_event_by_bib(processed)

        if bool(self.config.get("preprocessing", {}).get("suspicious_report", False)):
            report = self.analyze_suspicious(processed)
            logger.warning("SUSPICIOUS %s", report["counts"])
            logger.warning("SUSPICIOUS examples_event_rows:\n%s", report["examples_event_rows"].to_string(index=False))

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

    def disambiguate_runners_and_fill_city(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working = dataframe.copy()

        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        runner_id_series = working["runner_id"].astype("string").str.strip()
        split_table = runner_id_series.str.split("_", n=1, expand=True)
        working["surname"] = split_table[0]
        working["name"] = split_table[1]
        working["approx_birth_year"] = working["year"].astype(int) - working["age"].astype(int)
        t_parse = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask_valid = working["surname"].notna() & working["name"].notna()
        valid = working.loc[mask_valid]
        invalid = working.loc[~mask_valid]
        t_split_valid = time.perf_counter() - t0

        if not invalid.empty:
            logger.warning(
                "%d строк с некорректным runner_id не участвуют в разукрупнении, но будут возвращены как есть",
                int(len(invalid)),
            )

        if valid.empty:
            combined = pd.concat([valid, invalid], ignore_index=True)
            for column in ["surname", "name", "approx_birth_year", "city_canon"]:
                if column in combined.columns:
                    combined = combined.drop(columns=[column])
            return combined

        t0 = time.perf_counter()
        if "city" in valid.columns:
            city_raw = valid["city"].astype("string").str.strip()
            valid = valid.copy()
            valid["city_canon"] = city_raw.map(CLUSTER_CITY_CANONICAL_MAP).fillna(city_raw)
        else:
            valid = valid.copy()
            valid["city_canon"] = pd.NA
        t_city_canon = time.perf_counter() - t0

        report_flag = bool(self.config.get("preprocessing", {}).get("suspicious_report", False))
        if report_flag:
            self.build_suspicious_report(valid)

        group_cols = ["surname", "name", "gender"]

        t0 = time.perf_counter()
        group_sizes = valid.groupby(group_cols, sort=False).size()
        ambiguous_keys = group_sizes[group_sizes > 1].index
        t_group_sizes = time.perf_counter() - t0

        if len(ambiguous_keys) == 0:
            processed_valid = valid
            t_merges = 0.0
            t_groupby = 0.0
            t_loop = 0.0
        else:
            t0 = time.perf_counter()
            ambiguous_keys_df = pd.DataFrame(list(ambiguous_keys), columns=group_cols)

            t1 = time.perf_counter()
            ambiguous = valid.merge(ambiguous_keys_df, on=group_cols, how="inner")
            t_merge_amb = time.perf_counter() - t1

            t1 = time.perf_counter()
            singletons = valid.merge(ambiguous_keys_df, on=group_cols, how="left", indicator=True)
            singletons = singletons.loc[singletons["_merge"] == "left_only"].drop(columns=["_merge"])
            t_merge_sing = time.perf_counter() - t1
            t_merges = time.perf_counter() - t0

            t0 = time.perf_counter()
            grouped = ambiguous.groupby(group_cols, sort=False)
            total_groups = int(len(ambiguous_keys))
            t_groupby = time.perf_counter() - t0

            logger.debug(
                "disambiguate_runners_and_fill_city: %d групп (surname, name, gender), из них неоднозначных %d",
                int(group_sizes.shape[0]),
                int(len(ambiguous_keys)),
            )

            processed_groups: list[pd.DataFrame] = []
            processed_count = 0

            # агрегаты времени по внутренним функциям
            split_time = 0.0
            assign_time = 0.0
            fill_time = 0.0
            concat_clusters_rows = 0
            concat_clusters_count = 0

            # топ медленных групп
            slow: list[tuple[float, str, str, str, int]] = []

            t_loop_start = time.perf_counter()
            for (surname, name, gender), group in grouped:
                t_group_start = time.perf_counter()

                t1 = time.perf_counter()
                clusters = self._split_into_person_clusters(group)
                split_time += time.perf_counter() - t1

                base_runner_id = f"{surname}_{name}"

                t1 = time.perf_counter()
                clusters = self._assign_runner_ids(clusters, base_runner_id)
                assign_time += time.perf_counter() - t1

                t1 = time.perf_counter()
                clusters = self._fill_city_in_clusters(clusters, base_runner_id)
                fill_time += time.perf_counter() - t1

                for cluster_frame in clusters:
                    concat_clusters_rows += int(len(cluster_frame))
                concat_clusters_count += int(len(clusters))

                processed_groups.extend(clusters)

                t_group_elapsed = time.perf_counter() - t_group_start
                if self.debug_timing:
                    if len(slow) < self.debug_timing_top:
                        slow.append((t_group_elapsed, str(surname), str(name), str(gender), int(len(group))))
                    else:
                        worst_index = 0
                        worst_value = slow[0][0]
                        for idx, item in enumerate(slow):
                            if item[0] > worst_value:
                                worst_value = item[0]
                                worst_index = idx
                        if t_group_elapsed > worst_value:
                            slow[worst_index] = (t_group_elapsed, str(surname), str(name), str(gender), int(len(group)))

                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.debug(
                        "disambiguate_runners_and_fill_city: обработано %d/%d неоднозначных групп",
                        processed_count,
                        total_groups,
                    )

            t_loop = time.perf_counter() - t_loop_start

            t0 = time.perf_counter()
            if processed_groups:
                processed_ambiguous = pd.concat(processed_groups, ignore_index=True)
            else:
                processed_ambiguous = ambiguous
            t_concat_processed = time.perf_counter() - t0

            t0 = time.perf_counter()
            processed_valid = pd.concat([singletons, processed_ambiguous], ignore_index=True)
            t_concat_valid = time.perf_counter() - t0

        t0 = time.perf_counter()
        combined = pd.concat([processed_valid, invalid], ignore_index=True)
        for column in ["surname", "name", "approx_birth_year", "city_canon"]:
            if column in combined.columns:
                combined = combined.drop(columns=[column])
        original_columns = [column for column in dataframe.columns if column in combined.columns]
        extra_columns = [column for column in combined.columns if column not in dataframe.columns]
        combined = combined[original_columns + extra_columns]
        t_finalize = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if self.debug_timing:
            logger.warning(
                "TIMING disambiguate: total=%.3fs parse=%.3fs split_valid=%.3fs city_canon=%.3fs "
                "group_sizes=%.3fs merges=%.3fs groupby=%.3fs loop=%.3fs finalize=%.3fs",
                t_total, t_parse, t_split_valid, t_city_canon,
                t_group_sizes, t_merges, t_groupby, t_loop, t_finalize,
            )
            if len(ambiguous_keys) > 0:
                logger.warning(
                    "TIMING details: merge_amb=%.3fs merge_singletons=%.3fs "
                    "concat_processed=%.3fs concat_valid=%.3fs "
                    "split_clusters=%.3fs assign_ids=%.3fs fill_city=%.3fs "
                    "clusters_count=%d clusters_rows=%d",
                    t_merge_amb, t_merge_sing,
                    t_concat_processed, t_concat_valid,
                    split_time, assign_time, fill_time,
                    int(concat_clusters_count), int(concat_clusters_rows),
                )
                slow_sorted = sorted(slow, key=lambda item: item[0], reverse=True)
                for elapsed, surname_value, name_value, gender_value, group_len in slow_sorted:
                    logger.warning(
                        "SLOW group: %.4fs size=%d key=%s_%s gender=%s",
                        elapsed, int(group_len), surname_value, name_value, gender_value,
                    )

        logger.info("После разукрупнения и заполнения городов: %d строк", int(len(combined)))
        return combined

    def build_suspicious_report(self, dataframe: pd.DataFrame) -> dict[str, object]:
        working = dataframe.copy()

        runner_id_series = working["runner_id"].astype("string").str.strip()
        split_table = runner_id_series.str.split("_", n=1, expand=True)
        working["surname"] = split_table[0]
        working["name"] = split_table[1]
        working["approx_birth_year"] = working["year"].astype(int) - working["age"].astype(int)

        mask_valid = working["surname"].notna() & working["name"].notna()
        valid = working.loc[mask_valid].copy()

        city_raw = valid["city"].astype("string").str.strip() if "city" in valid.columns else pd.Series(pd.NA,
                                                                                                        index=valid.index)
        valid["city_canon"] = city_raw.map(CLUSTER_CITY_CANONICAL_MAP).fillna(city_raw)

        group_cols = ["surname", "name", "gender"]

        # Тип 2: разброс года рождения
        birth_stats = valid.groupby(group_cols, sort=False)["approx_birth_year"].agg(
            group_size="size",
            birth_year_min="min",
            birth_year_max="max",
            birth_year_nunique="nunique",
        )
        birth_stats["birth_year_range"] = birth_stats["birth_year_max"] - birth_stats["birth_year_min"]
        suspicious_birth = birth_stats[birth_stats["birth_year_range"] >= 3].sort_values(
            ["birth_year_range", "group_size"], ascending=[False, False]
        )

        # Тип 3: конфликт города (слабый)
        city_stats = valid.groupby(group_cols, sort=False)["city_canon"].nunique(dropna=True).to_frame("city_nunique")
        suspicious_city = city_stats[city_stats["city_nunique"] >= 2].sort_values("city_nunique", ascending=False)

        # Тип 1: дубль в одном забеге (жёстко)
        event_cols = group_cols + ["race_id", "year"]
        event_sizes = valid.groupby(event_cols, sort=False).size().to_frame("event_count")
        event_multi = event_sizes[event_sizes["event_count"] >= 2].reset_index()

        if event_multi.empty:
            suspicious_event = valid.iloc[0:0].copy()
        else:
            event_multi = event_multi[event_cols]
            candidates = valid.merge(event_multi, on=event_cols, how="inner")

            # “не одинаковые записи” внутри одного (ФИО, пол, трасса, год)
            compare_cols = ["age", "time_seconds", "bib_number", "city_canon"]
            nunique_inside = candidates.groupby(event_cols, sort=False)[compare_cols].nunique(dropna=False)
            conflict_mask = (
                    (nunique_inside["age"] >= 2)
                    | (nunique_inside["time_seconds"] >= 2)
                    | (nunique_inside["bib_number"] >= 2)
                    | (nunique_inside["city_canon"] >= 2)
            )
            conflict_keys = nunique_inside.loc[conflict_mask].reset_index()[event_cols]
            suspicious_event = candidates.merge(conflict_keys, on=event_cols, how="inner")

        # Итоги
        report = {}
        report["counts"] = {
            "valid_rows": int(len(valid)),
            "groups_total": int(valid.groupby(group_cols, sort=False).ngroups),
            "suspicious_birth_groups": int(len(suspicious_birth)),
            "suspicious_city_groups": int(len(suspicious_city)),
            "suspicious_event_groups": int(suspicious_event.groupby(event_cols, sort=False).ngroups),
        }

        # Примеры для ручной проверки
        report["examples_birth_groups"] = suspicious_birth.head(20).reset_index()
        report["examples_city_groups"] = suspicious_city.head(20).reset_index()

        report["examples_event_rows"] = (
            suspicious_event.sort_values(event_cols)
            .loc[:,
            ["runner_id", "surname", "name", "gender", "race_id", "year", "age", "city", "bib_number", "time_seconds",
             "approx_birth_year"]]
            .head(50)
            .reset_index(drop=True)
        )

        return report

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
            # logger.warning(
            #     "Пример подозрительных групп  один runner_id={surname}_{name}, трасса и год, но разные "
            #     "age/city/bib/time. Вероятно, склеены несколько людей:\n"
            #     f"{sample[display_columns].to_string(index=False)}"
            # )

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
        updated_clusters: list[pd.DataFrame] = []

        for cluster in clusters:
            if "city_canon" not in cluster.columns:
                updated_clusters.append(cluster)
                continue

            cluster_copy = cluster.copy()

            canon = cluster_copy["city_canon"]
            non_null = canon.dropna()

            if non_null.empty:
                updated_clusters.append(cluster_copy)
                continue

            unique_values = non_null.unique()
            if len(unique_values) == 1:
                target_city = str(unique_values[0])

                cluster_copy["city_canon"] = canon.fillna(target_city)
                cluster_copy["city"] = target_city

                updated_clusters.append(cluster_copy)
                continue

            num_missing = int(cluster_copy["city"].isna().sum()) if "city" in cluster_copy.columns else 0

            # конфликт логируем, но ничего не заполняем
            base_message = (
                f"Города в кластере {base_runner_id} "
                f"не сведены к одному значению: {sorted(str(value) for value in unique_values)}"
            )
            if num_missing > 0:
                logger.info(f"{base_message}; {num_missing} пропусков city не заполнен")
            else:
                logger.info(base_message)

            updated_clusters.append(cluster_copy)

        return updated_clusters

    def analyze_suspicious(self, df_clean):
        dataframe = df_clean.copy()

        runner_id_series = dataframe["runner_id"].astype("string").str.strip()
        split_table = runner_id_series.str.split("_", n=1, expand=True)
        dataframe["surname"] = split_table[0]
        dataframe["name"] = split_table[1]

        city_series = dataframe["city"].astype("string").str.strip()
        city_series = city_series.replace({"": pd.NA, "None": pd.NA})

        city_canon = city_series.map(CLUSTER_CITY_CANONICAL_MAP).fillna(city_series)
        city_norm = city_canon.map(REGION_NORMALIZATION_MAP).fillna(city_canon)
        dataframe["city_norm"] = city_norm

        group_cols_city = ["surname", "name", "gender"]
        city_nunique = dataframe.groupby(group_cols_city, sort=False)["city_norm"].nunique(dropna=True)
        suspicious_city_keys = city_nunique[city_nunique > 1].index

        city_key_index = pd.MultiIndex.from_tuples(suspicious_city_keys, names=group_cols_city)
        city_row_index = pd.MultiIndex.from_frame(dataframe[group_cols_city])
        mask_suspicious_city = city_row_index.isin(city_key_index)

        group_cols_event = ["runner_id", "race_id", "year"]
        event_sizes = dataframe.groupby(group_cols_event, sort=False).size()
        suspicious_event_keys = event_sizes[event_sizes > 1].index

        event_key_index = pd.MultiIndex.from_tuples(suspicious_event_keys, names=group_cols_event)
        event_row_index = pd.MultiIndex.from_frame(dataframe[group_cols_event])
        mask_suspicious_event = event_row_index.isin(event_key_index)

        rows_total = int(len(dataframe))
        rows_city = int(mask_suspicious_city.sum())
        rows_event = int(mask_suspicious_event.sum())
        rows_union = int((mask_suspicious_city | mask_suspicious_event).sum())
        rows_intersection = int((mask_suspicious_city & mask_suspicious_event).sum())

        groups_city_total = int(city_nunique.shape[0])
        groups_city_suspicious = int(len(suspicious_city_keys))
        groups_event_total = int(event_sizes.shape[0])
        groups_event_suspicious = int(len(suspicious_event_keys))
        report: dict[str, object] = {}
        report["counts"] = {
            "TOTAL rows": rows_total,
            "Подозрительные city groups": f"groups={groups_city_suspicious}/{groups_city_total}, rows={rows_city} ({rows_city / rows_total:.2%})",
            "Подозрительные event groups": f"groups={groups_event_suspicious}/{groups_event_total}, rows={rows_event} ({rows_event / rows_total:.2%})",
            "Rows in union(city ∪ event)": f"{rows_union} ({rows_union / rows_total:.2%})",
            "Rows in intersection(city ∩ event)": f"{rows_intersection} ({rows_intersection / rows_total:.2%})"
        }

        examples_event = dataframe.loc[mask_suspicious_event, [
            "surname", "name", "gender", "race_id", "year", "bib_number", "age", "city", "time_seconds"
        ]].sort_values(["race_id", "year", "surname", "name", "gender", "bib_number"]).head(30)

        report["examples_event_rows"] = examples_event
        return report


    def disambiguate_same_event_by_bib(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        dataframe = df_clean.copy()

        runner_id_series = dataframe["runner_id"].astype("string").str.strip()
        split_table = runner_id_series.str.split("_", n=1, expand=True)
        dataframe["surname"] = split_table[0]
        dataframe["name"] = split_table[1]

        group_cols_event = ["surname", "name", "gender", "race_id", "year"]
        bib_nunique_table = (
            dataframe.groupby(group_cols_event, sort=False)["bib_number"]
            .nunique(dropna=True)
            .reset_index(name="bib_nunique")
        )
        dataframe = dataframe.merge(bib_nunique_table, on=group_cols_event, how="left")

        mask = dataframe["bib_nunique"] > 1
        if mask.any():
            bib_series = dataframe["bib_number"].astype("Int64").astype("string")
            bib_series = bib_series.fillna("NA")
            base_runner_id = dataframe["surname"].astype("string") + "_" + dataframe["name"].astype("string")
            dataframe.loc[mask, "runner_id"] = base_runner_id.loc[mask] + "_B" + bib_series.loc[mask]

        dataframe = dataframe.drop(columns=["surname", "name", "bib_nunique"])
        return dataframe