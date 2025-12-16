from __future__ import annotations

import logging
import time
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from dictionaries import CLUSTER_CITY_CANONICAL_MAP, REGION_NORMALIZATION_MAP, PERSON_TOKEN_CYR_TO_LAT

logger = logging.getLogger(__name__)


class Preprocessor:
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
        """
        Разукрупнить однофамильцев и заполнить города.
        Ключ группы: (surname, name) без учёта gender.
        Шаги внутри (surname, name):
          1) gender -> к большинству; если большинства нет, выбрасываем всю группу;
          2) делим по approx_birth_year (year - age), если диапазон >= 2;
          3) страховка: внутри (race_id, year, surname, name) при нескольких bib_number
             выделяем отдельные под-кластеры по bib_number;
          4) город: заполняем только пропуски и только если в финальном кластере город единственный.
        Пишем JSON по финальным кластерам (опционально).
        """
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        working = dataframe.copy()
        runner_id_series = working["runner_id"].astype("string").str.strip()
        split_table = runner_id_series.str.split("_", n=1, expand=True)
        working["surname_raw"] = split_table[0]
        working["name_raw"] = split_table[1]
        mask_valid = working["surname_raw"].notna() & working["name_raw"].notna()
        valid = working.loc[mask_valid].copy()
        invalid = working.loc[~mask_valid].copy()
        t_parse = time.perf_counter() - t0

        if not invalid.empty:
            logger.warning(
                "%d строк с некорректным runner_id не участвуют в разукрупнении, но будут возвращены как есть",
                int(len(invalid)),
            )

        if valid.empty:
            combined = pd.concat([valid, invalid], ignore_index=True)
            return self._drop_service_columns(combined, dataframe)

        t0 = time.perf_counter()
        valid["name_base"] = valid["name_raw"].astype("string").str.replace(r"_\d+$", "", regex=True)
        valid["surname_key"] = self._person_token_to_key_vectorized(valid["surname_raw"])
        valid["name_key"] = self._person_token_to_key_vectorized(valid["name_base"])
        valid["approx_birth_year"] = valid["year"].astype(int) - valid["age"].astype(int)
        t_keys = time.perf_counter() - t0

        t0 = time.perf_counter()
        if "city" in valid.columns:
            city_raw = valid["city"].astype("string").str.strip()
            city_raw = city_raw.replace({"": pd.NA, "None": pd.NA})
            city_canon = city_raw.map(CLUSTER_CITY_CANONICAL_MAP).fillna(city_raw)
            city_norm = city_canon.map(REGION_NORMALIZATION_MAP).fillna(city_canon)
            valid["city_norm"] = city_norm
        else:
            valid["city_norm"] = pd.NA
        t_city_canon = time.perf_counter() - t0

        t0 = time.perf_counter()
        group_columns = ["surname_key", "name_key"]
        group_sizes = valid.groupby(group_columns, sort=False).size()
        logger.debug(
            "disambiguate_runners_and_fill_city: %d групп (surname, name), из них неоднозначных %d",
            int(group_sizes.shape[0]),
            int((group_sizes > 1).sum()),
        )
        group_to_index = valid.groupby(group_columns, sort=False).groups
        t_groupby = time.perf_counter() - t0

        dropped_indexes: list[int] = []
        cluster_records: list[dict[str, object]] = []

        # Агрегаты времени
        split_time = 0.0
        assign_time = 0.0
        fill_time = 0.0
        slow: list[tuple[float, str, str, int]] = []

        t_loop_start = time.perf_counter()
        for (surname_key, name_key), group_index in group_to_index.items():
            if len(group_index) == 1:
                continue

            t_group_start = time.perf_counter()

            group_frame = valid.loc[group_index]

            group_majority_gender = self._majority_gender(group_frame["gender"])
            if group_majority_gender is None:
                dropped_indexes.extend(list(group_index))
                continue

            valid.loc[group_index, "gender"] = group_majority_gender

            surname_display = self._select_display_token(group_frame["surname_raw"])
            name_display = self._select_display_token(group_frame["name_base"])
            base_runner_id = f"{surname_display}_{name_display}"

            t1 = time.perf_counter()
            birth_clusters = self._split_by_birth_year(group_frame, group_index)
            split_time += time.perf_counter() - t1

            final_clusters: list[dict[str, object]] = []
            for birth_cluster_index in birth_clusters:
                bib_clusters = self._split_by_multi_bib_events(valid, birth_cluster_index)
                for bib_cluster_index in bib_clusters:
                    cluster_stat = self._cluster_stats(valid, bib_cluster_index)
                    cluster_stat["index"] = bib_cluster_index
                    cluster_stat["base_runner_id"] = base_runner_id
                    final_clusters.append(cluster_stat)

            if not final_clusters:
                continue

            final_clusters_sorted = sorted(final_clusters, key=self._cluster_sort_key)

            suffix_counter = 0
            t_assign_start = time.perf_counter()
            for cluster_item in final_clusters_sorted:
                suffix_counter += 1
                assigned_runner_id = base_runner_id if suffix_counter == 1 else f"{base_runner_id}_{suffix_counter}"
                cluster_index = cluster_item["index"]
                valid.loc[cluster_index, "runner_id"] = assigned_runner_id

                t_fill_start = time.perf_counter()
                self._fill_city_within_cluster(valid, cluster_index)
                fill_time += time.perf_counter() - t_fill_start

                record = self._cluster_record(
                    surname_key=surname_key,
                    name_key=name_key,
                    base_runner_id=base_runner_id,
                    assigned_runner_id=assigned_runner_id,
                    group_majority_gender=group_majority_gender,
                    cluster_item=cluster_item,
                )
                cluster_records.append(record)
            assign_time += time.perf_counter() - t_assign_start

            t_group_elapsed = time.perf_counter() - t_group_start
            if self.debug_timing:
                if len(slow) < self.debug_timing_top:
                    slow.append((t_group_elapsed, str(surname_key), str(name_key), int(len(group_index))))
                else:
                    # Замена худшего
                    min_idx = min(range(len(slow)), key=lambda i: slow[i][0])
                    if t_group_elapsed > slow[min_idx][0]:
                        slow[min_idx] = (t_group_elapsed, str(surname_key), str(name_key), int(len(group_index)))

        t_loop = time.perf_counter() - t_loop_start

        if dropped_indexes:
            valid = valid.drop(index=pd.Index(dropped_indexes))

        self._maybe_write_cluster_json(cluster_records)

        t0 = time.perf_counter()
        combined = pd.concat([valid, invalid], ignore_index=True)
        combined = self._drop_service_columns(combined, dataframe)
        t_finalize = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if self.debug_timing:
            logger.warning(
                "TIMING disambiguate: total=%.3fs parse=%.3fs keys=%.3fs city_canon=%.3fs "
                "groupby=%.3fs loop=%.3fs finalize=%.3fs",
                t_total, t_parse, t_keys, t_city_canon,
                t_groupby, t_loop, t_finalize,
            )
            logger.warning(
                "TIMING details: split_birth=%.3fs assign_ids=%.3fs fill_city=%.3fs",
                split_time, assign_time, fill_time,
            )
            slow_sorted = sorted(slow, key=lambda item: item[0], reverse=True)
            for elapsed, surname_key_value, name_key_value, group_len in slow_sorted:
                logger.warning(
                    "SLOW group: %.4fs size=%d key=%s_%s",
                    elapsed, int(group_len), surname_key_value, name_key_value,
                )

        logger.info("После разукрупнения и заполнения городов: %d строк", int(len(combined)))
        return combined

    def _person_token_to_key_vectorized(self, series: pd.Series) -> pd.Series:
        """
        Векторизованная версия _person_token_to_key для ускорения.
        """
        s = series.astype(str).str.strip().str.lower().str.replace('ё', 'е')
        is_cyr = s.str.contains(r'[а-яё]', regex=True)



        s_cyr = s[is_cyr].str.translate(PERSON_TOKEN_CYR_TO_LAT)
        s_cyr = s_cyr.str.replace('[^a-z]', '', regex=True).str.replace('x', 'ks').str.replace('ye', 'e').str.replace(
            'yo', 'e')

        s_non_cyr = s[~is_cyr]
        s_non_cyr = s_non_cyr.str.replace('[^a-z]', '', regex=True).str.replace('x', 'ks').str.replace('ye',
                                                                                                       'e').str.replace(
            'yo', 'e')

        result = pd.Series(index=series.index, dtype=str)
        result[is_cyr] = s_cyr
        result[~is_cyr] = s_non_cyr
        result = result.fillna('')
        return result

    def _drop_service_columns(self, combined: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        service_columns = [
            "surname_raw",
            "name_raw",
            "name_base",
            "surname_key",
            "name_key",
            "approx_birth_year",
            "city_norm",
        ]
        combined = combined.drop(columns=service_columns, errors='ignore')
        original_columns = [column for column in original.columns if column in combined.columns]
        extra_columns = [column for column in combined.columns if column not in original.columns]
        combined = combined[original_columns + extra_columns]
        return combined

    def _select_display_token(self, series: pd.Series) -> str:
        cleaned = series.astype("string").str.strip()
        cleaned = cleaned.replace({"": pd.NA, "None": pd.NA}).dropna()
        if cleaned.empty:
            return "Unknown"
        counts = cleaned.value_counts()
        return str(counts.index[0])

    def _majority_gender(self, gender_series: pd.Series) -> str | None:
        cleaned = gender_series.astype("string").str.strip()
        cleaned = cleaned.replace({"": pd.NA, "None": pd.NA}).dropna()
        if cleaned.empty:
            return None
        counts = cleaned.value_counts()
        if counts.shape[0] == 1:
            return str(counts.index[0])
        top_count = int(counts.iloc[0])
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        if top_count == second_count:
            return None
        return str(counts.index[0])

    def _split_by_birth_year(self, group_frame: pd.DataFrame, group_index: pd.Index) -> list[pd.Index]:
        approx_birth_year = group_frame["approx_birth_year"].dropna()
        if approx_birth_year.empty:
            return [group_index]
        min_year = int(approx_birth_year.min())
        max_year = int(approx_birth_year.max())
        if (max_year - min_year) < 2:
            return [group_index]
        indexes_left: list[pd.Index] = [group_index]
        clusters: list[pd.Index] = []
        while indexes_left:
            current_index = indexes_left.pop()
            current_years = group_frame.loc[current_index, "approx_birth_year"].dropna()
            if current_years.empty:
                clusters.append(current_index)
                continue
            min_current = int(current_years.min())
            max_current = int(current_years.max())
            if (max_current - min_current) < 2:
                clusters.append(current_index)
                continue
            mode_year = int(current_years.value_counts().index[0])
            close_mask = (current_years - mode_year).abs() <= 1
            close_index = current_years.index[close_mask]
            far_index = current_years.index[~close_mask]
            clusters.append(pd.Index(close_index))
            if len(far_index) > 0:
                indexes_left.append(pd.Index(far_index))
        return clusters

    def _split_by_multi_bib_events(self, dataframe: pd.DataFrame, cluster_index: pd.Index) -> list[pd.Index]:
        cluster_frame = dataframe.loc[cluster_index]
        group_columns = ["race_id", "year"]
        event_bib_nunique = cluster_frame.groupby(group_columns, sort=False)["bib_number"].nunique(dropna=True)
        multi_bib_events = event_bib_nunique[event_bib_nunique > 1]
        if multi_bib_events.empty:
            return [cluster_index]
        remaining_indexes = set(cluster_index.tolist())
        output_clusters: list[pd.Index] = []
        for (race_id, year), _ in multi_bib_events.items():
            event_mask = (cluster_frame["race_id"] == race_id) & (cluster_frame["year"] == year)
            event_index = cluster_frame.index[event_mask]
            event_bib = dataframe.loc[event_index, "bib_number"]
            event_bib_non_na = event_bib.dropna().unique().tolist()
            for bib_value in sorted(event_bib_non_na):
                bib_mask = event_bib == bib_value
                bib_index = event_index[bib_mask]
                output_clusters.append(pd.Index(bib_index))
                for row_index in bib_index.tolist():
                    if row_index in remaining_indexes:
                        remaining_indexes.remove(row_index)
            missing_mask = event_bib.isna()
            missing_index = event_index[missing_mask]
            if len(missing_index) > 0:
                output_clusters.append(pd.Index(missing_index))
                for row_index in missing_index.tolist():
                    if row_index in remaining_indexes:
                        remaining_indexes.remove(row_index)
        if remaining_indexes:
            output_clusters.append(pd.Index(sorted(remaining_indexes)))
        return output_clusters

    def _cluster_stats(self, dataframe: pd.DataFrame, cluster_index: pd.Index) -> dict[str, object]:
        cluster_frame = dataframe.loc[cluster_index]
        approx_birth_year = cluster_frame["approx_birth_year"].dropna()
        record: dict[str, object] = {}
        record["rows"] = int(len(cluster_frame))
        record["birth_year_min"] = int(approx_birth_year.min()) if not approx_birth_year.empty else None
        record["birth_year_max"] = int(approx_birth_year.max()) if not approx_birth_year.empty else None
        record["birth_year_median"] = float(approx_birth_year.median()) if not approx_birth_year.empty else None
        record["year_min"] = int(cluster_frame["year"].min()) if not cluster_frame["year"].empty else None
        record["year_max"] = int(cluster_frame["year"].max()) if not cluster_frame["year"].empty else None
        record["bib_nunique"] = int(
            cluster_frame["bib_number"].nunique(dropna=True)) if "bib_number" in cluster_frame.columns else 0
        record["event_count"] = int(cluster_frame.groupby(["race_id", "year"], sort=False).ngroups)
        record["city_nunique"] = int(
            cluster_frame["city_norm"].nunique(dropna=True)) if "city_norm" in cluster_frame.columns else 0
        return record

    def _cluster_sort_key(self, cluster_item: dict[str, object]) -> tuple[float, int, int, int]:
        birth_median = cluster_item.get("birth_year_median")
        birth_value = float(birth_median) if birth_median is not None else 10000.0
        year_min_value = int(cluster_item.get("year_min", 10000))
        bib_nunique_value = int(cluster_item.get("bib_nunique", 0))
        rows_value = int(cluster_item.get("rows", 0))
        return (birth_value, year_min_value, bib_nunique_value, -rows_value)

    def _fill_city_within_cluster(self, dataframe: pd.DataFrame, cluster_index: pd.Index) -> None:
        if "city" not in dataframe.columns or "city_norm" not in dataframe.columns:
            return
        city_norm = dataframe.loc[cluster_index, "city_norm"]
        city_non_na = city_norm.dropna().unique().tolist()
        if len(city_non_na) != 1:
            return
        target_city = city_non_na[0]
        city_raw = dataframe.loc[cluster_index, "city"].astype("string").str.strip()
        missing_mask = city_raw.isna() | (city_raw == "") | (city_raw == "None")
        missing_index = cluster_index[missing_mask]
        if len(missing_index) == 0:
            return
        dataframe.loc[missing_index, "city"] = target_city

    def _cluster_record(
            self,
            surname_key: str,
            name_key: str,
            base_runner_id: str,
            assigned_runner_id: str,
            group_majority_gender: str,
            cluster_item: dict[str, object],
    ) -> dict[str, object]:
        record: dict[str, object] = {}
        record["surname_key"] = surname_key
        record["name_key"] = name_key
        record["base_runner_id"] = base_runner_id
        record["runner_id"] = assigned_runner_id
        record["group_majority_gender"] = group_majority_gender
        for key_name in [
            "rows", "birth_year_min", "birth_year_max", "birth_year_median",
            "year_min", "year_max", "city_nunique", "event_count", "bib_nunique",
        ]:
            record[key_name] = cluster_item.get(key_name)
        return record

    def _maybe_write_cluster_json(self, cluster_records: list[dict[str, object]]) -> None:
        preprocessing_config = self.config.get("preprocessing", {})
        enabled = bool(preprocessing_config.get("write_person_clusters_json", False))
        if not enabled:
            return
        output_path = preprocessing_config.get("person_clusters_json_path", "person_clusters.json")
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {}
            payload["clusters"] = cluster_records
            payload["clusters_count"] = int(len(cluster_records))
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("JSON отчёт по кластерам записан: %s (кластеров=%d)", str(path), int(len(cluster_records)))
        except Exception:
            logger.exception("Не удалось записать JSON отчёт по кластерам")

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


