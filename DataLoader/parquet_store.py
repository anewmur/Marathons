
from __future__ import annotations

from pathlib import Path
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStore:
    """
    Читает и пишет parquet-кэш.
    """

    def read_parquet(self, cache_path: Path) -> pd.DataFrame | None:
        """
        Input: путь к parquet-файлу.
        Returns: DataFrame или None.
        Does: читает parquet-кэш.
        """
        if not cache_path.exists():
            return None
        try:
            return pd.read_parquet(cache_path)
        except Exception as error:
            logger.warning("ParquetStore: не удалось прочитать кэш %s: %s", cache_path.name, error)
            return None

    def write_parquet_atomic(self, cache_path: Path, dataframe: pd.DataFrame) -> bool:
        """
        Input: путь и DataFrame.
        Returns: True если успешно.
        Does: атомарно записывает parquet (tmp + replace).
        """
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            dataframe.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, cache_path)
            return True
        except Exception as error:
            logger.warning("ParquetStore: не удалось записать кэш %s: %s", cache_path.name, error)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False
