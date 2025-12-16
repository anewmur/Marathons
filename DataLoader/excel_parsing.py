
from __future__ import annotations

from dataclasses import dataclass
import logging

from DataLoader.metadata_reader import ExcelMeta, ExcelMetadataReader
from DataLoader.race_title import RaceTitleParser
from DataLoader.results_table import ResultsTableParser, ResultsTableSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExcelParsingBundle:
    """
    связка объектов для парсинга Excel.

    Fields:
        title_parser: RaceTitleParser
        metadata_reader: ExcelMetadataReader
        city_normalizer: CityNormalizer
        table_schema: ResultsTableSchema
        table_parser: ResultsTableParser
    """

    title_parser: RaceTitleParser
    metadata_reader: ExcelMetadataReader
    table_schema: ResultsTableSchema
    table_parser: ResultsTableParser


__all__ = [
    "ExcelMeta",
    "ExcelMetadataReader",
    "RaceTitleParser",
    "ResultsTableSchema",
    "ResultsTableParser",
    "ExcelParsingBundle",
]
