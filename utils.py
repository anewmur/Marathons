"""
Вспомогательные функции для обработки данных марафонов.

Содержит функции парсинга времени, пола и статуса участников.
"""

import pandas as pd


def parse_time_to_seconds(time_str: str) -> float | None:
    """
    Преобразовать строку времени в секунды.
    
    Поддерживаемые форматы:
    - "HH:MM:SS" (например, "02:33:51")
    - "H:MM:SS" (например, "2:33:51")
    - "MM:SS" (например, "33:51")
    
    Args:
        time_str: Строка с временем
        
    Returns:
        Время в секундах или None при ошибке парсинга
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    
    time_str = time_str.strip()
    if not time_str:
        return None
    
    parts = time_str.split(':')
    
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return None
    except (ValueError, TypeError):
        return None


def parse_gender_from_category(category: str) -> str | None:
    """
    Определить пол по полю 'Категория'.
    
    Правила:
    - 'М' в начале или 'МУЖЧИН' → 'M'
    - 'Ж' в начале или 'ЖЕНЩИН' → 'F'
    
    Args:
        category: Значение поля Категория (например, "М 23-39", "Женщины")
        
    Returns:
        'M', 'F' или None если не удалось определить
    """
    if pd.isna(category) or not isinstance(category, str):
        return None
    
    category = category.strip().upper()
    
    if category.startswith('М') or 'МУЖЧИН' in category:
        return 'M'
    elif category.startswith('Ж') or 'ЖЕНЩИН' in category:
        return 'F'
    else:
        return None


def parse_status(status_value: object, time_seconds: float | None) -> str:
    """
    Определить статус финиша участника.
    
    Args:
        status_value: Значение поля Статус из Excel
        time_seconds: Распарсенное время финиша
        
    Returns:
        'OK', 'DNF', 'DNS' или 'DSQ'
    """
    if pd.notna(status_value):
        status_str = str(status_value).strip().upper()
        if 'DNF' in status_str:
            return 'DNF'
        elif 'DNS' in status_str:
            return 'DNS'
        elif 'DSQ' in status_str:
            return 'DSQ'
    
    # Если статус пустой, но время есть — финишировал
    if time_seconds is not None and time_seconds > 0:
        return 'OK'
    
    # Если нет ни статуса, ни времени — считаем DNF
    return 'DNF'


# ------
# Пример использования:
#
# from utils import parse_time_to_seconds, parse_gender_from_category, parse_status
#
# # Парсинг времени
# time1 = parse_time_to_seconds("02:33:51")  # 9231.0
# time2 = parse_time_to_seconds("2:33:51")   # 9231.0
# time3 = parse_time_to_seconds("33:51")     # 2031.0
# time4 = parse_time_to_seconds("invalid")   # None
#
# print(f"02:33:51 = {time1} секунд")
# print(f"33:51 = {time3} секунд")
#
# # Определение пола
# gender1 = parse_gender_from_category("М 23-39")        # 'M'
# gender2 = parse_gender_from_category("Женщины 40-49")  # 'F'
# gender3 = parse_gender_from_category("Unknown")        # None
#
# print(f"'М 23-39' → {gender1}")
# print(f"'Женщины 40-49' → {gender2}")
#
# # Определение статуса
# status1 = parse_status(None, 9231.0)     # 'OK'
# status2 = parse_status('DNF', None)      # 'DNF'
# status3 = parse_status(None, None)       # 'DNF'
#
# print(f"Время есть, статус пуст → {status1}")
# print(f"Статус DNF → {status2}")
