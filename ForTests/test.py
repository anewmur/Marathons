from pathlib import Path

import pandas as pd


def collect_raw_race_names(data_dir_path: str) -> None:
    data_dir = Path(data_dir_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {data_dir}")

    excel_files = sorted(data_dir.glob("*.xls*"))
    if not excel_files:
        print(f"В каталоге {data_dir} нет файлов *.xls*")
        return

    raw_names: set[str] = set()

    for filepath in excel_files:
        try:
            meta = pd.read_excel(filepath, header=None, nrows=1)
        except Exception as error:
            print(f"Ошибка чтения {filepath.name}: {error}")
            continue

        if meta.empty:
            print(f"Пустой meta-блок в {filepath.name}")
            continue

        value = meta.iloc[0, 0]
        if isinstance(value, str):
            name = value.strip()
            if name:
                raw_names.add(name)

    print(f"Найдено {len(raw_names)} уникальных сырых названий забегов:\n")
    for race_name in sorted(raw_names):
        print(race_name)


if __name__ == "__main__":
    collect_raw_race_names(r"C:\Users\andre\github\Marathons\Data")
