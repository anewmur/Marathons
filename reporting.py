from __future__ import annotations

from pathlib import Path


def _print_and_write_text(console_text: str, file_text: str, file_path: Path) -> None:
    print(console_text)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(file_text + "\n", encoding="utf-8")


def dump_references_report(model: object, output_path: str | Path) -> None:
    output_file = Path(output_path)

    console_lines: list[str] = []
    file_lines: list[str] = []

    console_lines.append("")
    console_lines.append("=" * 60)
    console_lines.append("Trace references:")
    console_lines.append(str(model.references.head()))

    file_lines.append("")
    file_lines.append("=" * 60)
    file_lines.append("Trace references:")
    file_lines.append(model.references.to_string(index=False))

    console_lines.append("")
    console_lines.append("=" * 60)
    console_lines.append("Age references (by race):")

    file_lines.append("")
    file_lines.append("=" * 60)
    file_lines.append("Age references (by race):")

    age_references = getattr(model, "age_references", {})
    for race_id_value in age_references:
        dataframe = age_references[race_id_value]

        console_lines.append("")
        console_lines.append(f"{race_id_value}:")
        console_lines.append(str(dataframe.head()))

        file_lines.append("")
        file_lines.append(f"{race_id_value}:")
        file_lines.append(dataframe.to_string(index=False))

    console_text = "\n".join(console_lines)
    file_text = "\n".join(file_lines)
    _print_and_write_text(console_text, file_text, output_file)
