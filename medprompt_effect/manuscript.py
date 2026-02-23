from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Template


@dataclass(frozen=True)
class ManuscriptContext:
    title: str
    magic_prompt: str
    model_panel: list[str]
    datasets: list[dict[str, Any]]
    key_results: dict[str, Any]
    tables: dict[str, str] | None = None


def render_template(template_path: str | Path, ctx: ManuscriptContext) -> str:
    tpl = Template(Path(template_path).read_text(encoding="utf-8"))
    return tpl.render(**ctx.__dict__)


def write_manuscript(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
