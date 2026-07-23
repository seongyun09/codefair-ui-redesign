from __future__ import annotations

import json
from pathlib import Path


class InsuranceCodeRegistry:
    """Read the same code maps used by ingestion; never guesses an unknown code."""

    def __init__(self, data_directory: str | Path = "data") -> None:
        root = Path(data_directory)
        self.company_codes = self._read(root / "config" / "company_codes.json")
        self.product_codes = self._read(root / "config" / "product_codes.json")
        self.manifest_path = root / "vector_store_manifest.json"

    @staticmethod
    def _read(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def resolve_company(self, name: str | None) -> str | None:
        return self._resolve(name, self.company_codes, "company", "company_code")

    def resolve_product(self, name: str | None) -> str | None:
        return self._resolve(name, self.product_codes, "product", "product_code")

    def _resolve(self, name: str | None, mapping: dict[str, str], value_key: str, code_key: str) -> str | None:
        if not name:
            return None
        if name in mapping:
            return mapping[name]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("documents", []):
                if item.get(value_key) == name and item.get(code_key):
                    return item[code_key]
        return None
