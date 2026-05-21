from pathlib import Path
from typing import Dict, List, Optional
import yaml

_ROOT = Path(__file__).resolve().parents[2]

with open(_ROOT / "config" / "config.yaml") as _f:
    _CFG = yaml.safe_load(_f)


class Universe:
    """Manages the tradeable symbol list defined in config.yaml."""

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or _CFG

    def get_symbols(self) -> List[str]:
        """Return the list of symbols from config."""
        return list(self.cfg.get("symbols", []))

    def validate_symbol(self, symbol: str) -> bool:
        """Return True if symbol is a non-empty string."""
        return isinstance(symbol, str) and bool(symbol.strip())
