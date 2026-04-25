"""``python -m zcollection3`` entry point."""
from __future__ import annotations

import sys

from .main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
