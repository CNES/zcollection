"""``python -m zcollection`` entry point."""

import sys

from .main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
