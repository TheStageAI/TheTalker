#!/usr/bin/env python3
"""Entry point for the streaming TTS load client; see benchmark/README.md.

The client itself lives in ``thetalker.client`` so that it is importable from an
installed wheel; this script only makes a plain checkout runnable without one.
Installed, the same CLI is ``thetalker-bench``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thetalker.client import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
