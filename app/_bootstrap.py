"""
NOT IMPORTED ANYMORE - kept only to document why each entry point carries
a small inline sys.path snippet instead of a shared import.

ORIGINAL IDEA (broken)
-----------------------
The plan was for every entry point to do `import app._bootstrap`, which
would add the repo root to sys.path so that `from app.state import ...`
etc. resolve (app/ is a plain sibling of src/, not part of the installable
gyroid_utils package, so nothing puts the repo root on sys.path by default).

Why it didn't work: `streamlit run app/Home.py` only puts the *script's
own directory* (app/) on sys.path - not the repo root. That means `app`
itself is not yet an importable package the moment a page starts running,
so `import app._bootstrap` fails before this file's fix would even get a
chance to run. Chicken-and-egg.

THE ACTUAL FIX
---------------
Every entry point (Home.py and each file in app/pages/) instead inlines
the two lines directly, computed from its own __file__ (no import needed
before the fix is applied):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))

where N is how many directories up from that file the repo root is
(1 for app/Home.py, 2 for app/pages/*.py). Only *after* that line can the
file do `from app.state import ...` etc.
"""
