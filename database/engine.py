"""
engine.py — Creates and exports the shared SQLAlchemy database engine.

All other modules that need a DB connection import `engine` from here.
The connection URL is built from settings.py so credentials live in one place.
"""


# ── MODULE TAG: Database Connection Engine ──
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from config.settings import (
    DB_DIALECT, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
)

# ── CONNECTION URL GENERATION ──────────────────────────────────────────────────
# Build the SQLAlchemy connection URL from environment-loaded settings
_url_obj = URL.create(
    drivername=f"{DB_DIALECT}+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)

DB_URL: str = _url_obj.render_as_string(hide_password=False)

# ── ENGINE INSTANTIATION ───────────────────────────────────────────────────────
# Shared engine — import this wherever a DB connection is needed
engine = create_engine(DB_URL)

