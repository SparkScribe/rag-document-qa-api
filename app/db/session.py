"""Database engine and session factory."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.tables import DocumentRecord  # noqa: F401 — register model with metadata


def _sqlite_connect_args(database_url: str) -> dict[str, bool]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def ensure_sqlite_parent_dir(database_url: str) -> None:
    """Create parent directory for file-backed SQLite databases."""
    if not database_url.startswith("sqlite:///./"):
        return
    db_path = database_url.removeprefix("sqlite:///./")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str) -> Engine:
    ensure_sqlite_parent_dir(database_url)
    if database_url in {"sqlite:///:memory:", "sqlite://"}:
        return create_engine(
            database_url,
            connect_args=_sqlite_connect_args(database_url),
            poolclass=StaticPool,
        )
    return create_engine(
        database_url,
        connect_args=_sqlite_connect_args(database_url),
    )


def init_database(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
