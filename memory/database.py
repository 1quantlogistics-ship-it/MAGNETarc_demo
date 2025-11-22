"""
Database connection and session management for MAGNET system.

This module provides SQLAlchemy-based connection pooling and session management
for PostgreSQL database operations. It uses psycopg3 (psycopg[binary]) for
Python 3.13 compatibility.

Usage:
    from memory.database import get_db_session, test_connection

    # Test connectivity
    if test_connection():
        print("Database ready")

    # Use in context manager
    with get_db_session() as session:
        result = session.execute("SELECT * FROM agents")
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

# Load environment variables
load_dotenv()

# Database connection configuration
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "magnet"),
    "user": os.getenv("POSTGRES_USER", "magnet_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "magnet_dev_password"),
}

# Connection pool settings
POOL_CONFIG = {
    "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
    "pool_pre_ping": True,  # Verify connections before using
}

# Construct database URL
DATABASE_URL = (
    f"postgresql+psycopg://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    poolclass=QueuePool,
    **POOL_CONFIG,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Automatically handles commit/rollback and session cleanup.

    Usage:
        with get_db_session() as session:
            result = session.execute(text("SELECT * FROM agents"))
            # Automatically commits on success, rolls back on exception

    Yields:
        Session: SQLAlchemy database session

    Raises:
        Exception: Re-raises any exception after rollback
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """
    Test database connectivity.

    Attempts to connect to PostgreSQL and execute a simple query.

    Returns:
        bool: True if connection successful, False otherwise

    Example:
        if test_connection():
            logger.info("Database is ready")
        else:
            logger.error("Database connection failed")
    """
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✓ Database connection successful")
            logger.info(f"  PostgreSQL version: {version.split(',')[0]}")
            return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


def get_table_info() -> dict:
    """
    Get information about database tables.

    Returns:
        dict: Dictionary with table names and row counts
    """
    tables = {}
    try:
        with get_db_session() as session:
            # Get all tables
            result = session.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
            )

            for (table_name,) in result:
                # Get row count for each table
                count_result = session.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                count = count_result.scalar()
                tables[table_name] = count

        return tables
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return {}


def execute_raw_sql(query: str, params: dict = None) -> list:
    """
    Execute raw SQL query with optional parameters.

    Args:
        query: SQL query string
        params: Optional dictionary of query parameters

    Returns:
        list: List of result rows as dictionaries

    Example:
        results = execute_raw_sql(
            "SELECT * FROM agents WHERE status = :status",
            {"status": "active"}
        )
    """
    try:
        with get_db_session() as session:
            result = session.execute(text(query), params or {})

            # Convert to list of dicts
            rows = []
            for row in result:
                rows.append(dict(row._mapping))

            return rows
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise


# Initialize logging
logger.add(
    "logs/database.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


if __name__ == "__main__":
    # Test script
    logger.info("Testing database connection...")

    if test_connection():
        logger.info("\n=== Database Tables ===")
        tables = get_table_info()
        for table, count in tables.items():
            logger.info(f"  {table}: {count} rows")

        logger.info("\n=== Testing Agents Query ===")
        agents = execute_raw_sql("SELECT * FROM agents LIMIT 5")
        for agent in agents:
            logger.info(f"  Agent: {agent}")
    else:
        logger.error("Database connection test failed!")
