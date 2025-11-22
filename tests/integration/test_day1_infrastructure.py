#!/usr/bin/env python3
"""
MAGNET Day 1 Infrastructure Test

Standalone test script that verifies:
- Docker services running
- PostgreSQL connectivity and schema
- Redis connectivity
- Message queue functionality
- End-to-end message + database flow

Run with:
    python tests/integration/test_day1_infrastructure.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import only what we need to avoid dependency issues
import psycopg
import redis
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "magnet"),
    "user": os.getenv("POSTGRES_USER", "magnet_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "magnet_dev_password"),
}

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
}


def test_postgres_connection():
    """Test PostgreSQL connection."""
    try:
        conn = psycopg.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        conn.close()
        logger.info(f"✓ PostgreSQL connected: {version.split(',')[0]}")
        return True
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return False


def test_postgres_schema():
    """Test that all required tables exist."""
    required_tables = {
        "agents",
        "designs",
        "simulations",
        "messages",
        "context_embeddings",
        "optimization_runs",
        "optimization_individuals",
    }

    try:
        conn = psycopg.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """
        )

        existing_tables = {row[0] for row in cursor.fetchall()}
        missing = required_tables - existing_tables

        conn.close()

        if missing:
            logger.error(f"✗ Missing tables: {missing}")
            return False

        logger.info(f"✓ All {len(required_tables)} required tables exist")
        return True

    except Exception as e:
        logger.error(f"✗ Schema check failed: {e}")
        return False


def test_redis_connection():
    """Test Redis connection."""
    try:
        client = redis.Redis(**REDIS_CONFIG, decode_responses=True)
        pong = client.ping()
        info = client.info()
        client.close()

        logger.info(f"✓ Redis connected: v{info['redis_version']}")
        return True
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return False


def test_message_queue():
    """Test message queue send/receive."""
    try:
        client = redis.Redis(**REDIS_CONFIG, decode_responses=True)

        # Clear test queue
        queue_name = "agent:test_agent:queue"
        client.delete(queue_name)

        # Send message
        message = {
            "from": "test_sender",
            "to": "test_agent",
            "type": "test",
            "content": {"data": "Hello World"},
        }

        client.lpush(queue_name, json.dumps(message))
        logger.info("  📤 Sent test message")

        # Receive message
        _, received_json = client.brpop(queue_name, timeout=2)
        received = json.loads(received_json)

        # Cleanup
        client.close()

        if received["content"]["data"] == "Hello World":
            logger.info("  📥 Received test message correctly")
            logger.info("✓ Message queue round-trip successful")
            return True
        else:
            logger.error("✗ Message content mismatch")
            return False

    except Exception as e:
        logger.error(f"✗ Message queue test failed: {e}")
        return False


def test_database_insert():
    """Test inserting data into database."""
    try:
        conn = psycopg.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Insert test design
        test_data = {"vessel": "Test Ship", "LOA": 50.0}

        cursor.execute(
            """
            INSERT INTO designs (name, parameters, created_at, status)
            VALUES (%s, %s, NOW(), %s)
            RETURNING id
            """,
            ("Day 1 Test Design", json.dumps(test_data), "test"),
        )

        design_id = cursor.fetchone()[0]
        conn.commit()

        # Verify
        cursor.execute("SELECT name, status FROM designs WHERE id = %s", (design_id,))
        row = cursor.fetchone()

        # Cleanup
        cursor.execute("DELETE FROM designs WHERE id = %s", (design_id,))
        conn.commit()
        conn.close()

        if row[0] == "Day 1 Test Design" and row[1] == "test":
            logger.info(f"✓ Database insert/query successful (ID: {design_id})")
            return True
        else:
            logger.error("✗ Database insert verification failed")
            return False

    except Exception as e:
        logger.error(f"✗ Database insert test failed: {e}")
        return False


def test_end_to_end():
    """Test complete message → database flow."""
    try:
        # Setup
        redis_client = redis.Redis(**REDIS_CONFIG, decode_responses=True)
        db_conn = psycopg.connect(**DB_CONFIG)
        db_cursor = db_conn.cursor()

        queue_name = "agent:e2e_test_agent:queue"
        redis_client.delete(queue_name)

        # Step 1: Send design data via message queue
        design_data = {
            "vessel_name": "E2E Test Vessel",
            "LOA": 47.5,
            "Beam": 8.2,
            "confidence": 0.95,
        }

        message = {
            "from": "geometry_loader",
            "to": "database_agent",
            "type": "store_baseline",
            "content": design_data,
        }

        redis_client.lpush(queue_name, json.dumps(message))
        logger.info("  📤 Sent design data via message queue")

        # Step 2: Receive message
        _, received_json = redis_client.brpop(queue_name, timeout=2)
        received_message = json.loads(received_json)
        logger.info("  📥 Received message")

        # Step 3: Store in database
        db_cursor.execute(
            """
            INSERT INTO designs (name, parameters, created_at, status, confidence_score)
            VALUES (%s, %s, NOW(), %s, %s)
            RETURNING id
            """,
            (
                received_message["content"]["vessel_name"],
                json.dumps(received_message["content"]),
                "complete",
                received_message["content"]["confidence"],
            ),
        )

        design_id = db_cursor.fetchone()[0]
        db_conn.commit()
        logger.info(f"  💾 Stored in database (ID: {design_id})")

        # Step 4: Verify
        db_cursor.execute(
            "SELECT name, status, confidence_score FROM designs WHERE id = %s",
            (design_id,),
        )
        row = db_cursor.fetchone()

        # Cleanup
        db_cursor.execute("DELETE FROM designs WHERE id = %s", (design_id,))
        db_conn.commit()
        redis_client.close()
        db_conn.close()

        if (
            row[0] == "E2E Test Vessel"
            and row[1] == "complete"
            and float(row[2]) == 0.95
        ):
            logger.info("✓ End-to-end pipeline successful")
            return True
        else:
            logger.error("✗ End-to-end verification failed")
            return False

    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    logger.info("=" * 70)
    logger.info("MAGNET Day 1 Infrastructure Integration Tests")
    logger.info("=" * 70)
    logger.info("")

    tests = [
        ("PostgreSQL Connection", test_postgres_connection),
        ("PostgreSQL Schema", test_postgres_schema),
        ("Redis Connection", test_redis_connection),
        ("Message Queue", test_message_queue),
        ("Database Insert/Query", test_database_insert),
        ("End-to-End Pipeline", test_end_to_end),
    ]

    results = []
    for name, test_func in tests:
        logger.info(f"Testing: {name}...")
        result = test_func()
        results.append((name, result))
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} - {name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("")
        logger.info("🎉 All Day 1 infrastructure tests PASSED!")
        return 0
    else:
        logger.info("")
        logger.error(f"⚠️  {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
