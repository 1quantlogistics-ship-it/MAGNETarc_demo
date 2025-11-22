"""
MAGNET Day 1 Integration Tests

Tests the complete infrastructure pipeline:
- Docker services (PostgreSQL, Redis, Neo4j)
- Database connectivity and schemas
- Redis message queue
- Message round-trip (send/receive)
- Database agent message processing

Run with:
    pytest tests/integration/test_day1_pipeline.py -v
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory.database import get_db_session, test_connection, get_table_info
from orchestration.message_queue import MessageQueue
from sqlalchemy import text


class TestInfrastructure:
    """Test all infrastructure components."""

    def test_database_connection(self):
        """Test PostgreSQL database connectivity."""
        assert test_connection(), "Database connection failed"

    def test_database_schema(self):
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

        tables = get_table_info()
        existing_tables = set(tables.keys())

        missing_tables = required_tables - existing_tables
        assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"

    def test_redis_connection(self):
        """Test Redis connectivity."""
        mq = MessageQueue()
        assert mq.test_connectivity(), "Redis connection failed"

    def test_redis_stats(self):
        """Test Redis server stats retrieval."""
        mq = MessageQueue()
        stats = mq.get_stats()

        assert "redis_version" in stats
        assert "connected_clients" in stats
        assert stats["redis_version"].startswith("7.")  # Expecting Redis 7.x


class TestMessageQueue:
    """Test message queue functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear test queues before and after each test."""
        mq = MessageQueue()
        mq.clear_queue("test_sender")
        mq.clear_queue("test_receiver")
        yield
        mq.clear_queue("test_sender")
        mq.clear_queue("test_receiver")

    def test_message_send(self):
        """Test sending a message to a queue."""
        mq = MessageQueue()

        message = mq.send_message(
            to_agent="test_receiver",
            message_type="test",
            content={"data": "test_value"},
            from_agent="test_sender",
        )

        assert message["to"] == "test_receiver"
        assert message["from"] == "test_sender"
        assert message["type"] == "test"
        assert message["content"]["data"] == "test_value"

    def test_message_receive_blocking(self):
        """Test receiving a message (blocking mode)."""
        mq = MessageQueue()

        # Send message
        mq.send_message(
            to_agent="test_receiver",
            message_type="test",
            content={"value": 123},
            from_agent="test_sender",
        )

        # Receive message
        received = mq.receive_messages("test_receiver", block=True, timeout=2)

        assert received is not None
        assert received["type"] == "test"
        assert received["content"]["value"] == 123

    def test_message_receive_non_blocking(self):
        """Test receiving a message (non-blocking mode)."""
        mq = MessageQueue()

        # Send message
        mq.send_message(
            to_agent="test_receiver",
            message_type="ping",
            content={},
            from_agent="test_sender",
        )

        # Receive message
        received = mq.receive_messages("test_receiver", block=False)

        assert received is not None
        assert received["type"] == "ping"

    def test_message_round_trip(self):
        """Test complete message send and receive."""
        mq = MessageQueue()

        test_content = {"test": "data", "value": 42, "timestamp": "2025-01-22T12:00:00"}

        # Send
        sent = mq.send_message(
            to_agent="test_receiver",
            message_type="test_round_trip",
            content=test_content,
            from_agent="test_sender",
        )

        # Receive
        received = mq.receive_messages("test_receiver", block=False)

        assert received is not None
        assert received["content"] == test_content
        assert received["from"] == "test_sender"
        assert received["to"] == "test_receiver"

    def test_queue_length(self):
        """Test getting queue length."""
        mq = MessageQueue()

        # Initially empty
        assert mq.get_queue_length("test_receiver") == 0

        # Send 3 messages
        for i in range(3):
            mq.send_message(
                to_agent="test_receiver",
                message_type="test",
                content={"index": i},
            )

        assert mq.get_queue_length("test_receiver") == 3

    def test_peek_messages(self):
        """Test peeking at messages without removing them."""
        mq = MessageQueue()

        # Send 2 messages
        mq.send_message(
            to_agent="test_receiver", message_type="msg1", content={"id": 1}
        )
        mq.send_message(
            to_agent="test_receiver", message_type="msg2", content={"id": 2}
        )

        # Peek
        messages = mq.peek_messages("test_receiver", count=2)

        assert len(messages) == 2
        assert messages[0]["content"]["id"] == 2  # Newest first
        assert messages[1]["content"]["id"] == 1

        # Queue should still have 2 messages
        assert mq.get_queue_length("test_receiver") == 2


class TestDatabaseOperations:
    """Test database operations."""

    def test_system_agent_exists(self):
        """Test that system agent was created during initialization."""
        with get_db_session() as session:
            result = session.execute(
                text("SELECT * FROM agents WHERE name = 'system'")
            )
            agent = result.fetchone()

            assert agent is not None
            assert agent.name == "system"
            assert agent.type == "orchestrator"
            assert agent.status == "active"

    def test_insert_design(self):
        """Test inserting a design into the database."""
        test_design = {
            "vessel_name": "Test Vessel",
            "LOA": 50.0,
            "Beam": 10.0,
            "Draft": 3.0,
        }

        with get_db_session() as session:
            query = text(
                """
                INSERT INTO designs (name, parameters, created_at, status)
                VALUES (:name, :params, NOW(), 'test')
                RETURNING id
                """
            )

            result = session.execute(
                query, {"name": "Test Vessel", "params": json.dumps(test_design)}
            )

            design_id = result.fetchone()[0]
            session.commit()

            assert design_id is not None
            assert design_id > 0

            # Verify it was inserted
            verify_result = session.execute(
                text("SELECT name, status FROM designs WHERE id = :id"),
                {"id": design_id},
            )

            row = verify_result.fetchone()
            assert row.name == "Test Vessel"
            assert row.status == "test"

            # Cleanup
            session.execute(text("DELETE FROM designs WHERE id = :id"), {"id": design_id})
            session.commit()

    def test_design_lineage_view(self):
        """Test the design_lineage view exists and is queryable."""
        with get_db_session() as session:
            result = session.execute(text("SELECT * FROM design_lineage LIMIT 1"))
            # Should not raise an error, even if empty
            rows = result.fetchall()
            # View exists and is queryable
            assert True


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup."""
        mq = MessageQueue()
        mq.clear_queue("integration_test_agent")
        yield
        mq.clear_queue("integration_test_agent")

        # Cleanup test designs
        with get_db_session() as session:
            session.execute(
                text("DELETE FROM designs WHERE name LIKE 'Integration Test%'")
            )
            session.commit()

    def test_message_to_database_flow(self):
        """
        Test a message can be sent via queue and then stored in database.

        This simulates the flow:
        1. Agent 2 sends design data via message queue
        2. Database agent receives message
        3. Database agent stores design in PostgreSQL
        """
        mq = MessageQueue()

        # Step 1: Send design data via message queue (simulating Agent 2)
        test_design_data = {
            "vessel_name": "Integration Test Vessel",
            "principal_dimensions": {"LOA": 48.0, "Beam": 8.5, "Draft": 2.5},
            "hydrostatics": {"displacement": 300.0, "GM": 2.5},
            "confidence": 0.98,
        }

        mq.send_message(
            to_agent="integration_test_agent",
            message_type="store_baseline",
            content=test_design_data,
            from_agent="test_geometry_loader",
        )

        # Step 2: Receive the message
        message = mq.receive_messages("integration_test_agent", block=False)

        assert message is not None
        assert message["type"] == "store_baseline"
        assert message["content"]["vessel_name"] == "Integration Test Vessel"

        # Step 3: Store in database (simulating what database agent would do)
        with get_db_session() as session:
            query = text(
                """
                INSERT INTO designs (name, parameters, created_at, status, confidence_score)
                VALUES (:name, :params, NOW(), 'complete', :confidence)
                RETURNING id
                """
            )

            result = session.execute(
                query,
                {
                    "name": test_design_data["vessel_name"],
                    "params": json.dumps(test_design_data),
                    "confidence": test_design_data["confidence"],
                },
            )

            design_id = result.fetchone()[0]
            session.commit()

            # Verify it was stored
            verify_result = session.execute(
                text("SELECT id, name, status, confidence_score FROM designs WHERE id = :id"),
                {"id": design_id},
            )

            row = verify_result.fetchone()
            assert row.id == design_id
            assert row.name == "Integration Test Vessel"
            assert row.status == "complete"
            assert float(row.confidence_score) == 0.98

        print(f"✓ End-to-end test passed! Design ID: {design_id}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
