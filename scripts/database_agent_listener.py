#!/usr/bin/env python3
"""
Database Agent Listener

Listens for messages on the 'database_agent' queue and processes them.
Handles storing baseline designs, simulation results, and other data
into the PostgreSQL database.

Message Types Handled:
    - store_baseline: Store a baseline vessel design
    - store_design: Store a generated design variant
    - store_simulation: Store simulation results
    - query_designs: Query designs from database

Usage:
    # Run the listener (blocking)
    python scripts/database_agent_listener.py

    # Or use as module
    from scripts.database_agent_listener import DatabaseAgent
    agent = DatabaseAgent()
    agent.run()
"""

import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from memory.database import get_db_session, test_connection
from orchestration.message_queue import MessageQueue
from sqlalchemy import text


class DatabaseAgent:
    """
    Agent that listens for database operations and executes them.
    """

    def __init__(self, agent_name: str = "database_agent"):
        """
        Initialize the database agent.

        Args:
            agent_name: Name of this agent (for queue routing)
        """
        self.agent_name = agent_name
        self.mq = MessageQueue()
        self.running = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"DatabaseAgent '{agent_name}' initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"\n🛑 Received shutdown signal ({signum})")
        self.stop()

    def run(self):
        """
        Main agent loop - listens for messages and processes them.
        """
        logger.info(f"🚀 DatabaseAgent '{self.agent_name}' starting...")

        # Test connections
        if not test_connection():
            logger.error("Database connection failed. Exiting.")
            return

        if not self.mq.test_connectivity():
            logger.error("Redis connection failed. Exiting.")
            return

        logger.info(f"✓ All connections verified")
        logger.info(f"👂 Listening on queue: agent:{self.agent_name}:queue")
        logger.info(f"   Press Ctrl+C to stop\n")

        self.running = True

        while self.running:
            try:
                # Block for up to 5 seconds waiting for messages
                message = self.mq.receive_messages(self.agent_name, timeout=5)

                if message:
                    self._process_message(message)
                else:
                    # No message received, print heartbeat
                    logger.debug(".", end="", flush=True)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"DatabaseAgent '{self.agent_name}' stopped")

    def stop(self):
        """Stop the agent loop."""
        self.running = False

    def _process_message(self, message: Dict[str, Any]):
        """
        Process a received message.

        Args:
            message: Message dictionary
        """
        message_type = message.get("type")
        content = message.get("content", {})
        from_agent = message.get("from", "unknown")

        logger.info(f"\n📨 Processing {message_type} from {from_agent}")

        try:
            if message_type == "store_baseline":
                self._store_baseline(content, from_agent)
            elif message_type == "store_design":
                self._store_design(content, from_agent)
            elif message_type == "store_simulation":
                self._store_simulation(content, from_agent)
            elif message_type == "query_designs":
                self._query_designs(content, from_agent)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def _store_baseline(self, data: Dict[str, Any], from_agent: str):
        """
        Store a baseline design in the database.

        Args:
            data: Design data including vessel_name, parameters, etc.
            from_agent: Name of agent sending the data
        """
        vessel_name = data.get("vessel_name", "Unknown Vessel")

        with get_db_session() as session:
            # Insert into designs table
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
                    "name": vessel_name,
                    "params": json.dumps(data),
                    "confidence": data.get("confidence", 1.0),
                },
            )

            design_id = result.fetchone()[0]
            session.commit()

            logger.info(f"✅ Stored baseline design '{vessel_name}' with ID: {design_id}")

            # Send confirmation back to sender
            self.mq.send_message(
                to_agent=from_agent,
                message_type="baseline_stored",
                content={"design_id": design_id, "vessel_name": vessel_name},
                from_agent=self.agent_name,
            )

    def _store_design(self, data: Dict[str, Any], from_agent: str):
        """
        Store a generated design variant.

        Args:
            data: Design data
            from_agent: Sender agent name
        """
        design_name = data.get("name", "Design Variant")
        parent_id = data.get("parent_design_id")

        with get_db_session() as session:
            query = text(
                """
                INSERT INTO designs (name, parent_design_id, parameters, created_at, status)
                VALUES (:name, :parent_id, :params, NOW(), :status)
                RETURNING id
                """
            )

            result = session.execute(
                query,
                {
                    "name": design_name,
                    "parent_id": parent_id,
                    "params": json.dumps(data.get("parameters", {})),
                    "status": data.get("status", "pending"),
                },
            )

            design_id = result.fetchone()[0]
            session.commit()

            logger.info(f"✅ Stored design '{design_name}' with ID: {design_id}")

            # Send confirmation
            self.mq.send_message(
                to_agent=from_agent,
                message_type="design_stored",
                content={"design_id": design_id, "name": design_name},
                from_agent=self.agent_name,
            )

    def _store_simulation(self, data: Dict[str, Any], from_agent: str):
        """
        Store simulation results.

        Args:
            data: Simulation data including design_id, simulation_type, results
            from_agent: Sender agent name
        """
        design_id = data.get("design_id")
        sim_type = data.get("simulation_type")

        with get_db_session() as session:
            query = text(
                """
                INSERT INTO simulations (
                    design_id, simulation_type, inputs, results,
                    confidence_score, status, started_at, completed_at
                )
                VALUES (
                    :design_id, :sim_type, :inputs, :results,
                    :confidence, 'complete', NOW(), NOW()
                )
                RETURNING id
                """
            )

            result = session.execute(
                query,
                {
                    "design_id": design_id,
                    "sim_type": sim_type,
                    "inputs": json.dumps(data.get("inputs", {})),
                    "results": json.dumps(data.get("results", {})),
                    "confidence": data.get("confidence_score", 0.95),
                },
            )

            sim_id = result.fetchone()[0]
            session.commit()

            logger.info(f"✅ Stored {sim_type} simulation (ID: {sim_id}) for design {design_id}")

            # Send confirmation
            self.mq.send_message(
                to_agent=from_agent,
                message_type="simulation_stored",
                content={"simulation_id": sim_id, "design_id": design_id},
                from_agent=self.agent_name,
            )

    def _query_designs(self, data: Dict[str, Any], from_agent: str):
        """
        Query designs from database.

        Args:
            data: Query parameters (limit, status, etc.)
            from_agent: Sender agent name
        """
        limit = data.get("limit", 10)
        status = data.get("status")

        with get_db_session() as session:
            if status:
                query = text(
                    """
                    SELECT id, name, created_at, status
                    FROM designs
                    WHERE status = :status
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                )
                result = session.execute(query, {"status": status, "limit": limit})
            else:
                query = text(
                    """
                    SELECT id, name, created_at, status
                    FROM designs
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                )
                result = session.execute(query, {"limit": limit})

            designs = []
            for row in result:
                designs.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "status": row[3],
                    }
                )

            logger.info(f"📊 Query returned {len(designs)} designs")

            # Send results back
            self.mq.send_message(
                to_agent=from_agent,
                message_type="query_results",
                content={"designs": designs, "count": len(designs)},
                from_agent=self.agent_name,
            )


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("MAGNET Database Agent")
    logger.info("=" * 60)

    agent = DatabaseAgent()
    agent.run()


if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/database_agent.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )

    main()
