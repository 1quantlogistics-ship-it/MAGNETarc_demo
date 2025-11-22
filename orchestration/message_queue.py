"""
Redis-based message queue for MAGNET agent communication.

This module provides a lightweight message passing system using Redis lists
for inter-agent communication. Messages are sent to agent-specific queues
and can be consumed synchronously or asynchronously.

Queue Naming Convention:
    agent:{agent_name}:queue

Message Format:
    {
        "from": "sender_agent_name",
        "to": "recipient_agent_name",
        "type": "message_type",
        "content": {...},
        "priority": 1-10,
        "timestamp": "ISO format"
    }

Usage:
    from orchestration.message_queue import MessageQueue

    mq = MessageQueue()

    # Send message
    mq.send_message(
        to_agent="hydrostatics_agent",
        message_type="analyze_design",
        content={"design_id": 123}
    )

    # Receive messages (blocking)
    message = mq.receive_messages("hydrostatics_agent", timeout=5)

    # Receive messages (non-blocking)
    message = mq.receive_messages("hydrostatics_agent", block=False)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "decode_responses": True,  # Automatically decode bytes to strings
    "socket_connect_timeout": 5,
    "socket_timeout": 5,
}


class MessageQueue:
    """
    Redis-based message queue for agent communication.

    Provides FIFO queue semantics using Redis lists (LPUSH/BRPOP).
    """

    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        logger.debug(f"MessageQueue initialized: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")

    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Dict[str, Any],
        from_agent: str = "system",
        priority: int = 5,
    ) -> Dict[str, Any]:
        """
        Send a message to an agent's queue.

        Args:
            to_agent: Name of the recipient agent
            message_type: Type of message (e.g., 'task_request', 'result', 'query')
            content: Message payload as dictionary
            from_agent: Name of the sender agent (default: 'system')
            priority: Message priority 1 (highest) to 10 (lowest) - currently informational

        Returns:
            dict: The sent message

        Example:
            message = mq.send_message(
                to_agent="geometry_agent",
                message_type="generate_hull",
                content={"LOA": 47.5, "Beam": 8.2},
                from_agent="orchestrator"
            )
        """
        message = {
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "content": content,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Push to agent's queue (LPUSH adds to left/head)
        queue_name = f"agent:{to_agent}:queue"
        self.redis_client.lpush(queue_name, json.dumps(message))

        logger.info(f"📤 Message sent: {from_agent} → {to_agent} ({message_type})")
        logger.debug(f"   Queue: {queue_name}, Content: {content}")

        return message

    def receive_messages(
        self, agent_name: str, block: bool = True, timeout: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Receive messages from an agent's queue.

        Args:
            agent_name: Name of the agent receiving messages
            block: If True, block until message arrives or timeout
            timeout: Timeout in seconds for blocking mode (default: 5)

        Returns:
            dict: Received message, or None if no message available

        Example:
            # Blocking mode (wait up to 5 seconds)
            message = mq.receive_messages("geometry_agent", timeout=5)

            # Non-blocking mode (poll)
            message = mq.receive_messages("geometry_agent", block=False)
        """
        queue_name = f"agent:{agent_name}:queue"

        try:
            if block:
                # BRPOP: Blocking pop from right/tail
                result = self.redis_client.brpop(queue_name, timeout=timeout)
                if result:
                    _, message_json = result
                    message = json.loads(message_json)
                    logger.info(f"📥 Message received by {agent_name}: {message['type']} from {message['from']}")
                    logger.debug(f"   Content: {message['content']}")
                    return message
            else:
                # RPOP: Non-blocking pop from right/tail
                message_json = self.redis_client.rpop(queue_name)
                if message_json:
                    message = json.loads(message_json)
                    logger.info(f"📥 Message received by {agent_name}: {message['type']} from {message['from']}")
                    return message

            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def get_queue_length(self, agent_name: str) -> int:
        """
        Get the number of messages waiting in an agent's queue.

        Args:
            agent_name: Name of the agent

        Returns:
            int: Number of pending messages
        """
        queue_name = f"agent:{agent_name}:queue"
        length = self.redis_client.llen(queue_name)
        return length

    def clear_queue(self, agent_name: str) -> int:
        """
        Clear all messages from an agent's queue.

        Args:
            agent_name: Name of the agent

        Returns:
            int: Number of messages deleted
        """
        queue_name = f"agent:{agent_name}:queue"
        length = self.redis_client.llen(queue_name)
        self.redis_client.delete(queue_name)
        logger.warning(f"🗑️  Cleared {length} messages from {agent_name}'s queue")
        return length

    def peek_messages(self, agent_name: str, count: int = 10) -> list[Dict[str, Any]]:
        """
        Peek at messages in an agent's queue without removing them.

        Args:
            agent_name: Name of the agent
            count: Maximum number of messages to peek (default: 10)

        Returns:
            list: List of messages (newest first)
        """
        queue_name = f"agent:{agent_name}:queue"
        messages_json = self.redis_client.lrange(queue_name, 0, count - 1)

        messages = []
        for msg_json in messages_json:
            try:
                messages.append(json.loads(msg_json))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message in peek: {msg_json}")

        return messages

    def test_connectivity(self) -> bool:
        """
        Test Redis connection.

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            response = self.redis_client.ping()
            if response:
                logger.info("✓ Redis connection successful")
                logger.info(f"  Server: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
                return True
            return False
        except redis.ConnectionError as e:
            logger.error(f"✗ Redis connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error testing Redis: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis server statistics.

        Returns:
            dict: Server info and statistics
        """
        try:
            info = self.redis_client.info()
            stats = {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}


# Initialize logging
logger.add(
    "logs/message_queue.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


if __name__ == "__main__":
    # Test script
    logger.info("=== Testing Message Queue ===\n")

    mq = MessageQueue()

    # Test connectivity
    if not mq.test_connectivity():
        logger.error("Cannot connect to Redis. Exiting.")
        exit(1)

    logger.info("\n=== Redis Stats ===")
    stats = mq.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n=== Message Send/Receive Test ===")

    # Clear test queue
    mq.clear_queue("test_agent")

    # Send test message
    mq.send_message(
        to_agent="test_agent",
        message_type="test",
        content={"message": "Hello World", "timestamp": datetime.now(timezone.utc).isoformat()},
        from_agent="system",
    )

    # Check queue length
    length = mq.get_queue_length("test_agent")
    logger.info(f"Queue length: {length}")

    # Peek at message (without removing)
    messages = mq.peek_messages("test_agent", count=1)
    logger.info(f"Peeked message: {messages[0] if messages else None}")

    # Receive message (non-blocking)
    received = mq.receive_messages("test_agent", block=False)
    logger.info(f"Received message: {received}")

    # Verify queue is empty
    length = mq.get_queue_length("test_agent")
    logger.info(f"Queue length after receive: {length}")

    logger.info("\n✓ Message queue test complete!")
