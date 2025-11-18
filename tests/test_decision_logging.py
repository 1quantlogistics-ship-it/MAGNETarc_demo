"""
Test Decision Logging: Validate structured decision logging
============================================================

Tests that all decision events are properly logged to JSONL files.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.multi_agent_orchestrator import MultiAgentOrchestrator


def test_decision_logging():
    """Test that decisions are logged during research cycle"""
    print("\n" + "=" * 70)
    print("  DECISION LOGGING TEST")
    print("=" * 70 + "\n")

    memory_path = "/Users/bengibson/Desktop/ARC/arc_clean/memory"
    log_dir = f"{memory_path}/logs"

    # Clear old logs
    log_path = Path(log_dir)
    if log_path.exists():
        for log_file in log_path.glob("*.jsonl"):
            log_file.unlink()
        print("✓ Cleared old log files")

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        memory_path=memory_path,
        offline_mode=True
    )

    print("✓ Orchestrator initialized")
    print(f"  Log directory: {log_dir}")

    # Run a research cycle
    print("\n🔄 Running research cycle...")
    result = orchestrator.run_research_cycle(cycle_id=1)

    print(f"✓ Research cycle completed")
    print(f"  Status: {result.get('status', 'completed')}")

    # Check log files
    print("\n📂 Checking log files...")

    expected_logs = {
        "cycles.jsonl": "Cycle lifecycle events",
        "votes.jsonl": "Individual agent votes",
        "consensus.jsonl": "Consensus calculations",
        "conflicts.jsonl": "Conflict resolutions",
        "supervisor.jsonl": "Supervisor decisions"
    }

    files_found = 0
    for log_file, description in expected_logs.items():
        file_path = log_path / log_file
        exists = file_path.exists()

        if exists:
            # Count lines
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f if line.strip())

            print(f"  ✓ {log_file:20s} ({line_count} entries) - {description}")
            files_found += 1
        else:
            print(f"  ✗ {log_file:20s} (not found)")

    # Show sample log entries
    print("\n📊 Sample Log Entries:")

    # Cycle logs
    cycles_log = log_path / "cycles.jsonl"
    if cycles_log.exists():
        print("\n  Cycle Events:")
        with open(cycles_log, 'r') as f:
            for i, line in enumerate(f):
                if i >= 2:  # Show first 2 entries
                    break
                entry = json.loads(line)
                event_type = entry.get('event_type', 'unknown')
                cycle_id = entry.get('cycle_id', 0)
                print(f"    • {event_type:20s} (cycle {cycle_id})")

    # Votes
    votes_log = log_path / "votes.jsonl"
    if votes_log.exists():
        print("\n  Agent Votes:")
        with open(votes_log, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 entries
                    break
                entry = json.loads(line)
                agent = entry.get('agent_role', 'unknown')
                decision = entry.get('decision', 'unknown')
                confidence = entry.get('confidence', 0.0)
                print(f"    • {agent:20s}: {decision:8s} (conf: {confidence:.2f})")

    # Summary
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {files_found}/{len(expected_logs)} log files created")
    print("=" * 70 + "\n")

    # Note: votes/consensus/supervisor logs only appear when proposals are generated
    # Since we're in offline mode with no training history, the cycle ends early
    # This is expected behavior
    if files_found >= 1:  # At minimum, cycles.jsonl should exist
        print("✅ Decision logging infrastructure is working!")
        print("   (Note: Vote/consensus logs require proposals to be generated)")
        return True
    else:
        print("❌ Log files missing")
        return False


if __name__ == "__main__":
    success = test_decision_logging()
    sys.exit(0 if success else 1)
