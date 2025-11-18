#!/usr/bin/env python3
"""
Decision Log Analyzer: Query and analyze multi-agent decision logs
==================================================================

Provides CLI tools to analyze JSONL decision logs:
- View voting patterns
- Analyze consensus quality
- Track supervisor overrides
- Identify controversial decisions
- Generate confidence trends
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime

from llm.decision_logger import get_decision_logger


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def analyze_voting_patterns(log_dir: str, cycle_id: int = None):
    """Analyze voting patterns across cycles"""
    print_header("VOTING PATTERN ANALYSIS")

    logger = get_decision_logger(log_dir)
    stats = logger.get_voting_stats(cycle_id=cycle_id)

    # Overall stats
    print(f"Total Votes Cast: {stats['total_votes']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2%}")

    # By decision type
    print(f"\n📊 Decision Distribution:")
    for decision, count in stats['by_decision'].items():
        pct = (count / stats['total_votes'] * 100) if stats['total_votes'] > 0 else 0
        print(f"  {decision:10s}: {count:4d} ({pct:5.1f}%)")

    # By agent
    print(f"\n🤖 Agent Voting Summary:")
    for agent_id, agent_stats in sorted(stats['by_agent'].items()):
        total = agent_stats['total']
        decisions = agent_stats['decisions']

        # Most common decision
        most_common = max(decisions.items(), key=lambda x: x[1]) if decisions else ('N/A', 0)

        print(f"  {agent_id:25s}: {total:3d} votes  (most: {most_common[0]})")


def analyze_consensus_quality(log_dir: str, cycle_id: int = None):
    """Analyze consensus quality and trends"""
    print_header("CONSENSUS QUALITY ANALYSIS")

    logger = get_decision_logger(log_dir)
    consensus_logs = logger.query_consensus(cycle_id=cycle_id, limit=1000)

    if not consensus_logs:
        print("No consensus logs found.")
        return

    # Stats
    total = len(consensus_logs)
    reached = sum(1 for c in consensus_logs if c.get('consensus_reached', False))
    failed = total - reached

    # Confidence stats
    confidences = [c.get('confidence', 0.0) for c in consensus_logs]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Weighted scores
    scores = [c.get('weighted_score', 0.0) for c in consensus_logs]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Total Consensus Attempts: {total}")
    print(f"  ✓ Reached: {reached} ({reached/total*100:.1f}%)")
    print(f"  ✗ Failed:  {failed} ({failed/total*100:.1f}%)")
    print(f"\nAverage Confidence: {avg_confidence:.2%}")
    print(f"Average Weighted Score: {avg_score:.3f}")

    # Decision distribution
    decision_counts = defaultdict(int)
    for c in consensus_logs:
        decision = c.get('final_decision', 'unknown')
        decision_counts[decision] += 1

    print(f"\n📊 Final Decision Distribution:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {decision:10s}: {count:4d} ({pct:5.1f}%)")


def analyze_conflicts(log_dir: str, cycle_id: int = None):
    """Analyze conflict resolution patterns"""
    print_header("CONFLICT RESOLUTION ANALYSIS")

    logger = get_decision_logger(log_dir)
    conflict_log_path = Path(log_dir) / "conflicts.jsonl"

    if not conflict_log_path.exists():
        print("No conflict logs found.")
        return

    conflicts = logger._query_log(conflict_log_path, cycle_id=cycle_id, limit=1000)

    if not conflicts:
        print("No conflicts detected.")
        return

    print(f"Total Conflicts: {len(conflicts)}")

    # By type
    by_type = defaultdict(int)
    for c in conflicts:
        conflict_type = c.get('conflict_type', 'unknown')
        by_type[conflict_type] += 1

    print(f"\n📊 Conflict Types:")
    for ctype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = count / len(conflicts) * 100
        print(f"  {ctype:20s}: {count:4d} ({pct:5.1f}%)")

    # Resolution strategies
    by_strategy = defaultdict(int)
    for c in conflicts:
        strategy = c.get('resolution_strategy', 'unknown')
        by_strategy[strategy] += 1

    print(f"\n🔧 Resolution Strategies:")
    for strategy, count in sorted(by_strategy.items(), key=lambda x: -x[1]):
        pct = count / len(conflicts) * 100
        print(f"  {strategy:20s}: {count:4d} ({pct:5.1f}%)")

    # Overrides
    overrides = sum(1 for c in conflicts if c.get('override_applied', False))
    print(f"\n⚠️  Overrides Applied: {overrides} ({overrides/len(conflicts)*100:.1f}%)")

    # Entropy analysis
    entropies = [c.get('entropy', 0.0) for c in conflicts]
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    max_entropy = max(entropies) if entropies else 0.0

    print(f"\n📈 Disagreement Metrics:")
    print(f"  Average Entropy: {avg_entropy:.3f}")
    print(f"  Maximum Entropy: {max_entropy:.3f}")


def analyze_supervisor_decisions(log_dir: str, cycle_id: int = None):
    """Analyze supervisor override patterns"""
    print_header("SUPERVISOR DECISION ANALYSIS")

    logger = get_decision_logger(log_dir)
    supervisor_log_path = Path(log_dir) / "supervisor.jsonl"

    if not supervisor_log_path.exists():
        print("No supervisor logs found.")
        return

    decisions = logger._query_log(supervisor_log_path, cycle_id=cycle_id, limit=1000)

    if not decisions:
        print("No supervisor decisions found.")
        return

    total = len(decisions)
    overrides = [d for d in decisions if d.get('override_consensus', False)]
    approvals = [d for d in decisions if not d.get('override_consensus', False)]

    print(f"Total Supervisor Decisions: {total}")
    print(f"  ✓ Approvals: {len(approvals)} ({len(approvals)/total*100:.1f}%)")
    print(f"  ⚠  Overrides: {len(overrides)} ({len(overrides)/total*100:.1f}%)")

    # Risk assessment distribution
    risk_counts = defaultdict(int)
    for d in decisions:
        risk = d.get('risk_assessment', 'unknown')
        risk_counts[risk] += 1

    print(f"\n🎯 Risk Assessment Distribution:")
    risk_order = ['low', 'medium', 'high', 'critical', 'unknown']
    for risk in risk_order:
        count = risk_counts.get(risk, 0)
        if count > 0:
            pct = count / total * 100
            print(f"  {risk:10s}: {count:4d} ({pct:5.1f}%)")

    # Override reasons (if any)
    if overrides:
        print(f"\n⚠️  Override Details:")
        for override in overrides[:5]:  # Show first 5
            proposal_id = override.get('proposal_id', 'unknown')
            risk = override.get('risk_assessment', 'unknown')
            reasoning = override.get('reasoning', 'No reason provided')[:60]

            print(f"  • {proposal_id}")
            print(f"    Risk: {risk}, Reason: {reasoning}...")

        if len(overrides) > 5:
            print(f"  ... and {len(overrides) - 5} more overrides")

    # Confidence stats
    confidences = [d.get('confidence', 0.0) for d in decisions]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    print(f"\n📊 Average Confidence: {avg_confidence:.2%}")


def show_recent_cycles(log_dir: str, limit: int = 10):
    """Show recent research cycles"""
    print_header(f"RECENT {limit} CYCLES")

    logger = get_decision_logger(log_dir)
    cycle_log_path = Path(log_dir) / "cycles.jsonl"

    if not cycle_log_path.exists():
        print("No cycle logs found.")
        return

    cycles = logger._query_log(cycle_log_path, limit=limit * 2)  # *2 for start+complete

    # Group by cycle_id
    cycles_by_id = defaultdict(list)
    for cycle in cycles:
        cycle_id = cycle.get('cycle_id')
        cycles_by_id[cycle_id].append(cycle)

    # Display
    for cycle_id in sorted(cycles_by_id.keys(), reverse=True)[:limit]:
        events = cycles_by_id[cycle_id]

        started = next((e for e in events if e.get('event_type') == 'cycle_started'), None)
        completed = next((e for e in events if e.get('event_type') == 'cycle_completed'), None)

        if started:
            start_time = started.get('timestamp', 'unknown')
            print(f"\n🔄 Cycle {cycle_id}")
            print(f"  Started: {start_time}")

            if completed:
                meta = completed.get('metadata', {})
                duration = meta.get('duration_seconds', 0)
                proposals = meta.get('total_proposals', 0)
                approved = meta.get('approved_proposals', 0)
                consensus_rate = meta.get('consensus_rate', 0)

                print(f"  Duration: {duration:.2f}s")
                print(f"  Proposals: {proposals} (approved: {approved})")
                print(f"  Consensus Rate: {consensus_rate:.1%}")
            else:
                print(f"  Status: INCOMPLETE")


def export_summary(log_dir: str, output_path: str):
    """Export comprehensive summary to JSON"""
    logger = get_decision_logger(log_dir)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "voting_stats": logger.get_voting_stats(),
        "consensus": {
            "logs": logger.query_consensus(limit=1000),
            "total": len(logger.query_consensus(limit=10000))
        },
        "conflicts": {
            "total": len(logger._query_log(Path(log_dir) / "conflicts.jsonl", limit=10000))
        },
        "supervisor_overrides": logger.query_supervisor_overrides(limit=1000)
    }

    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent decision logs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--log-dir',
        default='/Users/bengibson/Desktop/ARC/arc_clean/memory/logs',
        help='Directory containing decision logs'
    )

    parser.add_argument(
        '--cycle',
        type=int,
        help='Filter by specific cycle ID'
    )

    parser.add_argument(
        '--voting',
        action='store_true',
        help='Show voting pattern analysis'
    )

    parser.add_argument(
        '--consensus',
        action='store_true',
        help='Show consensus quality analysis'
    )

    parser.add_argument(
        '--conflicts',
        action='store_true',
        help='Show conflict resolution analysis'
    )

    parser.add_argument(
        '--supervisor',
        action='store_true',
        help='Show supervisor decision analysis'
    )

    parser.add_argument(
        '--cycles',
        action='store_true',
        help='Show recent research cycles'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all analyses'
    )

    parser.add_argument(
        '--export',
        metavar='PATH',
        help='Export summary to JSON file'
    )

    args = parser.parse_args()

    # If no specific analysis requested, show all
    if not any([args.voting, args.consensus, args.conflicts, args.supervisor, args.cycles, args.export]):
        args.all = True

    if args.export:
        export_summary(args.log_dir, args.export)
        return

    if args.all or args.cycles:
        show_recent_cycles(args.log_dir)

    if args.all or args.voting:
        analyze_voting_patterns(args.log_dir, args.cycle)

    if args.all or args.consensus:
        analyze_consensus_quality(args.log_dir, args.cycle)

    if args.all or args.conflicts:
        analyze_conflicts(args.log_dir, args.cycle)

    if args.all or args.supervisor:
        analyze_supervisor_decisions(args.log_dir, args.cycle)


if __name__ == "__main__":
    main()
