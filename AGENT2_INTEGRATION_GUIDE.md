# Agent 2 Integration Guide
## How to Use Agent 1's Infrastructure

**For:** Agent 2 - CAD/Physics Lead
**Date:** 2025-11-22
**Status:** Infrastructure Ready ✅

---

## Quick Start

### 1. Verify Infrastructure is Running

```bash
# Check Docker services
docker-compose ps

# Should show all healthy:
# - magnet-postgres (port 5432)
# - magnet-redis (port 6379)
# - magnet-neo4j (ports 7474, 7687)
```

### 2. Activate Virtual Environment

```bash
cd /Users/bengibson/MAGNETarc_demo
source venv/bin/activate
```

### 3. Test Connectivity

```bash
# Quick test
python -c "from orchestration.message_queue import MessageQueue; mq = MessageQueue(); print('✓ Ready' if mq.test_connectivity() else '✗ Failed')"
```

---

## Sending M48 Baseline Data

### Method 1: Using Message Queue (Recommended)

```python
from orchestration.message_queue import MessageQueue

# Initialize message queue
mq = MessageQueue()

# Your M48 baseline data (from Orca3D export)
m48_baseline = {
    "vessel_name": "M48 Patrol Boat",
    "vessel_type": "patrol_vessel",
    "principal_dimensions": {
        "LOA_m": 47.5,
        "LWL_m": 45.0,
        "Beam_m": 8.2,
        "Draft_m": 2.4,
        "Depth_m": 4.0
    },
    "hydrostatics": {
        "displacement_tonnes": 285.0,
        "LCB_m": 22.5,
        "LCF_m": 23.1,
        "KB_m": 1.3,
        "BM_m": 4.2,
        "GM_m": 2.8,
        "waterplane_area_m2": 310.0,
        "wetted_surface_m2": 420.0,
        "Cb": 0.45,
        "Cp": 0.58,
        "Cm": 0.78,
        "Cwp": 0.82
    },
    "performance": {
        "max_speed_knots": 28.0,
        "cruise_speed_knots": 20.0,
        "range_nm": 1200.0
    },
    "data_source": "orca3d_analysis",
    "confidence": 0.95
}

# Send to database agent
mq.send_message(
    to_agent="database_agent",
    message_type="store_baseline",
    content=m48_baseline,
    from_agent="geometry_loader"  # Your agent name
)

print("✓ M48 baseline data sent to database agent")

# Optional: Wait for confirmation
response = mq.receive_messages("geometry_loader", timeout=10)
if response and response['type'] == 'baseline_stored':
    design_id = response['content']['design_id']
    print(f"✓ M48 stored in database with ID: {design_id}")
```

### Method 2: Direct Database Access

```python
from memory.database import get_db_session
from sqlalchemy import text
import json

# Your M48 data
m48_data = { ... }  # Same as above

# Store directly
with get_db_session() as session:
    query = text("""
        INSERT INTO designs (name, parameters, created_at, status, confidence_score)
        VALUES (:name, :params, NOW(), 'complete', :confidence)
        RETURNING id
    """)

    result = session.execute(query, {
        'name': m48_data['vessel_name'],
        'params': json.dumps(m48_data),
        'confidence': m48_data.get('confidence', 0.95)
    })

    design_id = result.fetchone()[0]
    print(f"✓ M48 stored with ID: {design_id}")
```

---

## Running the Database Agent

The database agent needs to be running to process messages.

### Start Database Agent Listener

```bash
# In Terminal 1
source venv/bin/activate
python scripts/database_agent_listener.py

# You should see:
# ✓ All connections verified
# 👂 Listening on queue: agent:database_agent:queue
```

### Send Data from Agent 2 Script

```bash
# In Terminal 2
source venv/bin/activate
python scripts/load_m48_baseline.py  # Your script
```

The database agent will automatically:
1. Receive your message
2. Store design in PostgreSQL
3. Send confirmation back to you

---

## Querying Stored Designs

### Query via Message Queue

```python
from orchestration.message_queue import MessageQueue

mq = MessageQueue()

# Request designs
mq.send_message(
    to_agent="database_agent",
    message_type="query_designs",
    content={"limit": 10, "status": "complete"},
    from_agent="your_agent_name"
)

# Receive results
response = mq.receive_messages("your_agent_name", timeout=5)
designs = response['content']['designs']

for design in designs:
    print(f"Design {design['id']}: {design['name']} ({design['status']})")
```

### Query Directly

```python
from memory.database import execute_raw_sql

# Get all designs
designs = execute_raw_sql(
    "SELECT id, name, created_at, status FROM designs ORDER BY created_at DESC LIMIT 10"
)

for design in designs:
    print(f"{design['id']}: {design['name']}")

# Get M48 specifically
m48 = execute_raw_sql(
    "SELECT * FROM designs WHERE name LIKE :pattern",
    {"pattern": "%M48%"}
)
```

---

## Message Queue API Reference

### Sending Messages

```python
mq.send_message(
    to_agent="recipient_name",      # Who receives it
    message_type="message_type",    # e.g., "store_baseline", "analyze"
    content={...},                  # Your data payload
    from_agent="your_name",         # Your agent name
    priority=5                      # 1 (high) to 10 (low), default 5
)
```

### Receiving Messages

```python
# Blocking (wait up to 30 seconds)
message = mq.receive_messages("your_agent_name", timeout=30)

# Non-blocking (poll)
message = mq.receive_messages("your_agent_name", block=False)

# Check message
if message:
    print(f"From: {message['from']}")
    print(f"Type: {message['type']}")
    print(f"Content: {message['content']}")
```

### Queue Management

```python
# Check how many messages waiting
count = mq.get_queue_length("your_agent_name")
print(f"{count} messages pending")

# Peek at messages without consuming
messages = mq.peek_messages("your_agent_name", count=5)
for msg in messages:
    print(f"Pending: {msg['type']} from {msg['from']}")

# Clear all messages (use with caution!)
mq.clear_queue("your_agent_name")
```

---

## Database Schema Reference

### Key Tables

**designs** - Vessel designs
```sql
- id (serial primary key)
- name (varchar)
- parent_design_id (integer, references designs)
- parameters (jsonb) -- All your design data
- created_at (timestamp)
- status ('pending', 'analyzing', 'complete', 'failed')
- confidence_score (decimal 0.00-1.00)
```

**simulations** - Simulation results
```sql
- id (serial primary key)
- design_id (integer, references designs)
- simulation_type ('hydrostatics', 'resistance', 'seakeeping', etc.)
- inputs (jsonb)
- results (jsonb)
- confidence_score (decimal)
- status, started_at, completed_at
```

**agents** - Agent registry
```sql
- id, name, type, capabilities (jsonb)
- status ('active', 'inactive', 'busy')
- created_at, last_active
```

---

## Example: Complete Workflow

### 1. Export M48 from Orca3D
```python
# Your Rhino/Orca3D script
# ... export hydrostatics to JSON
```

### 2. Send to Database Agent
```python
from orchestration.message_queue import MessageQueue
import json

# Load your exported data
with open('m48_hydrostatics.json') as f:
    m48_data = json.load(f)

# Send to database
mq = MessageQueue()
mq.send_message(
    to_agent="database_agent",
    message_type="store_baseline",
    content=m48_data,
    from_agent="orca3d_exporter"
)

print("✓ M48 data sent to database")
```

### 3. Verify Storage
```python
from memory.database import execute_raw_sql

# Check it was stored
designs = execute_raw_sql(
    "SELECT id, name, created_at FROM designs WHERE name LIKE '%M48%'"
)

print(f"✓ Found {len(designs)} M48 designs in database")
for d in designs:
    print(f"  ID {d['id']}: {d['name']} (created {d['created_at']})")
```

---

## Troubleshooting

### Problem: "Cannot connect to Redis"

**Solution:**
```bash
# Check Redis is running
docker ps | grep redis

# Restart if needed
docker-compose restart redis

# Test connectivity
docker exec magnet-redis redis-cli ping
# Should respond: PONG
```

### Problem: "Cannot connect to PostgreSQL"

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
docker exec magnet-postgres psql -U magnet_user -d magnet -c "SELECT 1;"

# Check logs
docker logs magnet-postgres
```

### Problem: "Database agent not responding"

**Solution:**
```bash
# Make sure database agent is running
python scripts/database_agent_listener.py

# Check message queue
python -c "
from orchestration.message_queue import MessageQueue
mq = MessageQueue()
print(f'Messages pending: {mq.get_queue_length(\"database_agent\")}')
"
```

### Problem: "Module not found errors"

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Verify it shows (venv) in prompt
which python
# Should show: /Users/bengibson/MAGNETarc_demo/venv/bin/python
```

---

## Testing Your Integration

Use the integration test as a template:

```bash
# Run the integration test suite
python tests/integration/test_day1_infrastructure.py

# Should show:
# ✓ PostgreSQL Connection
# ✓ PostgreSQL Schema
# ✓ Redis Connection
# ✓ Message Queue
# ✓ Database Insert/Query
# ✓ End-to-End Pipeline
# 🎉 All tests PASSED!
```

---

## Contact & Support

If you encounter issues:

1. Check [AGENT1_DAY1_SUMMARY.md](AGENT1_DAY1_SUMMARY.md) for infrastructure status
2. Run integration tests: `python tests/integration/test_day1_infrastructure.py`
3. Check Docker logs: `docker-compose logs postgres redis neo4j`
4. Verify environment: `cat .env` (make sure database credentials are correct)

**Agent 1 Infrastructure Status:** ✅ All systems operational

---

**Last Updated:** 2025-11-22 15:55 EST
**Agent 1 - Infrastructure Lead**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
