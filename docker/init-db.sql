-- MAGNET Database Initialization Script
-- PostgreSQL 16 for agent system, design storage, and embeddings

-- Note: pgvector extension is optional for Day 1
-- If pgvector is installed, uncomment the line below:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- AGENTS TABLE
-- ============================================================================
-- Tracks all agents in the MAGNET system
CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,  -- e.g., 'geometry', 'hydrostatics', 'optimizer'
    capabilities JSONB,          -- Agent-specific capabilities as JSON
    status VARCHAR(20) DEFAULT 'inactive',  -- 'active', 'inactive', 'busy'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP
);

-- Index for fast agent lookup by status
CREATE INDEX idx_agents_status ON agents(status);

-- ============================================================================
-- DESIGNS TABLE
-- ============================================================================
-- Stores vessel designs and their parameters
CREATE TABLE designs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    parent_design_id INTEGER REFERENCES designs(id),  -- For design lineage
    parameters JSONB NOT NULL,    -- All design parameters as JSON
    created_by INTEGER REFERENCES agents(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    confidence_score DECIMAL(3,2) DEFAULT 0.0,  -- Overall design confidence
    status VARCHAR(20) DEFAULT 'pending'  -- 'pending', 'analyzing', 'complete', 'failed'
);

-- Indexes for design queries
CREATE INDEX idx_designs_parent ON designs(parent_design_id);
CREATE INDEX idx_designs_status ON designs(status);
CREATE INDEX idx_designs_created_at ON designs(created_at DESC);

-- ============================================================================
-- SIMULATIONS TABLE
-- ============================================================================
-- Stores simulation results for designs
CREATE TABLE simulations (
    id SERIAL PRIMARY KEY,
    design_id INTEGER REFERENCES designs(id) ON DELETE CASCADE,
    simulation_type VARCHAR(50) NOT NULL,  -- 'hydrostatics', 'resistance', 'seakeeping'
    inputs JSONB,                          -- Simulation input parameters
    results JSONB,                         -- Simulation output results
    confidence_score DECIMAL(3,2),         -- Confidence in results (0.00-1.00)
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'running', 'complete', 'failed'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for simulation queries
CREATE INDEX idx_simulations_design ON simulations(design_id);
CREATE INDEX idx_simulations_type ON simulations(simulation_type);
CREATE INDEX idx_simulations_status ON simulations(status);

-- ============================================================================
-- CONTEXT EMBEDDINGS TABLE
-- ============================================================================
-- Stores semantic embeddings for RAG and context retrieval
-- Note: vector type requires pgvector extension. Using BYTEA for Day 1.
CREATE TABLE context_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,               -- Original text content
    embedding BYTEA,                      -- Vector embedding as binary (upgrade to vector type later)
    metadata JSONB,                       -- Additional metadata (source, timestamp, etc.)
    importance_score DECIMAL(3,2),        -- Importance for context pruning (0.00-1.00)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for metadata searches
CREATE INDEX idx_context_metadata ON context_embeddings USING gin(metadata);

-- ============================================================================
-- MESSAGES TABLE
-- ============================================================================
-- Persistent message log (backup for Redis streams)
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    from_agent VARCHAR(100),
    to_agent VARCHAR(100),
    message_type VARCHAR(50),             -- 'task_request', 'result', 'query', etc.
    content JSONB,                        -- Message payload
    priority INTEGER DEFAULT 5,           -- 1 (highest) to 10 (lowest)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP
);

-- Indexes for message routing
CREATE INDEX idx_messages_to_agent ON messages(to_agent, processed);
CREATE INDEX idx_messages_priority ON messages(priority DESC, created_at);

-- ============================================================================
-- OPTIMIZATION RUNS TABLE
-- ============================================================================
-- Tracks multi-objective optimization runs
CREATE TABLE optimization_runs (
    id SERIAL PRIMARY KEY,
    baseline_design_id INTEGER REFERENCES designs(id),
    objectives JSONB NOT NULL,            -- Optimization objectives and weights
    constraints JSONB,                    -- Design constraints
    algorithm VARCHAR(50),                -- 'NSGA-II', 'PSO', etc.
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'complete', 'failed'
    num_generations INTEGER,
    population_size INTEGER,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- ============================================================================
-- OPTIMIZATION INDIVIDUALS TABLE
-- ============================================================================
-- Individual designs generated during optimization
CREATE TABLE optimization_individuals (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id) ON DELETE CASCADE,
    design_id INTEGER REFERENCES designs(id) ON DELETE CASCADE,
    generation INTEGER,
    fitness_values JSONB,                 -- Multi-objective fitness scores
    rank INTEGER,                         -- Pareto rank
    crowding_distance DECIMAL(10,4),      -- Diversity metric
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for optimization queries
CREATE INDEX idx_opt_individuals_run ON optimization_individuals(run_id);
CREATE INDEX idx_opt_individuals_rank ON optimization_individuals(rank);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================
-- Insert system agent
INSERT INTO agents (name, type, capabilities, status) VALUES
    ('system', 'orchestrator', '{"coordination": true, "task_management": true}', 'active');

-- ============================================================================
-- VIEWS
-- ============================================================================
-- View for active design lineage
CREATE VIEW design_lineage AS
WITH RECURSIVE lineage AS (
    SELECT id, name, parent_design_id, parameters, version, 1 as depth
    FROM designs
    WHERE parent_design_id IS NULL
    UNION ALL
    SELECT d.id, d.name, d.parent_design_id, d.parameters, d.version, l.depth + 1
    FROM designs d
    INNER JOIN lineage l ON d.parent_design_id = l.id
)
SELECT * FROM lineage;

-- View for design performance summary
CREATE VIEW design_performance AS
SELECT
    d.id,
    d.name,
    d.version,
    COUNT(s.id) as simulation_count,
    AVG(s.confidence_score) as avg_confidence,
    MAX(s.completed_at) as last_simulation
FROM designs d
LEFT JOIN simulations s ON d.id = s.design_id
GROUP BY d.id, d.name, d.version;

-- ============================================================================
-- GRANTS
-- ============================================================================
-- Grant permissions to magnet_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO magnet_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO magnet_user;
GRANT ALL PRIVILEGES ON ALL VIEWS IN SCHEMA public TO magnet_user;
