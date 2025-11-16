#!/bin/bash
# ARC Deployment Verification Script
# Tests that a fresh clone/deployment is functional

echo "=== ARC Deployment Verification ==="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"
if [[ $python_version == *"3.10"* ]] || [[ $python_version == *"3.11"* ]] || [[ $python_version == *"3.12"* ]]; then
    echo "   ✅ Python version OK"
else
    echo "   ⚠️  Python 3.10+ recommended"
fi
echo ""

# Check directory structure
echo "2. Checking directory structure..."
required_dirs=("api" "scripts" "config")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/ exists"
    else
        echo "   ❌ $dir/ missing"
        exit 1
    fi
done
echo ""

# Check core files
echo "3. Checking core files..."
required_files=(
    "api/control_plane.py"
    "api/dashboard.py"
    "api/full_cycle_orchestrator.py"
    "requirements.txt"
    "README.md"
)
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file exists"
    else
        echo "   ❌ $file missing"
        exit 1
    fi
done
echo ""

# Check if dependencies are installed
echo "4. Checking Python dependencies..."
if python3 -c "import fastapi" 2>/dev/null; then
    echo "   ✅ fastapi installed"
else
    echo "   ⚠️  fastapi not installed (run: pip install -r requirements.txt)"
fi

if python3 -c "import streamlit" 2>/dev/null; then
    echo "   ✅ streamlit installed"
else
    echo "   ⚠️  streamlit not installed (run: pip install -r requirements.txt)"
fi
echo ""

# Check if memory is initialized
echo "5. Checking memory initialization..."
if [ -d "memory" ]; then
    echo "   ✅ memory/ directory exists"
    if [ -f "memory/system_state.json" ]; then
        echo "   ✅ memory initialized"
    else
        echo "   ⚠️  memory not initialized (run: python scripts/init_memory.py)"
    fi
else
    echo "   ⚠️  memory/ not found (run: python scripts/init_memory.py)"
fi
echo ""

# Check if experiments directory exists
echo "6. Checking workspace directories..."
for dir in "experiments" "logs" "snapshots"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/ exists"
    else
        echo "   ⚠️  $dir/ not found (will be created on first use)"
    fi
done
echo ""

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Initialize memory: python scripts/init_memory.py"
echo "3. Set up vLLM endpoint (see README.md)"
echo "4. Start Control Plane: python api/control_plane.py"
echo "5. Start Dashboard: streamlit run api/dashboard.py"
