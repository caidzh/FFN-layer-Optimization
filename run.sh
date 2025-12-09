# Run baseline
echo ""
echo "======================================"
echo "Running baseline test..."
echo "======================================"
python ffn.py

# Run correctness tests
echo ""
echo "======================================"
echo "Running correctness tests..."
echo "======================================"
python tests/test_correctness.py

# Run benchmark
echo ""
echo "======================================"
echo "Running benchmark tests..."
echo "======================================"
python tests/test_benchmark.py