# Experiment Runner Guide

This guide explains how to use the automated experiment runner to execute all experiments in sequence with isolated virtual environments.

## Overview

The experiment runner provides two implementations:
1. **Bash script** (`run_all_experiments.sh`) - For Linux/macOS
2. **Python script** (`run_all_experiments.py`) - Cross-platform (recommended)

Both scripts:
- Create isolated virtual environments for each experiment
- Install dependencies automatically
- Run experiments in sequence
- Save results to `experiment_results/`
- Handle Ollama server checks for generation experiments
- Provide detailed logging

## Quick Start

### Using Python Script (Recommended)

```bash
# Install requests library (only needed once)
pip install requests

# Run all experiments
python run_all_experiments.py

# Run specific experiments
python run_all_experiments.py --experiments traditional_ml,setfit,roberta

# Skip virtual environment creation (use existing)
python run_all_experiments.py --skip-venv
```

### Using Bash Script (Linux/macOS)

```bash
# Make executable (first time only)
chmod +x run_all_experiments.sh

# Run all experiments
./run_all_experiments.sh

# Run specific experiments
./run_all_experiments.sh --experiments "traditional_ml,setfit"

# Skip virtual environment creation
./run_all_experiments.sh --skip-venv
```

## Available Experiments

The runner includes the following experiments:

1. **traditional_ml** - Traditional ML models (SVM, Random Forest, Logistic Regression, Naive Bayes)
2. **modernbert** - ModernBERT fine-tuning with subset training
3. **zero_shot** - Zero-shot classification with Sentence-BERT
4. **setfit** - SetFit few-shot learning
5. **roberta** - RoBERTa fine-tuning
6. **few_shot_generation** - Few-shot text generation with Ollama (requires Ollama server)
7. **zero_shot_generation** - Zero-shot text generation with Ollama (requires Ollama server)

## Prerequisites

### General Requirements

- Python 3.8 or higher
- `venv` module (usually included with Python)
- Dataset files: `train_StackOverFlow.csv` and `test_StackOverFlow.csv`

### For Generation Experiments

Generation experiments require Ollama to be installed and running:

```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the model (in another terminal)
ollama pull llama3.2:1b
```

The runner will automatically detect if Ollama is running and skip generation experiments if it's not available.

## Directory Structure

After running experiments, the directory structure will be:

```
.
├── .venvs/                      # Virtual environments
│   ├── traditional_ml/
│   ├── modernbert/
│   ├── zero_shot/
│   ├── setfit/
│   ├── roberta/
│   ├── few_shot_generation/
│   └── zero_shot_generation/
│
├── experiment_results/          # Results and logs
│   ├── traditional_ml/
│   │   ├── traditional_ml_20231206_120000.log
│   │   └── traditional_ml_results.json
│   ├── modernbert/
│   │   ├── modernbert_20231206_120530.log
│   │   └── modernbert_results.json
│   └── ...
│
└── run_all_experiments.py      # Main runner script
```

## Usage Examples

### Run All Experiments

```bash
python run_all_experiments.py
```

This will:
1. Create virtual environments for each experiment
2. Install dependencies
3. Run all experiments sequentially
4. Save results to `experiment_results/`

### Run Specific Experiments

```bash
# Run only traditional ML and SetFit
python run_all_experiments.py --experiments traditional_ml,setfit

# Run only generation experiments (requires Ollama)
python run_all_experiments.py --experiments few_shot_generation,zero_shot_generation
```

### Skip Virtual Environment Creation

If you've already created virtual environments and want to rerun experiments:

```bash
python run_all_experiments.py --skip-venv
```

This is useful for:
- Quick reruns without recreating environments
- Testing after making code changes
- Debugging specific experiments

### Get Help

```bash
python run_all_experiments.py --help
```

## Experiment Configuration

Each experiment runs with default parameters optimized for thesis work:

| Experiment | Key Parameters |
|------------|----------------|
| Traditional ML | All models, TF-IDF + BoW, n-grams |
| ModernBERT | 100 samples/class, 3 epochs |
| Zero-shot | Sentence-BERT method |
| SetFit | 50 samples per label |
| RoBERTa | 2 epochs, batch size 16 |
| Few-shot Gen | 10 generations, temp=0.8 |
| Zero-shot Gen | 10 generations, temp=0.8 |

To customize parameters, edit the script or run experiments individually.

## Results

### Output Files

Each experiment generates:
- **Log file**: `<experiment>_<timestamp>.log` - Complete execution log
- **Results JSON**: Metrics and model outputs
- **Other files**: Model-specific outputs (CSVs, classification reports, etc.)

### Checking Results

After running experiments:

```bash
# View experiment logs
ls -la experiment_results/*/

# Compare model results
python compare_models.py \
    --traditional_ml experiment_results/traditional_ml/traditional_ml_results.json \
    --modernbert experiment_results/modernbert/modernbert_results.json \
    --zero_shot experiment_results/zero_shot/zero_shot_results.json
```

## Troubleshooting

### Virtual Environment Issues

If virtual environment creation fails:

```bash
# Ensure venv module is installed
python3 -m venv --help

# On Ubuntu/Debian, install python3-venv
sudo apt-get install python3-venv
```

### Dependency Installation Failures

If dependency installation fails:

```bash
# Check pip is working
pip --version

# Upgrade pip
pip install --upgrade pip

# Install dependencies manually
pip install -r traditional_ml/requirements.txt
```

### Ollama Not Running

If generation experiments are skipped:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# In another terminal, verify it's running
ollama list
```

### Experiment Failures

If an experiment fails:

1. Check the log file in `experiment_results/<experiment>/`
2. Look at the last 20 lines for error details
3. Run the experiment individually for debugging:
   ```bash
   source .venvs/<experiment>/bin/activate
   cd <experiment_directory>
   python <script>.py <args>
   ```

### Out of Memory

For GPU memory issues with deep learning experiments:

```bash
# Reduce batch size or samples per class
# Edit the script and modify parameters:
# - ModernBERT: --samples_per_class 50 (instead of 100)
# - SetFit: -n 25 (instead of 50)
```

## Advanced Usage

### Custom Experiment Order

Edit the script to change experiment order:

```python
# In run_all_experiments.py
experiments = [
    ('traditional_ml', ...),  # Runs first
    ('setfit', ...),          # Runs second
    # Add or reorder as needed
]
```

### Parallel Execution

To run experiments in parallel (requires careful resource management):

```bash
# Terminal 1
python run_all_experiments.py --experiments traditional_ml

# Terminal 2
python run_all_experiments.py --experiments setfit --skip-venv

# Terminal 3
python run_all_experiments.py --experiments roberta --skip-venv
```

**Note**: Be careful with GPU memory when running deep learning experiments in parallel.

### Integration with CI/CD

For automated testing in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run experiments
  run: |
    python run_all_experiments.py --experiments traditional_ml,zero_shot
    
- name: Upload results
  uses: actions/upload-artifact@v2
  with:
    name: experiment-results
    path: experiment_results/
```

## Performance Expectations

| Experiment | Time (CPU) | Time (GPU) | Memory |
|------------|-----------|-----------|---------|
| Traditional ML | 5-10 min | N/A | 2-4 GB |
| ModernBERT | Very slow | 10-15 min | 6-8 GB |
| Zero-shot | 10-20 min | 5-10 min | 4-6 GB |
| SetFit | 15-20 min | 10-15 min | 4-6 GB |
| RoBERTa | Very slow | 15-20 min | 6-8 GB |
| Few-shot Gen | 5-10 min | N/A | 2-4 GB |
| Zero-shot Gen | 5-10 min | N/A | 2-4 GB |

**Total time**: ~1-2 hours (with GPU), 3-5 hours (CPU only)

## Best Practices

1. **Start with Small Tests**: Run one experiment first to verify setup
2. **Monitor Resources**: Check disk space and memory during execution
3. **Save Logs**: Keep log files for debugging and analysis
4. **Use GPU**: Deep learning experiments are much faster with GPU
5. **Clean Up**: Remove `.venvs/` directory to free disk space when done

## Next Steps

After running experiments:

1. Review results in `experiment_results/`
2. Compare models using `compare_models.py`
3. Generate visualizations using EDA scripts
4. Run grid search for hyperparameter tuning
5. Iterate with different configurations

For more details, see:
- [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) - Detailed pipeline documentation
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step experiment guide
- [README.md](README.md) - Repository overview
