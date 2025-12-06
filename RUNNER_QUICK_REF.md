# Quick Reference: Experiment Runner

## TL;DR

Run all experiments with a single command:

```bash
pip install requests
python run_all_experiments.py
```

## What Gets Run

1. **traditional_ml** - SVM, Random Forest, Logistic Regression, Naive Bayes
2. **modernbert** - ModernBERT fine-tuning (100 samples/class)
3. **zero_shot** - Zero-shot classification with Sentence-BERT
4. **setfit** - SetFit few-shot learning
5. **roberta** - RoBERTa fine-tuning
6. **few_shot_generation** - Few-shot generation (requires Ollama)
7. **zero_shot_generation** - Zero-shot generation (requires Ollama)

## Common Commands

```bash
# Run all experiments
python run_all_experiments.py

# Run specific experiments
python run_all_experiments.py --experiments traditional_ml,setfit

# Skip venv creation (use existing)
python run_all_experiments.py --skip-venv

# Get help
python run_all_experiments.py --help
```

## For Generation Experiments

Start Ollama in a separate terminal:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run experiments
python run_all_experiments.py
```

## Results Location

All results saved to: `experiment_results/<experiment_name>/`

Each experiment gets:
- Log file: `<experiment>_<timestamp>.log`
- Results JSON: Metrics and outputs
- Additional files: Model-specific outputs

## Virtual Environments

Created in: `.venvs/<experiment_name>/`

Each experiment has isolated dependencies - no conflicts!

## Troubleshooting

**Venv creation fails?**
```bash
# Install python3-venv
sudo apt-get install python3-venv  # Ubuntu/Debian
```

**Experiment fails?**
```bash
# Check the log file
cat experiment_results/<experiment>/<experiment>_*.log

# Run individually for debugging
source .venvs/<experiment>/bin/activate
cd <experiment_directory>
python <script>.py <args>
```

**Ollama not detected?**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

## More Info

- Complete guide: [EXPERIMENT_RUNNER_GUIDE.md](EXPERIMENT_RUNNER_GUIDE.md)
- Pipeline details: [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)
- Step-by-step: [QUICKSTART.md](QUICKSTART.md)
