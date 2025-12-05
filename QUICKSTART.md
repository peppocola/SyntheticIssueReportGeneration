# Quick Start Guide for Experiments

This guide provides step-by-step instructions for running all experiments.

## Prerequisites

```bash
# Install dependencies
pip install pandas scikit-learn numpy matplotlib PyYAML

# For ModernBERT (optional, requires GPU for better performance)
pip install transformers torch datasets accelerate

# For zero-shot (optional)
pip install sentence-transformers

# For generation (requires Ollama installed)
pip install ollama pydantic
```

## Running Quick Tests (No GPU Required)

### 1. Run Unit Tests

```bash
cd tests
python test_pipeline.py
```

Expected output: All 15 tests should pass.

### 2. Test Traditional ML on Small Subset

```bash
# Create small test subsets
head -21 train_StackOverFlow.csv > /tmp/train_subset.csv
head -11 test_StackOverFlow.csv > /tmp/test_subset.csv

# Run traditional ML
cd traditional_ml
python train_traditional_ml.py \
    --train_file /tmp/train_subset.csv \
    --test_file /tmp/test_subset.csv \
    --models logistic_regression \
    --feature_types tfidf \
    --ngram_ranges 1,1
```

Expected output: Results table with accuracy, F1 score, etc.

### 3. Test EDA Script

```bash
cd datasets
python eda.py \
    --file /tmp/train_subset.csv \
    --output_dir /tmp/eda_test
```

Expected output: Statistics and visualizations saved to /tmp/eda_test/

### 4. Test Dataset Import

```bash
cd datasets
python import_datasets.py \
    --dataset github_emotion \
    --download \
    --output_dir /tmp/datasets_test
```

Expected output: Train and test CSV files created.

## Running Full Experiments

### Experiment 1: Traditional ML Comparison

```bash
cd traditional_ml

# Run all models with different features
python train_traditional_ml.py \
    --train_file ../train_StackOverFlow.csv \
    --test_file ../test_StackOverFlow.csv \
    --output_file ../results_traditional_ml.json

# Results saved in results_traditional_ml.json
```

### Experiment 2: ModernBERT Fine-tuning

**Note: Requires GPU for reasonable training time. For testing without GPU, use very small samples:**

```bash
cd modernbert_pipeline

# Small subset for testing (no GPU)
python train_modernbert.py \
    --train_file /tmp/train_subset.csv \
    --test_file /tmp/test_subset.csv \
    --samples_per_class 5 \
    --num_epochs 1 \
    --batch_size 2 \
    --output_file ../results_modernbert_test.json

# Full experiment (with GPU)
python train_modernbert.py \
    --train_file ../train_StackOverFlow.csv \
    --test_file ../test_StackOverFlow.csv \
    --samples_per_class 100 \
    --num_epochs 3 \
    --output_file ../results_modernbert.json
```

### Experiment 3: Zero-shot Classification

**Note: First run may download large models (BART, Sentence-BERT)**

```bash
cd zero_shot_classifier

# Test on small subset
python zero_shot_classification.py \
    --train_file /tmp/train_subset.csv \
    --test_file /tmp/test_subset.csv \
    --methods sentence-bert \
    --output_file ../results_zeroshot_test.json

# Full experiment
python zero_shot_classification.py \
    --train_file ../train_StackOverFlow.csv \
    --test_file ../test_StackOverFlow.csv \
    --methods bart sentence-bert \
    --output_file ../results_zeroshot.json
```

### Experiment 4: Hyperparameter Grid Search

**Note: Requires Ollama installed and running**

```bash
# Grid search for temperature
python generation_grid_search.py \
    --param_name temperature \
    --param_values 0.5 0.7 0.9 \
    --n_generations 5 \
    --model llama3.2:1b

# Results saved in grid_search_results/
```

### Experiment 5: Compare All Models

```bash
# After running experiments 1, 2, and 3
python compare_models.py \
    --traditional_ml results_traditional_ml.json \
    --modernbert results_modernbert.json \
    --zero_shot results_zeroshot.json \
    --output model_comparison.csv
```

## Running with Configuration File

Instead of passing many arguments, use the config file:

```bash
# Edit configs/experiments_config.yaml to set your parameters
# Then run experiments

python traditional_ml/train_traditional_ml.py \
    --train_file train_StackOverFlow.csv \
    --config configs/experiments_config.yaml

python modernbert_pipeline/train_modernbert.py \
    --train_file train_StackOverFlow.csv \
    --config configs/experiments_config.yaml
```

## Working with New Datasets

### Import GitHub Emotion Dataset

```bash
cd datasets

# Download and preprocess
python import_datasets.py \
    --dataset github_emotion \
    --download \
    --output_dir .

# Run EDA
python eda.py \
    --file github_emotion_train.csv \
    --text_column Text \
    --label_column Polarity \
    --output_dir ../eda_results_github

# Train models on new dataset
cd ../traditional_ml
python train_traditional_ml.py \
    --train_file ../datasets/github_emotion_train.csv \
    --test_file ../datasets/github_emotion_test.csv
```

### Import Custom Dataset

```bash
# Your dataset should have columns: text/Text, label/Polarity
cd datasets

python import_datasets.py \
    --dataset nasa_bugs \
    --input_file path/to/your/data.csv \
    --output_dir .
```

## Expected Results Format

### Traditional ML Output
```json
{
  "dataset": "train_StackOverFlow.csv",
  "train_size": 3000,
  "test_size": 1000,
  "results": [
    {
      "model": "svm",
      "feature_type": "tfidf",
      "ngram_range": [1, 1],
      "metrics": {
        "accuracy": 0.85,
        "f1": 0.84,
        "precision": 0.85,
        "recall": 0.84,
        "train_time": 2.5
      }
    }
  ]
}
```

### ModernBERT Output
```json
{
  "model_name": "answerdotai/ModernBERT-base",
  "samples_per_class": 100,
  "metrics": {
    "accuracy": 0.88,
    "f1": 0.87,
    "precision": 0.88,
    "recall": 0.87
  },
  "train_time": 120.5
}
```

### Zero-shot Output
```json
{
  "results": [
    {
      "model": "BART-large-mnli",
      "accuracy": 0.75,
      "f1": 0.74,
      "inference_time": 45.2
    }
  ]
}
```

## Troubleshooting

### Out of Memory (GPU)
- Reduce batch size: `--batch_size 4`
- Reduce samples: `--samples_per_class 50`
- Use CPU instead: The code auto-detects GPU

### Slow Training
- Use smaller subsets for testing
- For traditional ML: reduce `--max_features`
- For deep learning: reduce `--num_epochs`

### Missing Dependencies
```bash
# Install all dependencies at once
pip install -r traditional_ml/requirements.txt
pip install -r modernbert_pipeline/requirements.txt
pip install -r zero_shot_classifier/requirements.txt
```

### Ollama Not Working
- Ensure Ollama is installed: `curl https://ollama.ai/install.sh | sh`
- Start Ollama: `ollama serve`
- Pull model: `ollama pull llama3.2:1b`

## Performance Expectations

### On CPU (without GPU)
- **Traditional ML**: 5-30 seconds per model
- **ModernBERT**: Very slow, not recommended for full dataset
- **Zero-shot**: 2-10 minutes depending on dataset size

### On GPU
- **Traditional ML**: Same as CPU (doesn't use GPU)
- **ModernBERT**: 5-15 minutes for 100 samples/class
- **Zero-shot**: 1-5 minutes (BART uses GPU if available)

## Recommended Workflow

1. **Quick test**: Run tests and small subset experiments
2. **EDA**: Analyze your dataset characteristics
3. **Baseline**: Train traditional ML models (fast)
4. **Advanced**: Train ModernBERT (if GPU available)
5. **Zero-shot**: Run zero-shot classifiers
6. **Compare**: Use compare_models.py to generate comparison table
7. **Generation**: Run grid search for generation hyperparameters

## Next Steps

After running experiments:
1. Review results JSON files
2. Analyze model_comparison.csv
3. Check EDA visualizations
4. Iterate with different hyperparameters
5. Try on new datasets

For more details, see [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)
