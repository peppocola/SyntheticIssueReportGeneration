# Pipeline Documentation

This document describes the unified pipeline for thesis experiments including traditional ML, ModernBERT, zero-shot classification, and synthetic data generation with hyperparameter tuning.

## Directory Structure

```
.
├── configs/                    # YAML configuration files
│   └── experiments_config.yaml # Unified experiment configuration
├── traditional_ml/            # Traditional ML pipeline
│   ├── train_traditional_ml.py
│   └── requirements.txt
├── modernbert_pipeline/       # ModernBERT fine-tuning
│   ├── train_modernbert.py
│   └── requirements.txt
├── zero_shot_classifier/      # Zero-shot classification
│   ├── zero_shot_classification.py
│   └── requirements.txt
├── datasets/                  # Dataset management
│   ├── import_datasets.py
│   └── eda.py
├── tests/                     # Unit tests
│   └── test_pipeline.py
├── generation_grid_search.py  # Hyperparameter grid search
├── fewShot_generation/        # Few-shot generation (updated)
├── zeroShot_generation/       # Zero-shot generation (updated)
└── SetFit/                    # SetFit training
```

## 1. Traditional ML Pipeline

Train traditional ML models (SVM, Random Forest, Logistic Regression, Naive Bayes) with various feature extraction methods.

### Usage

```bash
cd traditional_ml

# Basic usage
python train_traditional_ml.py \
    --train_file ../train_StackOverFlow.csv \
    --test_file ../test_StackOverFlow.csv

# With specific models and features
python train_traditional_ml.py \
    --train_file ../train_StackOverFlow.csv \
    --models svm logistic_regression \
    --feature_types tfidf \
    --ngram_ranges 1,1 1,2 \
    --max_features 5000

# Using config file
python train_traditional_ml.py \
    --train_file ../train_StackOverFlow.csv \
    --config ../configs/experiments_config.yaml
```

### Features
- **Models**: SVM, Random Forest, Logistic Regression, Naive Bayes
- **Feature extraction**: Bag-of-Words, TF-IDF
- **N-grams**: Unigrams, Bigrams, Trigrams
- **Metrics**: Accuracy, F1, Precision, Recall, Training Time

## 2. ModernBERT Fine-tuning Pipeline

Fine-tune ModernBERT on a subset of data (e.g., 100 samples per class).

### Usage

```bash
cd modernbert_pipeline

# Basic usage with subset
python train_modernbert.py \
    --train_file ../train_StackOverFlow.csv \
    --test_file ../test_StackOverFlow.csv \
    --samples_per_class 100

# Full training
python train_modernbert.py \
    --train_file ../train_StackOverFlow.csv \
    --samples_per_class 0 \
    --num_epochs 3

# Using config file
python train_modernbert.py \
    --train_file ../train_StackOverFlow.csv \
    --config ../configs/experiments_config.yaml
```

### Features
- Default model: `answerdotai/ModernBERT-base`
- Subset training for efficiency
- Configurable epochs, batch size, learning rate
- Automatic GPU detection

## 3. Zero-shot Classification Pipeline

Classify without fine-tuning using BART-large-mnli and Sentence-BERT.

### Usage

```bash
cd zero_shot_classifier

# Using BART
python zero_shot_classification.py \
    --train_file ../train_StackOverFlow.csv \
    --methods bart

# Using Sentence-BERT
python zero_shot_classification.py \
    --train_file ../train_StackOverFlow.csv \
    --methods sentence-bert

# Using both methods
python zero_shot_classification.py \
    --train_file ../train_StackOverFlow.csv \
    --methods bart sentence-bert

# Using config file
python zero_shot_classification.py \
    --train_file ../train_StackOverFlow.csv \
    --config ../configs/experiments_config.yaml
```

### Features
- BART-large-mnli for zero-shot classification
- Sentence-BERT with semantic similarity
- No training required
- Metrics: Accuracy, F1, Precision, Recall, Inference Time

## 4. Generation with Hyperparameter Tuning

### Updated Generation Scripts

Both few-shot and zero-shot generation scripts now support:
- Temperature control
- Top-p sampling
- Top-k sampling
- Configuration from YAML files

#### Few-shot Generation

```bash
cd fewShot_generation

# Basic usage with custom hyperparameters
python fewShot_generation.py \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 50 \
    --n_generazioni 50

# Using config file
python fewShot_generation.py \
    --config ../configs/experiments_config.yaml
```

#### Zero-shot Generation

```bash
cd zeroShot_generation

# Basic usage with custom hyperparameters
python zeroShot_generation.py \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 40 \
    --generations 50

# Using config file
python zeroShot_generation.py \
    --config ../configs/experiments_config.yaml
```

### Grid Search for Hyperparameters

Systematically explore hyperparameter space with grid search.

```bash
# Search temperature values
python generation_grid_search.py \
    --param_name temperature \
    --param_values 0.5 0.7 0.9 1.0 \
    --n_generations 10

# Search top-p values
python generation_grid_search.py \
    --param_name top_p \
    --param_values 0.8 0.9 0.95 1.0 \
    --n_generations 10

# Search top-k values
python generation_grid_search.py \
    --param_name top_k \
    --param_values 20 40 50 100 \
    --n_generations 10

# Using config file
python generation_grid_search.py \
    --config configs/experiments_config.yaml \
    --param_name temperature
```

**Output**: Results saved in `grid_search_results/` directory with:
- Individual JSON files per configuration
- Complete results JSON
- CSV summary with success rates

## 5. Dataset Management

### Import and Preprocess Datasets

```bash
cd datasets

# Import GitHub emotion dataset (sample)
python import_datasets.py \
    --dataset github_emotion \
    --download \
    --output_dir ../datasets

# Import NASA bug/feature dataset
python import_datasets.py \
    --dataset nasa_bugs \
    --input_file path/to/nasa_data.csv \
    --output_dir ../datasets
```

### Exploratory Data Analysis (EDA)

```bash
cd datasets

# Run EDA on any dataset
python eda.py \
    --file ../train_StackOverFlow.csv \
    --output_dir ../eda_results

# For GitHub emotion dataset
python eda.py \
    --file github_emotion_train.csv \
    --text_column text \
    --label_column emotion
```

**Output**: 
- EDA report JSON
- Visualizations: label distribution, text length distribution, etc.

## 6. Testing

Run unit tests for critical functions:

```bash
cd tests

# Run all tests
python test_pipeline.py

# Or using pytest
pytest test_pipeline.py -v
```

Tests include:
- Data loading and stratification
- Feature engineering (BoW, TF-IDF, n-grams)
- ML model training (small scale)
- Metrics computation
- Configuration loading

**Note**: Tests are designed to run on small subsets (10-20 records) without GPU.

## 7. Configuration File

The unified configuration file `configs/experiments_config.yaml` centralizes all experiment settings:

```yaml
datasets:
  stackoverflow:
    train_file: "train_StackOverFlow.csv"
    test_file: "test_StackOverFlow.csv"

traditional_ml:
  models: ["svm", "random_forest", "logistic_regression", "naive_bayes"]
  feature_types: ["bow", "tfidf"]
  ngram_ranges: ["1,1", "1,2", "1,3"]

modernbert:
  model_name: "answerdotai/ModernBERT-base"
  samples_per_class: 100
  num_epochs: 3

generation:
  temperature_grid: [0.5, 0.7, 0.8, 0.9, 1.0]
  top_p_grid: [0.8, 0.9, 0.95, 1.0]
  top_k_grid: [20, 40, 50, 100]
  default:
    temperature: 0.8
    top_p: 0.9
    top_k: 40
```

## 8. Best Practices

### Quick Testing
All scripts support testing with small subsets:

```bash
# Traditional ML with small subset
python train_traditional_ml.py \
    --train_file train_subset_20.csv \
    --test_file test_subset_10.csv

# ModernBERT with minimal samples
python train_modernbert.py \
    --train_file train_subset_20.csv \
    --samples_per_class 5 \
    --num_epochs 1

# Grid search with few generations
python generation_grid_search.py \
    --n_generations 5
```

### Logging and Tracking

The config file includes WandB support (currently a placeholder):

```yaml
wandb:
  project: "synthetic-issue-generation"
  enabled: true
```

To enable WandB logging, install wandb and update scripts accordingly.

## 9. Example Workflows

### Complete Pipeline Example

```bash
# 1. Import and preprocess new dataset
cd datasets
python import_datasets.py --dataset github_emotion --download

# 2. Run EDA
python eda.py --file github_emotion_train.csv --text_column text --label_column emotion

# 3. Train traditional ML models
cd ../traditional_ml
python train_traditional_ml.py --train_file ../datasets/github_emotion_train.csv

# 4. Train ModernBERT (subset)
cd ../modernbert_pipeline
python train_modernbert.py --train_file ../datasets/github_emotion_train.csv --samples_per_class 100

# 5. Zero-shot classification
cd ../zero_shot_classifier
python zero_shot_classification.py --train_file ../datasets/github_emotion_train.csv

# 6. Grid search for generation
cd ..
python generation_grid_search.py --param_name temperature --n_generations 10
```

### Comparison Workflow

```bash
# Train all models and collect metrics
python traditional_ml/train_traditional_ml.py --train_file train.csv --output_file results_traditional.json
python modernbert_pipeline/train_modernbert.py --train_file train.csv --output_file results_modernbert.json
python zero_shot_classifier/zero_shot_classification.py --train_file train.csv --output_file results_zeroshot.json

# Compare results (create custom comparison script or manually review JSON files)
```

## 10. Requirements

Install dependencies for each component:

```bash
# Traditional ML
pip install -r traditional_ml/requirements.txt

# ModernBERT
pip install -r modernbert_pipeline/requirements.txt

# Zero-shot
pip install -r zero_shot_classifier/requirements.txt

# Generation (already installed in fewShot/zeroShot directories)
pip install -r fewShot_generation/requirements_fewShot.txt

# Datasets (minimal dependencies)
pip install pandas scikit-learn matplotlib
```

## Notes

- All scripts use `random_state=42` for reproducibility
- Data splits use stratification to maintain label distribution
- GPU is automatically detected and used when available
- Results are saved in JSON format for easy comparison
- Visualizations are saved as PNG files
