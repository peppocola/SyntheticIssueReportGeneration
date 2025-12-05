# SyntheticIssueReportGeneration

Unified pipeline for thesis experiments on synthetic issue report generation with traditional ML, deep learning, and zero-shot classification baselines.

## 📁 Repository Structure

```
.
├── configs/                    # YAML configuration files
├── traditional_ml/            # Traditional ML pipeline (SVM, RF, LR, NB)
├── modernbert_pipeline/       # ModernBERT fine-tuning
├── zero_shot_classifier/      # Zero-shot classification (BART, S-BERT)
├── datasets/                  # Dataset management and EDA
├── tests/                     # Unit tests
├── fewShot_generation/        # Few-shot generation with Ollama
├── zeroShot_generation/       # Zero-shot generation with Ollama
├── SetFit/                    # SetFit training
├── RoBerta/                   # RoBerta training
├── Result/                    # Result processing
├── generation_grid_search.py  # Hyperparameter grid search
├── compare_models.py          # Model comparison script
└── PIPELINE_DOCUMENTATION.md  # Detailed pipeline documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# For traditional ML
pip install -r traditional_ml/requirements.txt

# For ModernBERT
pip install -r modernbert_pipeline/requirements.txt

# For zero-shot classification
pip install -r zero_shot_classifier/requirements.txt

# For generation (Ollama required)
pip install -r fewShot_generation/requirements_fewShot.txt
```

### 2. Run Baseline Experiments

```bash
# Traditional ML models
python traditional_ml/train_traditional_ml.py \
    --train_file train_StackOverFlow.csv \
    --test_file test_StackOverFlow.csv

# ModernBERT fine-tuning (100 samples per class)
python modernbert_pipeline/train_modernbert.py \
    --train_file train_StackOverFlow.csv \
    --test_file test_StackOverFlow.csv \
    --samples_per_class 100

# Zero-shot classification
python zero_shot_classifier/zero_shot_classification.py \
    --train_file train_StackOverFlow.csv \
    --test_file test_StackOverFlow.csv
```

### 3. Generation with Hyperparameter Tuning

```bash
# Few-shot generation with custom parameters
python fewShot_generation/fewShot_generation.py \
    --temperature 0.8 --top_p 0.9 --top_k 40

# Grid search for temperature
python generation_grid_search.py \
    --param_name temperature \
    --param_values 0.5 0.7 0.9 1.0 \
    --n_generations 10
```

### 4. Dataset Management

```bash
# Import and preprocess GitHub emotion dataset
python datasets/import_datasets.py \
    --dataset github_emotion --download

# Run EDA
python datasets/eda.py \
    --file train_StackOverFlow.csv \
    --output_dir eda_results
```

### 5. Compare Models

```bash
# Compare all models
python compare_models.py \
    --traditional_ml traditional_ml_results.json \
    --modernbert modernbert_results.json \
    --zero_shot zero_shot_results.json
```

### 6. Run Tests

```bash
cd tests
python test_pipeline.py
```

## 📊 Features

### Sviluppo 1 - Baseline Alternative
- ✅ Traditional ML models (SVM, Random Forest, Logistic Regression, Naive Bayes)
- ✅ Feature extraction: Bag-of-Words, TF-IDF, n-grams
- ✅ ModernBERT fine-tuning with subset training
- ✅ Zero-shot classification (BART-large-mnli, Sentence-BERT)
- ✅ Comprehensive metrics comparison

### Sviluppo 2 - Hyperparameter Tuning
- ✅ YAML configuration support
- ✅ CLI arguments for temperature, top-p, top-k
- ✅ Grid search for systematic exploration
- ✅ Results logging and checkpointing

### Sviluppo 3 - New Datasets
- ✅ Dataset import pipeline (GitHub emotion, NASA bugs)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Preprocessing with stratified split (random_state=42)
- ✅ Compatible with all training pipelines

### Best Practices
- ✅ YAML configuration for experiments
- ✅ WandB integration (placeholder)
- ✅ Unit tests for critical functions
- ✅ Small subset testing (10-20 records, no GPU required)

## 📖 Documentation

For detailed documentation, see [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)

## 🧪 Testing

All tests are designed to run on small subsets without GPU:

```bash
cd tests
python test_pipeline.py
```

Tests cover:
- Data loading and stratification
- Feature engineering (BoW, TF-IDF, n-grams)
- ML model training
- Metrics computation
- Configuration loading

## 🎯 Usage Examples

### Example 1: Complete Pipeline on New Dataset

```bash
# 1. Import dataset
python datasets/import_datasets.py --dataset github_emotion --download

# 2. Run EDA
python datasets/eda.py --file datasets/github_emotion_train.csv

# 3. Train all models
python traditional_ml/train_traditional_ml.py \
    --train_file datasets/github_emotion_train.csv \
    --output_file results_traditional.json

python modernbert_pipeline/train_modernbert.py \
    --train_file datasets/github_emotion_train.csv \
    --output_file results_modernbert.json

# 4. Compare results
python compare_models.py \
    --traditional_ml results_traditional.json \
    --modernbert results_modernbert.json
```

### Example 2: Hyperparameter Grid Search

```bash
# Search temperature values
python generation_grid_search.py \
    --param_name temperature \
    --param_values 0.5 0.7 0.9 1.0 \
    --n_generations 10

# Results saved in grid_search_results/
```

## 📝 Configuration

Edit `configs/experiments_config.yaml` to configure:
- Dataset paths
- Model hyperparameters
- Generation parameters
- Grid search values

## 🔬 Requirements

- Python 3.8+
- scikit-learn for traditional ML
- transformers for ModernBERT and zero-shot
- Ollama for generation
- pandas, numpy for data processing

## 📄 License

See repository license.

## 👥 Contributors

- PhD Thesis Project
