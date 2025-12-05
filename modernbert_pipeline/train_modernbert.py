"""
ModernBERT Fine-tuning Pipeline
Fine-tunes ModernBERT model on a subset of data (e.g., 100 samples per class)
"""

import argparse
import pandas as pd
import numpy as np
import time
import json
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def load_and_sample_data(train_file, test_file=None, split_ratio=0.3, samples_per_class=100):
    """Load data and sample subset for training"""
    df_train = pd.read_csv(train_file, delimiter=';', quotechar='"')
    
    if test_file:
        df_test = pd.read_csv(test_file, delimiter=';', quotechar='"')
    else:
        # Split train data
        df_train, df_test = train_test_split(
            df_train,
            test_size=split_ratio,
            stratify=df_train['Polarity'],
            random_state=42
        )
    
    # Sample subset per class for training if specified
    if samples_per_class > 0:
        df_train_sampled = df_train.groupby('Polarity', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42)
        ).reset_index(drop=True)
        print(f"✅ Sampled {len(df_train_sampled)} training samples from {len(df_train)} total")
    else:
        df_train_sampled = df_train
        print(f"✅ Using all {len(df_train_sampled)} training samples")
    
    return df_train_sampled, df_test


def prepare_dataset(df_train, df_test):
    """Prepare HuggingFace dataset"""
    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = pd.concat([df_train['Polarity'], df_test['Polarity']])
    label_encoder.fit(all_labels)
    
    df_train['label'] = label_encoder.transform(df_train['Polarity'])
    df_test['label'] = label_encoder.transform(df_test['Polarity'])
    
    # Create datasets
    train_dataset = Dataset.from_pandas(df_train[['Text', 'label']].fillna(''))
    test_dataset = Dataset.from_pandas(df_test[['Text', 'label']].fillna(''))
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict, label_encoder


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description='ModernBERT Fine-tuning Pipeline')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test CSV file')
    parser.add_argument('--split_ratio', type=float, default=0.3, help='Test split ratio')
    parser.add_argument('--samples_per_class', type=int, default=100, 
                        help='Number of samples per class (0 for all)')
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base',
                        help='Model name from HuggingFace')
    parser.add_argument('--output_dir', type=str, default='./modernbert_results',
                        help='Output directory for model')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output_file', type=str, default='modernbert_results.json',
                        help='Output file for results')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.get('modernbert', {}).items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    print("=" * 80)
    print("ModernBERT Fine-tuning Pipeline")
    print("=" * 80)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load and sample data
    print("\n📂 Loading data...")
    df_train, df_test = load_and_sample_data(
        args.train_file, 
        args.test_file, 
        args.split_ratio,
        args.samples_per_class
    )
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Prepare dataset
    print("\n🔧 Preparing dataset...")
    dataset, label_encoder = prepare_dataset(df_train, df_test)
    num_labels = len(label_encoder.classes_)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {label_encoder.classes_}")
    
    # Load tokenizer and model
    print(f"\n🤖 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['Text'],
            padding=False,
            truncation=True,
            max_length=args.max_length
        )
    
    print("\n🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n🏋️  Training model...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save results
    results = {
        'model_name': args.model_name,
        'train_size': len(df_train),
        'test_size': len(df_test),
        'samples_per_class': args.samples_per_class,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_time': train_time,
        'metrics': {
            'accuracy': float(eval_results.get('eval_accuracy', 0)),
            'f1': float(eval_results.get('eval_f1', 0)),
            'precision': float(eval_results.get('eval_precision', 0)),
            'recall': float(eval_results.get('eval_recall', 0))
        },
        'device': device
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_file}")
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"Training time: {train_time:.2f}s")


if __name__ == '__main__':
    main()
