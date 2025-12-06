"""
Zero-shot Classification Pipeline
Supports: BART-large-mnli, Sentence-BERT
No fine-tuning required
"""

import argparse
import pandas as pd
import numpy as np
import time
import json
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch


def load_data(train_file, test_file=None, split_ratio=0.3):
    """Load data for zero-shot classification"""
    df_train = pd.read_csv(train_file, delimiter=';', quotechar='"')
    
    if test_file:
        df_test = pd.read_csv(test_file, delimiter=';', quotechar='"')
    else:
        df_train, df_test = train_test_split(
            df_train,
            test_size=split_ratio,
            stratify=df_train['Polarity'],
            random_state=42
        )
    
    return df_train, df_test


def zero_shot_bart(texts, candidate_labels, model_name='facebook/bart-large-mnli'):
    """Zero-shot classification using BART-large-mnli"""
    print(f"\n🤖 Loading BART model: {model_name}")
    classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    
    predictions = []
    print("\n🔮 Running zero-shot classification...")
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)} samples")
        
        result = classifier(str(text), candidate_labels, multi_label=False)
        predictions.append(result['labels'][0])
    
    return predictions


def zero_shot_sentence_bert(texts, candidate_labels, model_name='all-MiniLM-L6-v2'):
    """Zero-shot classification using Sentence-BERT"""
    print(f"\n🤖 Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create label descriptions for better matching
    label_descriptions = {
        'positive': "This text expresses positive sentiment, satisfaction, or happiness",
        'negative': "This text expresses negative sentiment, frustration, or dissatisfaction",
        'neutral': "This text expresses neutral sentiment without strong emotions"
    }
    
    # Encode label descriptions
    label_embeddings = model.encode(
        [label_descriptions.get(label, label) for label in candidate_labels],
        convert_to_tensor=True
    )
    
    predictions = []
    print("\n🔮 Running Sentence-BERT classification...")
    
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)} samples")
        
        # Encode texts
        text_embeddings = model.encode(
            [str(text) for text in batch_texts],
            convert_to_tensor=True
        )
        
        # Compute similarities
        similarities = util.cos_sim(text_embeddings, label_embeddings)
        
        # Get predictions
        batch_predictions = [candidate_labels[sim.argmax().item()] for sim in similarities]
        predictions.extend(batch_predictions)
    
    return predictions


def evaluate_predictions(y_true, y_pred, model_name):
    """Compute evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Zero-shot Classification Pipeline')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test CSV file')
    parser.add_argument('--split_ratio', type=float, default=0.3, help='Test split ratio')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['bart', 'sentence-bert'],
                        help='Zero-shot methods to use')
    parser.add_argument('--bart_model', type=str, default='facebook/bart-large-mnli',
                        help='BART model name')
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-BERT model name')
    parser.add_argument('--candidate_labels', type=str, nargs='+',
                        default=['positive', 'negative', 'neutral'],
                        help='Candidate labels for classification')
    parser.add_argument('--output_file', type=str, default='zero_shot_results.json',
                        help='Output file for results')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.get('zero_shot', {}).items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    print("=" * 80)
    print("Zero-shot Classification Pipeline")
    print("=" * 80)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    df_train, df_test = load_data(args.train_file, args.test_file, args.split_ratio)
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    X_test = df_test['Text'].astype(str).fillna('').tolist()
    y_test = df_test['Polarity'].tolist()
    
    print(f"\n🎯 Candidate labels: {args.candidate_labels}")
    
    all_results = []
    
    # BART zero-shot
    if 'bart' in args.methods:
        print("\n" + "=" * 80)
        print("BART-large-mnli Zero-shot Classification")
        print("=" * 80)
        
        start_time = time.time()
        predictions_bart = zero_shot_bart(X_test, args.candidate_labels, args.bart_model)
        bart_time = time.time() - start_time
        
        metrics_bart = evaluate_predictions(y_test, predictions_bart, 'BART-large-mnli')
        metrics_bart['inference_time'] = bart_time
        
        all_results.append(metrics_bart)
        
        print(f"\n✅ BART Results:")
        print(f"Accuracy: {metrics_bart['accuracy']:.4f}")
        print(f"F1 Score: {metrics_bart['f1']:.4f}")
        print(f"Time: {bart_time:.2f}s")
    
    # Sentence-BERT zero-shot
    if 'sentence-bert' in args.methods or 'sbert' in args.methods:
        print("\n" + "=" * 80)
        print("Sentence-BERT Zero-shot Classification")
        print("=" * 80)
        
        start_time = time.time()
        predictions_sbert = zero_shot_sentence_bert(X_test, args.candidate_labels, args.sbert_model)
        sbert_time = time.time() - start_time
        
        metrics_sbert = evaluate_predictions(y_test, predictions_sbert, 'Sentence-BERT')
        metrics_sbert['inference_time'] = sbert_time
        
        all_results.append(metrics_sbert)
        
        print(f"\n✅ Sentence-BERT Results:")
        print(f"Accuracy: {metrics_sbert['accuracy']:.4f}")
        print(f"F1 Score: {metrics_sbert['f1']:.4f}")
        print(f"Time: {sbert_time:.2f}s")
    
    # Save results
    output_data = {
        'dataset': args.train_file,
        'test_size': len(df_test),
        'candidate_labels': args.candidate_labels,
        'results': all_results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model']:<30} "
              f"{result['accuracy']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['inference_time']:<10.2f}")


if __name__ == '__main__':
    main()
