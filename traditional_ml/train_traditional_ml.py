"""
Traditional ML Pipeline for Text Classification
Supports: SVM, Random Forest, Logistic Regression, Naive Bayes
Features: Bag-of-words, TF-IDF, n-grams
"""

import argparse
import pandas as pd
import numpy as np
import time
import yaml
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


def load_data(train_file, test_file=None, split_ratio=0.3):
    """Load and split data"""
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
    
    return df_train, df_test


def extract_features(X_train, X_test, feature_type='tfidf', ngram_range=(1, 1), max_features=5000):
    """Extract features using BoW, TF-IDF, or n-grams"""
    if feature_type == 'bow':
        vectorizer = CountVectorizer(
            ngram_range=ngram_range, 
            max_features=max_features,
            min_df=2,
            max_df=0.95
        )
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, 
            max_features=max_features,
            min_df=2,
            max_df=0.95
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    return X_train_features, X_test_features, vectorizer


def get_model(model_name, random_state=42):
    """Get model instance"""
    models = {
        'svm': SVC(kernel='linear', random_state=random_state, probability=True),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
        'naive_bayes': MultinomialNB()
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train model and compute metrics"""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'train_time': float(train_time)
    }
    
    return model, metrics, y_pred


def main():
    parser = argparse.ArgumentParser(description='Traditional ML Pipeline for Text Classification')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test CSV file')
    parser.add_argument('--split_ratio', type=float, default=0.3, help='Test split ratio if no test file provided')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['svm', 'random_forest', 'logistic_regression', 'naive_bayes'],
                        help='Models to train')
    parser.add_argument('--feature_types', type=str, nargs='+', 
                        default=['bow', 'tfidf'],
                        help='Feature extraction types')
    parser.add_argument('--ngram_ranges', type=str, nargs='+',
                        default=['1,1', '1,2', '1,3'],
                        help='N-gram ranges (format: min,max)')
    parser.add_argument('--max_features', type=int, default=5000, help='Maximum number of features')
    parser.add_argument('--output_file', type=str, default='traditional_ml_results.json',
                        help='Output file for results')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Override args with config
            for key, value in config.get('traditional_ml', {}).items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    print("=" * 80)
    print("Traditional ML Pipeline for Text Classification")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    df_train, df_test = load_data(args.train_file, args.test_file, args.split_ratio)
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Prepare labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['Polarity'])
    y_test = label_encoder.transform(df_test['Polarity'])
    
    X_train = df_train['Text'].astype(str).fillna('')
    X_test = df_test['Text'].astype(str).fillna('')
    
    # Results storage
    all_results = []
    
    # Iterate through all combinations
    for feature_type in args.feature_types:
        for ngram_str in args.ngram_ranges:
            ngram_range = tuple(map(int, ngram_str.split(',')))
            
            print(f"\n🔧 Extracting features: {feature_type}, n-grams: {ngram_range}")
            X_train_features, X_test_features, vectorizer = extract_features(
                X_train, X_test, feature_type, ngram_range, args.max_features
            )
            print(f"Feature shape: {X_train_features.shape}")
            
            for model_name in args.models:
                print(f"\n🤖 Training model: {model_name}")
                
                try:
                    model = get_model(model_name)
                    trained_model, metrics, y_pred = train_and_evaluate(
                        model, X_train_features, y_train, X_test_features, y_test
                    )
                    
                    result = {
                        'model': model_name,
                        'feature_type': feature_type,
                        'ngram_range': list(ngram_range),
                        'max_features': args.max_features,
                        'metrics': metrics
                    }
                    
                    all_results.append(result)
                    
                    print(f"✅ Accuracy: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1']:.4f}, "
                          f"Train time: {metrics['train_time']:.2f}s")
                
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}")
                    all_results.append({
                        'model': model_name,
                        'feature_type': feature_type,
                        'ngram_range': list(ngram_range),
                        'error': str(e)
                    })
    
    # Save results
    output_data = {
        'dataset': args.train_file,
        'train_size': len(df_train),
        'test_size': len(df_test),
        'results': all_results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Feature':<10} {'N-gram':<10} {'Accuracy':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for result in all_results:
        if 'error' not in result:
            ngram = f"{result['ngram_range'][0]}-{result['ngram_range'][1]}"
            print(f"{result['model']:<20} "
                  f"{result['feature_type']:<10} "
                  f"{ngram:<10} "
                  f"{result['metrics']['accuracy']:<10.4f} "
                  f"{result['metrics']['f1']:<10.4f} "
                  f"{result['metrics']['train_time']:<10.2f}")


if __name__ == '__main__':
    main()
