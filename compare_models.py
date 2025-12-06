"""
Model Comparison Script
Aggregates results from all models and creates comparison tables
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys


def load_traditional_ml_results(file_path):
    """Load traditional ML results"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for result in data.get('results', []):
        if 'error' not in result:
            results.append({
                'Model Type': 'Traditional ML',
                'Model': f"{result['model']} ({result['feature_type']}, {result['ngram_range'][0]}-{result['ngram_range'][1]})",
                'Accuracy': result['metrics']['accuracy'],
                'F1': result['metrics']['f1'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'Train Time (s)': result['metrics']['train_time']
            })
    
    return results


def load_modernbert_results(file_path):
    """Load ModernBERT results"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return [{
        'Model Type': 'Deep Learning',
        'Model': f"ModernBERT ({data.get('samples_per_class', 'all')} samples/class)",
        'Accuracy': data['metrics']['accuracy'],
        'F1': data['metrics']['f1'],
        'Precision': data['metrics']['precision'],
        'Recall': data['metrics']['recall'],
        'Train Time (s)': data['train_time']
    }]


def load_zero_shot_results(file_path):
    """Load zero-shot results"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for result in data.get('results', []):
        results.append({
            'Model Type': 'Zero-shot',
            'Model': result['model'],
            'Accuracy': result['accuracy'],
            'F1': result['f1'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'Train Time (s)': 0.0,  # Zero-shot doesn't train
            'Inference Time (s)': result.get('inference_time', 0)
        })
    
    return results


def load_setfit_results(file_path):
    """Load SetFit results from classification report"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return [{
        'Model Type': 'Few-shot (SetFit)',
        'Model': 'SetFit (all-mpnet-base-v2)',
        'Accuracy': data['accuracy'],
        'F1': data['weighted avg']['f1-score'],
        'Precision': data['weighted avg']['precision'],
        'Recall': data['weighted avg']['recall'],
        'Train Time (s)': None  # Not recorded in classification report
    }]


def create_comparison_table(results_files, output_file='model_comparison.csv'):
    """Create comparison table from all results"""
    all_results = []
    
    for file_info in results_files:
        file_path = Path(file_info['path'])
        if not file_path.exists():
            print(f"⚠️  Warning: {file_path} not found, skipping...")
            continue
        
        print(f"📂 Loading {file_info['type']} results from {file_path}")
        
        try:
            if file_info['type'] == 'traditional_ml':
                results = load_traditional_ml_results(file_path)
            elif file_info['type'] == 'modernbert':
                results = load_modernbert_results(file_path)
            elif file_info['type'] == 'zero_shot':
                results = load_zero_shot_results(file_path)
            elif file_info['type'] == 'setfit':
                results = load_setfit_results(file_path)
            else:
                print(f"❌ Unknown result type: {file_info['type']}")
                continue
            
            all_results.extend(results)
            print(f"✅ Loaded {len(results)} result(s)")
        
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    if not all_results:
        print("❌ No results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by F1 score
    df = df.sort_values('F1', ascending=False)
    
    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n✅ Comparison table saved to {output_file}")
    
    return df


def print_comparison_table(df):
    """Print formatted comparison table"""
    print("\n" + "=" * 120)
    print("📊 MODEL COMPARISON TABLE")
    print("=" * 120)
    print(f"{'Model Type':<20} {'Model':<40} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Time(s)':<10}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        time_str = f"{row['Train Time (s)']:.2f}" if pd.notna(row['Train Time (s)']) else "N/A"
        print(f"{row['Model Type']:<20} "
              f"{row['Model']:<40} "
              f"{row['Accuracy']:<10.4f} "
              f"{row['F1']:<10.4f} "
              f"{row['Precision']:<10.4f} "
              f"{row['Recall']:<10.4f} "
              f"{time_str:<10}")
    
    print("=" * 120)
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   Best Accuracy: {df['Accuracy'].max():.4f} ({df.loc[df['Accuracy'].idxmax(), 'Model']})")
    print(f"   Best F1: {df['F1'].max():.4f} ({df.loc[df['F1'].idxmax(), 'Model']})")
    print(f"   Average Accuracy: {df['Accuracy'].mean():.4f}")
    print(f"   Average F1: {df['F1'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compare results from different models')
    parser.add_argument('--traditional_ml', type=str, default=None,
                        help='Path to traditional ML results JSON')
    parser.add_argument('--modernbert', type=str, default=None,
                        help='Path to ModernBERT results JSON')
    parser.add_argument('--zero_shot', type=str, default=None,
                        help='Path to zero-shot results JSON')
    parser.add_argument('--setfit', type=str, default=None,
                        help='Path to SetFit classification report JSON')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    # Collect all specified result files
    results_files = []
    
    if args.traditional_ml:
        results_files.append({'type': 'traditional_ml', 'path': args.traditional_ml})
    if args.modernbert:
        results_files.append({'type': 'modernbert', 'path': args.modernbert})
    if args.zero_shot:
        results_files.append({'type': 'zero_shot', 'path': args.zero_shot})
    if args.setfit:
        results_files.append({'type': 'setfit', 'path': args.setfit})
    
    if not results_files:
        print("❌ Error: No result files specified!")
        print("Usage example:")
        print("  python compare_models.py --traditional_ml results_traditional.json --modernbert results_modernbert.json")
        sys.exit(1)
    
    print("=" * 80)
    print("Model Comparison Script")
    print("=" * 80)
    
    # Create comparison table
    df = create_comparison_table(results_files, args.output)
    
    if df is not None:
        # Print comparison table
        print_comparison_table(df)
        print(f"\n✅ Comparison completed successfully!")
    else:
        print("❌ Failed to create comparison table")
        sys.exit(1)


if __name__ == '__main__':
    main()
