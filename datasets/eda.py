"""
Exploratory Data Analysis (EDA) Script
Analyzes dataset characteristics and generates statistics
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_dataset(file_path):
    """Load dataset from CSV"""
    try:
        # Try with semicolon delimiter first
        df = pd.read_csv(file_path, delimiter=';', quotechar='"')
    except:
        # Fallback to comma delimiter
        df = pd.read_csv(file_path)
    
    return df


def analyze_text_length(df, text_column='Text'):
    """Analyze text length statistics"""
    df['text_length'] = df[text_column].astype(str).str.len()
    df['word_count'] = df[text_column].astype(str).str.split().str.len()
    
    stats = {
        'text_length': {
            'mean': float(df['text_length'].mean()),
            'median': float(df['text_length'].median()),
            'min': int(df['text_length'].min()),
            'max': int(df['text_length'].max()),
            'std': float(df['text_length'].std())
        },
        'word_count': {
            'mean': float(df['word_count'].mean()),
            'median': float(df['word_count'].median()),
            'min': int(df['word_count'].min()),
            'max': int(df['word_count'].max()),
            'std': float(df['word_count'].std())
        }
    }
    
    return stats, df


def analyze_label_distribution(df, label_column='Polarity'):
    """Analyze label distribution"""
    label_counts = df[label_column].value_counts()
    total = len(df)
    
    distribution = {}
    for label, count in label_counts.items():
        distribution[str(label)] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    
    return distribution


def generate_visualizations(df, output_dir, text_column='Text', label_column='Polarity'):
    """Generate EDA visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Label distribution
    plt.figure(figsize=(10, 6))
    df[label_column].value_counts().plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path / 'label_distribution.png')
    plt.close()
    
    # 2. Text length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['text_length'], bins=50, edgecolor='black')
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_distribution.png')
    plt.close()
    
    # 3. Word count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['word_count'], bins=50, edgecolor='black')
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path / 'word_count_distribution.png')
    plt.close()
    
    # 4. Text length by label
    plt.figure(figsize=(12, 6))
    df.boxplot(column='text_length', by=label_column)
    plt.title('Text Length by Label')
    plt.suptitle('')
    plt.xlabel('Label')
    plt.ylabel('Text Length (characters)')
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_by_label.png')
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_path}/")


def perform_eda(file_path, output_dir='eda_results', text_column='Text', label_column='Polarity'):
    """Perform complete EDA"""
    print("=" * 80)
    print("Exploratory Data Analysis (EDA)")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {file_path}")
    df = load_dataset(file_path)
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Check for missing values
    print("\n📊 Missing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values")
    else:
        print(missing[missing > 0])
    
    # Text length analysis
    print("\n" + "=" * 80)
    print("Text Length Analysis")
    print("=" * 80)
    text_stats, df = analyze_text_length(df, text_column)
    
    print("\n📏 Text Length Statistics:")
    print(f"   Mean: {text_stats['text_length']['mean']:.2f} characters")
    print(f"   Median: {text_stats['text_length']['median']:.2f} characters")
    print(f"   Min: {text_stats['text_length']['min']} characters")
    print(f"   Max: {text_stats['text_length']['max']} characters")
    print(f"   Std Dev: {text_stats['text_length']['std']:.2f} characters")
    
    print("\n📝 Word Count Statistics:")
    print(f"   Mean: {text_stats['word_count']['mean']:.2f} words")
    print(f"   Median: {text_stats['word_count']['median']:.2f} words")
    print(f"   Min: {text_stats['word_count']['min']} words")
    print(f"   Max: {text_stats['word_count']['max']} words")
    print(f"   Std Dev: {text_stats['word_count']['std']:.2f} words")
    
    # Label distribution
    print("\n" + "=" * 80)
    print("Label Distribution")
    print("=" * 80)
    label_dist = analyze_label_distribution(df, label_column)
    
    for label, stats in label_dist.items():
        print(f"   {label}: {stats['count']} samples ({stats['percentage']:.2f}%)")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    generate_visualizations(df, output_dir, text_column, label_column)
    
    # Save statistics to JSON
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    report = {
        'file': str(file_path),
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': missing.to_dict(),
        'text_statistics': text_stats,
        'label_distribution': label_dist
    }
    
    report_file = output_path / 'eda_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ EDA report saved to {report_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='eda_results',
                        help='Output directory for EDA results')
    parser.add_argument('--text_column', type=str, default='Text',
                        help='Name of text column')
    parser.add_argument('--label_column', type=str, default='Polarity',
                        help='Name of label column')
    
    args = parser.parse_args()
    
    perform_eda(args.file, args.output_dir, args.text_column, args.label_column)
    
    print("\n✅ EDA completed successfully!")


if __name__ == '__main__':
    main()
