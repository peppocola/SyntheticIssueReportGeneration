"""
Dataset Import and Preprocessing Pipeline
Supports: GitHub emotion dataset, NASA bug/feature dataset
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests
import zipfile
import io
import yaml


def download_github_emotion_dataset(output_dir='datasets'):
    """
    Download GitHub emotion dataset
    Note: This is a placeholder. Replace with actual dataset URL/source
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("GitHub Emotion Dataset Import")
    print("=" * 80)
    
    # Placeholder: In real implementation, download from actual source
    # For this implementation, we'll create a sample structure
    print("\n⚠️  Note: Using sample dataset structure")
    print("To use real data, replace this with actual dataset download")
    
    # Sample data structure - replace with real data loading
    sample_data = {
        'text': [
            'I love this feature!',
            'This is so frustrating',
            'Great work on this',
            'Having issues with this',
            'Perfect solution',
            'Not working as expected',
            'Amazing job!',
            'This makes me angry',
            'I am so sad about this',
            'What a surprise!',
            'I feel happy about this',
            'This is terrible',
            'Feeling so sad',
            'Another surprise here',
            'More happiness',
            'Very angry now'
        ],
        'emotion': ['joy', 'anger', 'joy', 'sadness', 'joy', 'anger', 
                   'joy', 'anger', 'sadness', 'surprise', 'joy', 'anger',
                   'sadness', 'surprise', 'joy', 'anger']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample
    sample_file = output_path / 'github_emotion_sample.csv'
    df.to_csv(sample_file, index=False)
    print(f"\n✅ Sample data saved to {sample_file}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Rows: {len(df)}")
    
    return df


def preprocess_github_emotion(input_file, output_dir='datasets', test_size=0.3, random_state=42):
    """
    Preprocess GitHub emotion dataset and create train/test split
    """
    print("\n" + "=" * 80)
    print("Preprocessing GitHub Emotion Dataset")
    print("=" * 80)
    
    # Load data
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        df = pd.read_csv(input_file)
    
    print(f"\n📂 Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    text_col = 'text' if 'text' in df.columns else 'Text'
    label_col = 'emotion' if 'emotion' in df.columns else 'label'
    
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Clean data
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    
    print(f"\n✅ After cleaning: {len(df)} samples")
    print(f"\n📊 Label distribution:")
    print(df[label_col].value_counts())
    
    # Add ID column
    df['ID'] = [f'gh_{i}' for i in range(len(df))]
    
    # Standardize column names
    df_standard = df.rename(columns={text_col: 'Text', label_col: 'Polarity'})
    df_standard = df_standard[['ID', 'Text', 'Polarity']]
    
    # Train/test split with stratification
    df_train, df_test = train_test_split(
        df_standard,
        test_size=test_size,
        stratify=df_standard['Polarity'],
        random_state=random_state
    )
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(df_train)} samples")
    print(f"   Test: {len(df_test)} samples")
    
    # Save processed datasets
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    train_file = output_path / 'github_emotion_train.csv'
    test_file = output_path / 'github_emotion_test.csv'
    
    df_train.to_csv(train_file, index=False, sep=';', quotechar='"')
    df_test.to_csv(test_file, index=False, sep=';', quotechar='"')
    
    print(f"\n✅ Train data saved to: {train_file}")
    print(f"✅ Test data saved to: {test_file}")
    
    return df_train, df_test


def preprocess_nasa_dataset(input_file, output_dir='datasets', test_size=0.3, random_state=42):
    """
    Preprocess NASA bug/feature dataset and create train/test split
    """
    print("\n" + "=" * 80)
    print("Preprocessing NASA Bug/Feature Dataset")
    print("=" * 80)
    
    # Load data
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        df = pd.read_csv(input_file)
    
    print(f"\n📂 Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    text_col = next((col for col in ['description', 'text', 'Text'] if col in df.columns), None)
    label_col = next((col for col in ['type', 'label', 'Polarity'] if col in df.columns), None)
    
    if not text_col or not label_col:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Clean data
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    
    print(f"\n✅ After cleaning: {len(df)} samples")
    print(f"\n📊 Label distribution:")
    print(df[label_col].value_counts())
    
    # Add ID column
    df['ID'] = [f'nasa_{i}' for i in range(len(df))]
    
    # Standardize column names
    df_standard = df.rename(columns={text_col: 'Text', label_col: 'Polarity'})
    df_standard = df_standard[['ID', 'Text', 'Polarity']]
    
    # Train/test split with stratification
    df_train, df_test = train_test_split(
        df_standard,
        test_size=test_size,
        stratify=df_standard['Polarity'],
        random_state=random_state
    )
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(df_train)} samples")
    print(f"   Test: {len(df_test)} samples")
    
    # Save processed datasets
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    train_file = output_path / 'nasa_bugs_train.csv'
    test_file = output_path / 'nasa_bugs_test.csv'
    
    df_train.to_csv(train_file, index=False, sep=';', quotechar='"')
    df_test.to_csv(test_file, index=False, sep=';', quotechar='"')
    
    print(f"\n✅ Train data saved to: {train_file}")
    print(f"✅ Test data saved to: {test_file}")
    
    return df_train, df_test


def main():
    parser = argparse.ArgumentParser(description='Dataset Import and Preprocessing')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['github_emotion', 'nasa_bugs'],
                        help='Dataset to import')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file path (if already downloaded)')
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test split ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset (if not already present)')
    
    args = parser.parse_args()
    
    if args.dataset == 'github_emotion':
        if args.download or not args.input_file:
            df = download_github_emotion_dataset(args.output_dir)
            preprocess_github_emotion(df, args.output_dir, args.test_size, args.random_state)
        else:
            preprocess_github_emotion(args.input_file, args.output_dir, args.test_size, args.random_state)
    
    elif args.dataset == 'nasa_bugs':
        if not args.input_file:
            print("❌ Error: NASA dataset requires --input_file parameter")
            return
        preprocess_nasa_dataset(args.input_file, args.output_dir, args.test_size, args.random_state)
    
    print("\n✅ Dataset preprocessing completed!")


if __name__ == '__main__':
    main()
