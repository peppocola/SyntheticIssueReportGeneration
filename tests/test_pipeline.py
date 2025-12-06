"""
Unit tests for data loading and feature engineering
Tests are designed to run on small subsets without GPU
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoading(unittest.TestCase):
    """Test data loading functions"""
    
    def setUp(self):
        """Create test data"""
        self.test_data = pd.DataFrame({
            'ID': ['t1', 't2', 't3', 't4', 't5', 't6'],
            'Text': [
                'This is positive',
                'This is negative',
                'This is neutral',
                'Another positive',
                'Another negative',
                'Another neutral'
            ],
            'Polarity': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.test_data.to_csv(self.temp_file.name, index=False, sep=';', quotechar='"')
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary file"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_csv(self):
        """Test loading CSV file"""
        df = pd.read_csv(self.temp_file.name, delimiter=';', quotechar='"')
        self.assertEqual(len(df), 6)
        self.assertIn('Text', df.columns)
        self.assertIn('Polarity', df.columns)
    
    def test_data_stratification(self):
        """Test stratified split maintains label distribution"""
        from sklearn.model_selection import train_test_split
        
        # Create larger test data for stratification
        test_data_large = pd.DataFrame({
            'ID': [f't{i}' for i in range(30)],
            'Text': [f'Text {i}' for i in range(30)],
            'Polarity': ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
        })
        
        df_train, df_test = train_test_split(
            test_data_large,
            test_size=0.3,
            stratify=test_data_large['Polarity'],
            random_state=42
        )
        
        # Check that all classes are represented
        train_classes = set(df_train['Polarity'].unique())
        test_classes = set(df_test['Polarity'].unique())
        all_classes = set(test_data_large['Polarity'].unique())
        
        # All classes should be in both splits
        self.assertEqual(train_classes, all_classes)
        self.assertEqual(test_classes, all_classes)
    
    def test_missing_values(self):
        """Test handling missing values"""
        df_with_na = self.test_data.copy()
        df_with_na.loc[0, 'Text'] = np.nan
        
        # Clean data
        df_clean = df_with_na.dropna(subset=['Text'])
        
        self.assertEqual(len(df_clean), 5)
        self.assertFalse(df_clean['Text'].isnull().any())


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions"""
    
    def setUp(self):
        """Create test data"""
        self.texts = [
            'This is a test',
            'Another test here',
            'Testing features'
        ]
    
    def test_bow_features(self):
        """Test Bag-of-Words feature extraction"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=100)
        X = vectorizer.fit_transform(self.texts)
        
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)
        self.assertLessEqual(X.shape[1], 100)
    
    def test_tfidf_features(self):
        """Test TF-IDF feature extraction"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=100)
        X = vectorizer.fit_transform(self.texts)
        
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)
        self.assertLessEqual(X.shape[1], 100)
    
    def test_ngram_features(self):
        """Test n-gram feature extraction"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Bigrams
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
        X = vectorizer.fit_transform(self.texts)
        
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)
    
    def test_label_encoding(self):
        """Test label encoding"""
        from sklearn.preprocessing import LabelEncoder
        
        labels = ['positive', 'negative', 'neutral', 'positive']
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        
        self.assertEqual(len(encoded), 4)
        self.assertEqual(len(encoder.classes_), 3)
        
        # Test inverse transform
        decoded = encoder.inverse_transform(encoded)
        self.assertEqual(list(decoded), labels)


class TestMLModels(unittest.TestCase):
    """Test ML model training (small scale)"""
    
    def setUp(self):
        """Create test data"""
        self.X_train = np.random.rand(20, 10)
        self.y_train = np.random.choice([0, 1, 2], size=20)
        self.X_test = np.random.rand(10, 10)
        self.y_test = np.random.choice([0, 1, 2], size=10)
    
    def test_logistic_regression(self):
        """Test Logistic Regression training"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_naive_bayes(self):
        """Test Naive Bayes training"""
        from sklearn.naive_bayes import MultinomialNB
        
        # Ensure non-negative features
        X_train_pos = np.abs(self.X_train)
        X_test_pos = np.abs(self.X_test)
        
        model = MultinomialNB()
        model.fit(X_train_pos, self.y_train)
        
        predictions = model.predict(X_test_pos)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_random_forest(self):
        """Test Random Forest training"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_svm(self):
        """Test SVM training"""
        from sklearn.svm import SVC
        
        model = SVC(kernel='linear', random_state=42)
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))


class TestMetrics(unittest.TestCase):
    """Test metrics computation"""
    
    def test_accuracy(self):
        """Test accuracy computation"""
        from sklearn.metrics import accuracy_score
        
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]
        
        accuracy = accuracy_score(y_true, y_pred)
        self.assertGreater(accuracy, 0.5)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_f1_score(self):
        """Test F1 score computation"""
        from sklearn.metrics import f1_score
        
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
    
    def test_precision_recall(self):
        """Test precision and recall computation"""
        from sklearn.metrics import precision_score, recall_score
        
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading"""
    
    def test_yaml_loading(self):
        """Test loading YAML config"""
        import yaml
        
        config_str = """
        traditional_ml:
          models: ["svm", "logistic_regression"]
          max_features: 5000
        generation:
          default:
            temperature: 0.8
            top_p: 0.9
        """
        
        config = yaml.safe_load(config_str)
        
        self.assertIn('traditional_ml', config)
        self.assertIn('generation', config)
        self.assertEqual(config['traditional_ml']['max_features'], 5000)
        self.assertEqual(config['generation']['default']['temperature'], 0.8)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestMLModels))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
