#!/usr/bin/env python3
"""
Master Experiment Runner

This script creates virtual environments for each experiment and runs them
sequentially. It handles all experiments including traditional ML, deep
learning models, generation experiments, and more.

Requirements:
- Python 3.8+
- Ollama installed and running (for generation experiments)

Usage:
    python run_all_experiments.py [--skip-venv] [--experiments exp1,exp2,...]
"""

import argparse
import subprocess
import sys
import os
import shutil
import requests
from pathlib import Path
from datetime import datetime
import time


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(message):
    """Log info message"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message):
    """Log success message"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message):
    """Log warning message"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message):
    """Log error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


class ExperimentRunner:
    """Main experiment runner class"""
    
    def __init__(self, repo_root, skip_venv=False, selected_experiments=None):
        self.repo_root = Path(repo_root)
        self.venv_base_dir = self.repo_root / '.venvs'
        self.results_dir = self.repo_root / 'experiment_results'
        self.skip_venv = skip_venv
        self.selected_experiments = selected_experiments or []
        self.run_all = len(self.selected_experiments) == 0
        
        # Create directories
        self.venv_base_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def should_run_experiment(self, exp_name):
        """Check if experiment should run"""
        if self.run_all:
            return True
        return exp_name in self.selected_experiments
    
    def create_venv(self, exp_name, requirements_file):
        """Create virtual environment for an experiment"""
        if self.skip_venv:
            log_info(f"Skipping venv creation for {exp_name}")
            return True
        
        venv_path = self.venv_base_dir / exp_name
        
        log_info(f"Creating virtual environment for {exp_name}...")
        
        if venv_path.exists():
            log_warning(f"Virtual environment already exists for {exp_name}, removing...")
            shutil.rmtree(venv_path)
        
        # Create venv
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to create venv for {exp_name}: {e}")
            return False
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / 'Scripts' / 'pip.exe'
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:  # Unix-like
            pip_path = venv_path / 'bin' / 'pip'
            python_path = venv_path / 'bin' / 'python'
        
        # Upgrade pip
        try:
            subprocess.run([str(pip_path), 'install', '--upgrade', 'pip', '-q'], check=True)
        except subprocess.CalledProcessError as e:
            log_warning(f"Failed to upgrade pip for {exp_name}: {e}")
        
        # Install requirements
        if Path(requirements_file).exists():
            log_info(f"Installing dependencies from {requirements_file}...")
            try:
                subprocess.run([str(pip_path), 'install', '-r', requirements_file, '-q'], check=True)
            except subprocess.CalledProcessError as e:
                log_error(f"Failed to install dependencies for {exp_name}: {e}")
                return False
        else:
            log_warning(f"No requirements file found at {requirements_file}")
        
        log_success(f"Virtual environment created for {exp_name}")
        return True
    
    def check_ollama(self):
        """Check if Ollama server is running"""
        log_info("Checking if Ollama server is running...")
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                log_success("Ollama server is running")
                return True
        except:
            pass
        
        log_warning("Ollama server is not running")
        log_warning("Generation experiments require Ollama to be running")
        log_warning("Start Ollama with: ollama serve")
        return False
    
    def run_experiment(self, exp_name, script_path, script_args):
        """Run a single experiment"""
        log_info("=" * 50)
        log_info(f"Running experiment: {exp_name}")
        log_info("=" * 50)
        
        venv_path = self.venv_base_dir / exp_name
        script_path = Path(script_path)
        exp_dir = script_path.parent
        
        # Check venv exists
        if not self.skip_venv and not venv_path.exists():
            log_error(f"Virtual environment not found for {exp_name}")
            return False
        
        # Determine python path
        if self.skip_venv:
            python_path = sys.executable
        else:
            if os.name == 'nt':  # Windows
                python_path = venv_path / 'Scripts' / 'python.exe'
            else:  # Unix-like
                python_path = venv_path / 'bin' / 'python'
        
        # Create results directory for this experiment
        exp_results_dir = self.results_dir / exp_name
        exp_results_dir.mkdir(exist_ok=True)
        
        # Prepare log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = exp_results_dir / f"{exp_name}_{timestamp}.log"
        
        # Build command
        cmd = [str(python_path), script_path.name] + script_args.split()
        
        log_info(f"Running: {' '.join([str(python_path.name), script_path.name] + script_args.split())}")
        log_info(f"Log file: {log_file}")
        
        # Run the experiment
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(exp_dir),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode == 0:
                log_success(f"Experiment {exp_name} completed successfully")
                
                # Move generated results to results directory
                for pattern in ['*_results*.json', '*.csv', 'classification_report*.json']:
                    for file in exp_dir.glob(pattern):
                        try:
                            shutil.move(str(file), str(exp_results_dir / file.name))
                        except:
                            pass
                
                return True
            else:
                log_error(f"Experiment {exp_name} failed. Check log: {log_file}")
                # Print last 20 lines of log
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(line.rstrip())
                return False
        
        except subprocess.TimeoutExpired:
            log_error(f"Experiment {exp_name} timed out after 1 hour")
            return False
        except Exception as e:
            log_error(f"Experiment {exp_name} failed with error: {e}")
            return False
    
    def run_all(self):
        """Run all experiments"""
        log_info("=" * 50)
        log_info("Master Experiment Runner")
        log_info("=" * 50)
        log_info(f"Repository: {self.repo_root}")
        log_info(f"Results directory: {self.results_dir}")
        print()
        
        # Check for dataset files
        train_file = self.repo_root / 'train_StackOverFlow.csv'
        test_file = self.repo_root / 'test_StackOverFlow.csv'
        
        if not train_file.exists():
            log_error(f"Dataset file not found: {train_file}")
            return False
        
        if not test_file.exists():
            log_warning(f"Test file not found: {test_file}")
        
        experiments = [
            # (exp_name, requirements_file, script_path, script_args)
            ('traditional_ml', 
             self.repo_root / 'traditional_ml' / 'requirements.txt',
             self.repo_root / 'traditional_ml' / 'train_traditional_ml.py',
             f'--train_file {train_file} --test_file {test_file} --output_file traditional_ml_results.json'),
            
            ('modernbert',
             self.repo_root / 'modernbert_pipeline' / 'requirements.txt',
             self.repo_root / 'modernbert_pipeline' / 'train_modernbert.py',
             f'--train_file {train_file} --test_file {test_file} --samples_per_class 100 --output_file modernbert_results.json'),
            
            ('zero_shot',
             self.repo_root / 'zero_shot_classifier' / 'requirements.txt',
             self.repo_root / 'zero_shot_classifier' / 'zero_shot_classification.py',
             f'--train_file {train_file} --test_file {test_file} --methods sentence-bert --output_file zero_shot_results.json'),
            
            ('setfit',
             self.repo_root / 'SetFit' / 'requirements.txt',
             self.repo_root / 'SetFit' / 'train_model.py',
             f'-d {train_file} -t {test_file} -n 50'),
            
            ('roberta',
             self.repo_root / 'RoBerta' / 'requirements.txt',
             self.repo_root / 'RoBerta' / 'train_and_predict.py',
             f'-d {train_file} -t {test_file}'),
        ]
        
        # Run non-generation experiments
        for exp_name, req_file, script_path, args in experiments:
            if self.should_run_experiment(exp_name):
                log_info(f"\n=== Experiment: {exp_name.upper()} ===")
                self.create_venv(exp_name, req_file)
                self.run_experiment(exp_name, script_path, args)
        
        # Check Ollama for generation experiments
        ollama_running = False
        if self.should_run_experiment('few_shot_generation') or self.should_run_experiment('zero_shot_generation'):
            ollama_running = self.check_ollama()
            if not ollama_running:
                log_warning("Skipping generation experiments (Ollama not running)")
        
        # Generation experiments
        if ollama_running:
            generation_experiments = [
                ('few_shot_generation',
                 self.repo_root / 'fewShot_generation' / 'requirements_fewShot.txt',
                 self.repo_root / 'fewShot_generation' / 'fewShot_generation.py',
                 '--n_generazioni 10 --temperature 0.8 --model llama3.2:1b'),
                
                ('zero_shot_generation',
                 self.repo_root / 'zeroShot_generation' / 'requirements_zeroShot.txt',
                 self.repo_root / 'zeroShot_generation' / 'zeroShot_generation.py',
                 '--generations 10 --temperature 0.8 --model llama3.2:1b'),
            ]
            
            for exp_name, req_file, script_path, args in generation_experiments:
                if self.should_run_experiment(exp_name):
                    log_info(f"\n=== Experiment: {exp_name.upper()} ===")
                    self.create_venv(exp_name, req_file)
                    self.run_experiment(exp_name, script_path, args)
        
        # Summary
        log_info("\n" + "=" * 50)
        log_info("All experiments completed!")
        log_info("=" * 50)
        log_info(f"Results saved in: {self.results_dir}")
        print()
        log_info("Next steps:")
        log_info(f"1. Review individual experiment logs in {self.results_dir}")
        log_info("2. Compare models using: python compare_models.py")
        print()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments with virtual environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_all_experiments.py
  
  # Run specific experiments
  python run_all_experiments.py --experiments traditional_ml,setfit
  
  # Skip venv creation (use existing environments)
  python run_all_experiments.py --skip-venv
  
Available experiments:
  traditional_ml, modernbert, zero_shot, setfit, roberta
  few_shot_generation, zero_shot_generation
        """
    )
    
    parser.add_argument('--skip-venv', action='store_true',
                        help='Skip virtual environment creation')
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated list of experiments to run (default: all)')
    
    args = parser.parse_args()
    
    # Parse selected experiments
    selected_experiments = []
    if args.experiments:
        selected_experiments = [exp.strip() for exp in args.experiments.split(',')]
    
    # Get repository root
    repo_root = Path(__file__).parent.absolute()
    
    # Create and run experiment runner
    runner = ExperimentRunner(repo_root, args.skip_venv, selected_experiments)
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
