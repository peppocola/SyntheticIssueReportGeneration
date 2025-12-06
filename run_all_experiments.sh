#!/bin/bash

###############################################################################
# Master Experiment Runner
# 
# This script creates virtual environments for each experiment and runs them
# sequentially. It handles all experiments including traditional ML, deep
# learning models, generation experiments, and more.
#
# Requirements:
# - Python 3.8+
# - Ollama installed and running (for generation experiments)
#
# Usage:
#   ./run_all_experiments.sh [--skip-venv] [--experiments "exp1,exp2,..."]
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BASE_DIR="${REPO_ROOT}/.venvs"
RESULTS_DIR="${REPO_ROOT}/experiment_results"
SKIP_VENV=false
RUN_ALL=true
SELECTED_EXPERIMENTS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --experiments)
            RUN_ALL=false
            SELECTED_EXPERIMENTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--skip-venv] [--experiments \"exp1,exp2,...\"]"
            echo ""
            echo "Options:"
            echo "  --skip-venv              Skip virtual environment creation"
            echo "  --experiments \"exp1,...\" Run only specific experiments (comma-separated)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Available experiments:"
            echo "  traditional_ml, modernbert, zero_shot, setfit, roberta"
            echo "  few_shot_generation, zero_shot_generation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if experiment should run
should_run_experiment() {
    local exp_name=$1
    if [ "$RUN_ALL" = true ]; then
        return 0
    fi
    
    if echo "$SELECTED_EXPERIMENTS" | grep -q "$exp_name"; then
        return 0
    fi
    return 1
}

# Create virtual environment for an experiment
create_venv() {
    local exp_name=$1
    local requirements_file=$2
    local venv_path="${VENV_BASE_DIR}/${exp_name}"
    
    if [ "$SKIP_VENV" = true ]; then
        log_info "Skipping venv creation for ${exp_name}"
        return 0
    fi
    
    log_info "Creating virtual environment for ${exp_name}..."
    
    if [ -d "$venv_path" ]; then
        log_warning "Virtual environment already exists for ${exp_name}, removing..."
        rm -rf "$venv_path"
    fi
    
    python3 -m venv "$venv_path"
    
    # Activate and install dependencies
    source "${venv_path}/bin/activate"
    pip install --upgrade pip -q
    
    if [ -f "$requirements_file" ]; then
        log_info "Installing dependencies from ${requirements_file}..."
        pip install -r "$requirements_file" -q
    else
        log_warning "No requirements file found at ${requirements_file}"
    fi
    
    deactivate
    log_success "Virtual environment created for ${exp_name}"
}

# Check if Ollama is running
check_ollama() {
    log_info "Checking if Ollama server is running..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama server is running"
        return 0
    else
        log_warning "Ollama server is not running"
        log_warning "Generation experiments require Ollama to be running"
        log_warning "Start Ollama with: ollama serve"
        return 1
    fi
}

# Run experiment
run_experiment() {
    local exp_name=$1
    local venv_path="${VENV_BASE_DIR}/${exp_name}"
    local script_path=$2
    local script_args=$3
    local exp_dir=$(dirname "$script_path")
    
    log_info "========================================"
    log_info "Running experiment: ${exp_name}"
    log_info "========================================"
    
    # Activate virtual environment
    if [ ! -d "$venv_path" ] && [ "$SKIP_VENV" = false ]; then
        log_error "Virtual environment not found for ${exp_name}"
        return 1
    fi
    
    if [ "$SKIP_VENV" = false ]; then
        source "${venv_path}/bin/activate"
    fi
    
    # Create results directory for this experiment
    local exp_results_dir="${RESULTS_DIR}/${exp_name}"
    mkdir -p "$exp_results_dir"
    
    # Run the experiment
    cd "$exp_dir"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="${exp_results_dir}/${exp_name}_${timestamp}.log"
    
    log_info "Running: python $(basename $script_path) $script_args"
    log_info "Log file: ${log_file}"
    
    if python "$(basename $script_path)" $script_args > "$log_file" 2>&1; then
        log_success "Experiment ${exp_name} completed successfully"
        
        # Move any generated results to the results directory
        if ls *_results*.json 1> /dev/null 2>&1; then
            mv *_results*.json "$exp_results_dir/" 2>/dev/null || true
        fi
        if ls *.csv 1> /dev/null 2>&1; then
            mv *.csv "$exp_results_dir/" 2>/dev/null || true
        fi
        if ls classification_report*.json 1> /dev/null 2>&1; then
            mv classification_report*.json "$exp_results_dir/" 2>/dev/null || true
        fi
    else
        log_error "Experiment ${exp_name} failed. Check log: ${log_file}"
        cat "$log_file" | tail -20
        if [ "$SKIP_VENV" = false ]; then
            deactivate
        fi
        return 1
    fi
    
    if [ "$SKIP_VENV" = false ]; then
        deactivate
    fi
    
    cd "$REPO_ROOT"
}

# Main execution
main() {
    log_info "========================================"
    log_info "Master Experiment Runner"
    log_info "========================================"
    log_info "Repository: ${REPO_ROOT}"
    log_info "Results directory: ${RESULTS_DIR}"
    log_info ""
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$VENV_BASE_DIR"
    
    # Check for dataset files
    if [ ! -f "${REPO_ROOT}/train_StackOverFlow.csv" ]; then
        log_error "Dataset file not found: train_StackOverFlow.csv"
        exit 1
    fi
    
    # Experiment 1: Traditional ML
    if should_run_experiment "traditional_ml"; then
        log_info "\n=== Experiment 1: Traditional ML ==="
        create_venv "traditional_ml" "${REPO_ROOT}/traditional_ml/requirements.txt"
        run_experiment "traditional_ml" \
            "${REPO_ROOT}/traditional_ml/train_traditional_ml.py" \
            "--train_file ${REPO_ROOT}/train_StackOverFlow.csv --test_file ${REPO_ROOT}/test_StackOverFlow.csv --output_file traditional_ml_results.json"
    fi
    
    # Experiment 2: ModernBERT
    if should_run_experiment "modernbert"; then
        log_info "\n=== Experiment 2: ModernBERT Fine-tuning ==="
        create_venv "modernbert" "${REPO_ROOT}/modernbert_pipeline/requirements.txt"
        run_experiment "modernbert" \
            "${REPO_ROOT}/modernbert_pipeline/train_modernbert.py" \
            "--train_file ${REPO_ROOT}/train_StackOverFlow.csv --test_file ${REPO_ROOT}/test_StackOverFlow.csv --samples_per_class 100 --output_file modernbert_results.json"
    fi
    
    # Experiment 3: Zero-shot Classification
    if should_run_experiment "zero_shot"; then
        log_info "\n=== Experiment 3: Zero-shot Classification ==="
        create_venv "zero_shot" "${REPO_ROOT}/zero_shot_classifier/requirements.txt"
        run_experiment "zero_shot" \
            "${REPO_ROOT}/zero_shot_classifier/zero_shot_classification.py" \
            "--train_file ${REPO_ROOT}/train_StackOverFlow.csv --test_file ${REPO_ROOT}/test_StackOverFlow.csv --methods sentence-bert --output_file zero_shot_results.json"
    fi
    
    # Experiment 4: SetFit
    if should_run_experiment "setfit"; then
        log_info "\n=== Experiment 4: SetFit ==="
        create_venv "setfit" "${REPO_ROOT}/SetFit/requirements.txt"
        run_experiment "setfit" \
            "${REPO_ROOT}/SetFit/train_model.py" \
            "-d ${REPO_ROOT}/train_StackOverFlow.csv -t ${REPO_ROOT}/test_StackOverFlow.csv -n 50"
    fi
    
    # Experiment 5: RoBERTa
    if should_run_experiment "roberta"; then
        log_info "\n=== Experiment 5: RoBERTa ==="
        create_venv "roberta" "${REPO_ROOT}/RoBerta/requirements.txt"
        run_experiment "roberta" \
            "${REPO_ROOT}/RoBerta/train_and_predict.py" \
            "-d ${REPO_ROOT}/train_StackOverFlow.csv -t ${REPO_ROOT}/test_StackOverFlow.csv"
    fi
    
    # Check if Ollama is running for generation experiments
    OLLAMA_RUNNING=false
    if should_run_experiment "few_shot_generation" || should_run_experiment "zero_shot_generation"; then
        if check_ollama; then
            OLLAMA_RUNNING=true
        else
            log_warning "Skipping generation experiments (Ollama not running)"
        fi
    fi
    
    # Experiment 6: Few-shot Generation
    if should_run_experiment "few_shot_generation" && [ "$OLLAMA_RUNNING" = true ]; then
        log_info "\n=== Experiment 6: Few-shot Generation ==="
        create_venv "few_shot_generation" "${REPO_ROOT}/fewShot_generation/requirements_fewShot.txt"
        run_experiment "few_shot_generation" \
            "${REPO_ROOT}/fewShot_generation/fewShot_generation.py" \
            "--n_generazioni 10 --temperature 0.8 --model llama3.2:1b"
    fi
    
    # Experiment 7: Zero-shot Generation
    if should_run_experiment "zero_shot_generation" && [ "$OLLAMA_RUNNING" = true ]; then
        log_info "\n=== Experiment 7: Zero-shot Generation ==="
        create_venv "zero_shot_generation" "${REPO_ROOT}/zeroShot_generation/requirements_zeroShot.txt"
        run_experiment "zero_shot_generation" \
            "${REPO_ROOT}/zeroShot_generation/zeroShot_generation.py" \
            "--generations 10 --temperature 0.8 --model llama3.2:1b"
    fi
    
    # Summary
    log_info "\n========================================"
    log_info "All experiments completed!"
    log_info "========================================"
    log_info "Results saved in: ${RESULTS_DIR}"
    log_info ""
    log_info "Next steps:"
    log_info "1. Review individual experiment logs in ${RESULTS_DIR}"
    log_info "2. Compare models using: python compare_models.py"
    log_info ""
}

# Run main
main
