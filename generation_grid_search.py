"""
Grid Search Script for LLM Generation Hyperparameters
Supports temperature, top-p, top-k tuning with wandb logging
"""

import argparse
import json
import yaml
import pandas as pd
import random
import os
from pathlib import Path
from ollama import chat
from pydantic import BaseModel
import itertools


class GeneratedText(BaseModel):
    """Schema for generated text"""
    Text: str


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(train_file, max_samples_per_class=30):
    """Load and prepare training data"""
    df = pd.read_csv(train_file, delimiter=';', quotechar='"')
    
    emotions = df['Polarity'].unique()
    emotion_data = {}
    
    for emotion in emotions:
        filtered = df[df['Polarity'] == emotion]
        if not filtered.empty:
            limited = filtered.sample(
                n=min(max_samples_per_class, len(filtered)), 
                random_state=42
            )
            emotion_data[emotion] = limited
            print(f"✅ Loaded {len(limited)} samples for {emotion}")
    
    return emotion_data


def generate_with_config(model, system_prompt, user_prompt, hyperparams):
    """Generate text with specific hyperparameters"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = chat(
            model=model,
            messages=messages,
            format=GeneratedText.model_json_schema(),
            options={
                "temperature": hyperparams.get("temperature", 0.8),
                "num_predict": hyperparams.get("num_predict", 500),
                "top_p": hyperparams.get("top_p", 0.9),
                "top_k": hyperparams.get("top_k", 40),
                "repeat_penalty": hyperparams.get("repeat_penalty", 1.1),
            }
        )
        
        raw = response.message.content
        result = GeneratedText.model_validate_json(raw)
        return result.Text, None
    
    except Exception as e:
        return None, str(e)


def run_grid_search(config, args):
    """Run grid search over hyperparameters"""
    
    # Setup output directory
    output_dir = Path(config['grid_search']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load prompts
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)
        messages = prompts_data.get('messages', [])
        system_prompt = messages[0]['content'] if len(messages) > 0 else ""
        user_prompt_template = messages[-1]['content'] if len(messages) > 1 else ""
    
    # Load training data
    emotion_data = load_data(args.train_file, config['generation']['few_shot']['max_samples_per_class'])
    
    # Define hyperparameter grid
    if args.param_name == 'temperature':
        param_values = args.param_values or config['generation']['temperature_grid']
        fixed_params = {
            'top_p': config['generation']['default']['top_p'],
            'top_k': config['generation']['default']['top_k'],
            'repeat_penalty': config['generation']['default']['repeat_penalty'],
            'num_predict': config['generation']['default']['num_predict']
        }
    elif args.param_name == 'top_p':
        param_values = args.param_values or config['generation']['top_p_grid']
        fixed_params = {
            'temperature': config['generation']['default']['temperature'],
            'top_k': config['generation']['default']['top_k'],
            'repeat_penalty': config['generation']['default']['repeat_penalty'],
            'num_predict': config['generation']['default']['num_predict']
        }
    elif args.param_name == 'top_k':
        param_values = args.param_values or config['generation']['top_k_grid']
        fixed_params = {
            'temperature': config['generation']['default']['temperature'],
            'top_p': config['generation']['default']['top_p'],
            'repeat_penalty': config['generation']['default']['repeat_penalty'],
            'num_predict': config['generation']['default']['num_predict']
        }
    else:
        raise ValueError(f"Unknown parameter: {args.param_name}")
    
    print("=" * 80)
    print(f"Grid Search: {args.param_name}")
    print(f"Values: {param_values}")
    print(f"Fixed params: {fixed_params}")
    print("=" * 80)
    
    all_results = []
    
    # Iterate over parameter values
    for param_value in param_values:
        print(f"\n{'='*80}")
        print(f"Testing {args.param_name} = {param_value}")
        print(f"{'='*80}")
        
        hyperparams = fixed_params.copy()
        hyperparams[args.param_name] = param_value
        
        config_results = {
            'param_name': args.param_name,
            'param_value': param_value,
            'hyperparams': hyperparams,
            'generations': []
        }
        
        # Generate multiple samples with this configuration
        for i in range(args.n_generations):
            print(f"\nGeneration {i+1}/{args.n_generations}")
            
            # Prepare prompt
            target_emotion = random.choice(list(emotion_data.keys()))
            user_prompt = user_prompt_template.replace("{{emotion}}", target_emotion)
            
            # Generate
            generated_text, error = generate_with_config(
                args.model,
                system_prompt,
                user_prompt,
                hyperparams
            )
            
            result = {
                'generation_id': i + 1,
                'target_emotion': target_emotion,
                'generated_text': generated_text,
                'error': error,
                'hyperparams': hyperparams
            }
            
            config_results['generations'].append(result)
            
            if generated_text:
                print(f"✅ Generated: {generated_text[:100]}...")
            else:
                print(f"❌ Error: {error}")
        
        all_results.append(config_results)
        
        # Save intermediate results
        output_file = output_dir / f"grid_search_{args.param_name}_{param_value}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved results to {output_file}")
    
    # Save all results
    final_output = output_dir / f"grid_search_{args.param_name}_complete.json"
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Grid search complete! Results saved to {final_output}")
    
    # Create CSV summary
    summary_data = []
    for config_result in all_results:
        param_value = config_result['param_value']
        successful = sum(1 for g in config_result['generations'] if g['generated_text'])
        failed = sum(1 for g in config_result['generations'] if g['error'])
        
        summary_data.append({
            'param_name': args.param_name,
            'param_value': param_value,
            'successful_generations': successful,
            'failed_generations': failed,
            'success_rate': successful / len(config_result['generations']) if config_result['generations'] else 0
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_csv = output_dir / f"grid_search_{args.param_name}_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"📊 Summary saved to {summary_csv}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Grid Search for LLM Generation Hyperparameters')
    parser.add_argument('--config', type=str, default='configs/experiments_config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--train_file', type=str, default='train_StackOverFlow.csv',
                        help='Path to training data')
    parser.add_argument('--prompts_file', type=str, default='fewShot_generation/prompts.yaml',
                        help='Path to prompts YAML file')
    parser.add_argument('--param_name', type=str, default='temperature',
                        choices=['temperature', 'top_p', 'top_k'],
                        help='Parameter to search')
    parser.add_argument('--param_values', type=float, nargs='+', default=None,
                        help='Values to search (overrides config)')
    parser.add_argument('--n_generations', type=int, default=10,
                        help='Number of generations per config')
    parser.add_argument('--model', type=str, default='llama3.2:1b',
                        help='Model name for Ollama')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run grid search
    results = run_grid_search(config, args)
    
    print("\n✅ Grid search completed successfully!")


if __name__ == '__main__':
    main()
