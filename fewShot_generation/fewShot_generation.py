import argparse
import json
import yaml
import pandas as pd
import random
from ollama import chat
from pydantic import BaseModel

parser = argparse.ArgumentParser(description="Few-shot generation with Ollama")
parser.add_argument("--n_samples", type=int, default=3, help="Number of examples to use for few-shot context PER EMOTION")
parser.add_argument("--n_generazioni", type=int, default=50, help="Number of generations to perform")
parser.add_argument("--target_polarity", type=str, default="positive", choices=["positive", "neutral", "negative"], help="Target emotion for the generated text")
parser.add_argument("--model", type=str, default="llama3.2:1b", help="Model name to use with Ollama")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--num_predict", type=int, default=500, help="Maximum number of tokens to predict")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
parser.add_argument("--repeat_penalty", type=float, default=1.1, help="Penalty for repeated tokens")
parser.add_argument("--max_samples_per_class", type=int, default=30, help="Maximum number of samples to use per emotion class")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (overrides other args)")
args = parser.parse_args()

# Load config file if provided
if args.config:
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        # Override args with config values
        gen_config = config.get('generation', {})
        few_shot_config = gen_config.get('few_shot', {})
        default_config = gen_config.get('default', {})
        
        if 'n_samples' in few_shot_config:
            args.n_samples = few_shot_config['n_samples']
        if 'n_generazioni' in few_shot_config:
            args.n_generazioni = few_shot_config['n_generazioni']
        if 'max_samples_per_class' in few_shot_config:
            args.max_samples_per_class = few_shot_config['max_samples_per_class']
        if 'temperature' in default_config:
            args.temperature = default_config['temperature']
        if 'top_p' in default_config:
            args.top_p = default_config['top_p']
        if 'top_k' in default_config:
            args.top_k = default_config['top_k']
        if 'repeat_penalty' in default_config:
            args.repeat_penalty = default_config['repeat_penalty']
        if 'num_predict' in default_config:
            args.num_predict = default_config['num_predict']

initial_seed = 42
random.seed(initial_seed)

with open("prompts.yaml", "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)
    base_messages = yaml_data.get("messages", [])

system_prompt = base_messages[0]["content"] if len(base_messages) > 0 else ""
user_prompt_base = base_messages[-1]["content"] if len(base_messages) > 1 else ""

user_prompt_base = user_prompt_base.replace("{{emotion}}", args.target_polarity)

df = pd.read_csv("train_StackOverFlow.csv", delimiter=';', quotechar='"')

emotions = ["positive", "negative", "neutral"]
emotion_data = {}

MAX_SAMPLES_PER_CLASS = args.max_samples_per_class

for emotion in emotions:
    filtered = df[df['Polarity'] == emotion]
    if filtered.empty:
        print(f"⚠️ Attenzione: Nessuna riga con polarità '{emotion}'")
        emotion_data[emotion] = pd.DataFrame()
    else:
        limited = filtered.sample(n=min(MAX_SAMPLES_PER_CLASS, len(filtered)), random_state=initial_seed)
        emotion_data[emotion] = limited
        print(f"✅ Selezionate {len(limited)} righe (su {len(filtered)}) per polarità '{emotion}'")

if all(data.empty for data in emotion_data.values()):
    raise ValueError("Nessun dato disponibile per alcuna emotion")

generated_seeds = random.sample(range(1_000_000), args.n_generazioni)

class User(BaseModel):
    Text: str

all_results = []
all_prompts = []

for i, seed in enumerate(generated_seeds, start=1):
    print(f"\n--- Generazione {i} (seed: {seed}) ---")
    random.seed(seed)
    
    all_samples = []
    sample_info = []
    
    for emotion in emotions:
        if not emotion_data[emotion].empty:
            available_samples = len(emotion_data[emotion])
            samples_to_take = min(args.n_samples, available_samples)
            
            if samples_to_take > 0:
                sampled = emotion_data[emotion].sample(n=samples_to_take, random_state=seed)
                texts = sampled["Text"].astype(str).tolist()
                
                for text in texts:
                    all_samples.append({"text": text, "emotion": emotion})
                
                sample_info.append(f"{emotion}: {samples_to_take} samples")
                print(f"  📊 {emotion}: {samples_to_take}/{available_samples} samples")
            else:
                print(f"  ❌ {emotion}: nessun sample disponibile")
        else:
            print(f"  ❌ {emotion}: nessun dato disponibile")
    
    if not all_samples:
        print(f"⚠️ Nessun sample disponibile per la generazione {i}")
        continue
    
    random.shuffle(all_samples)
    
    user_prompt_complete = user_prompt_base + "\n\nExamples:\n"
    for j, sample in enumerate(all_samples, start=1):
        user_prompt_complete += f"Example {j} (emotion: {sample['emotion']}): {sample['text']}\n"
    
    user_prompt_complete += f"\nNow generate a new text with {args.target_polarity} emotion, following the pattern of the examples above."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_complete}
    ]

    all_prompts.append({
        "generation": i,
        "seed": seed,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt_complete,
        "full_messages": messages
    })

    try:
        res = chat(
            model=args.model,
            format=User.model_json_schema(),
            messages=messages,
            options={
                "temperature": args.temperature,
                "num_predict": args.num_predict,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repeat_penalty": args.repeat_penalty,
            }
        )

        raw = res.message.content
        try:
            user = User.model_validate_json(raw)
            print("Parsed:", user)
            all_results.append({
                "generation": i,
                "seed": seed,
                "target_emotion": args.target_polarity,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt_complete,
                "samples": all_samples,
                "sample_distribution": sample_info,
                "total_samples": len(all_samples),
                "result": user.model_dump()
            })
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            all_results.append({
                "generation": i,
                "seed": seed,
                "target_emotion": args.target_polarity,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt_complete,
                "samples": all_samples,
                "sample_distribution": sample_info,
                "total_samples": len(all_samples),
                "error": str(e),
                "raw": raw
            })

    except Exception as e:
        print(f"Errore durante la generazione {i}: {e}")
        all_results.append({
            "generation": i,
            "seed": seed,
            "target_emotion": args.target_polarity,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt_complete,
            "samples": all_samples,
            "sample_distribution": sample_info,
            "total_samples": len(all_samples),
            "error": str(e)
        })

print("\n📄 Risultati completi:")
print(json.dumps(all_results, indent=2, ensure_ascii=False))

filename = f"fewShot_generation_Ollama_{args.model.replace(':', '_')}_{args.target_polarity}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"✅ Salvati i risultati in '{filename}'")

print("\n" + "="*80)
print("🔍 STAMPA COMPLETA DI TUTTI I PROMPT UTILIZZATI")
print("="*80)

for prompt_data in all_prompts:
    print(f"\n{'='*50}")
    print(f"GENERAZIONE {prompt_data['generation']} (Seed: {prompt_data['seed']})")
    print(f"{'='*50}")
    
    print(f"\n🤖 SYSTEM PROMPT:")
    print("-" * 40)
    print(prompt_data['system_prompt'])
    
    print(f"\n👤 USER PROMPT:")
    print("-" * 40)
    print(prompt_data['user_prompt'])
    
    print(f"\n📋 MESSAGGI COMPLETI INVIATI A OLLAMA:")
    print("-" * 40)
    for msg in prompt_data['full_messages']:
        print(f"Role: {msg['role']}")
        print(f"Content: {msg['content']}")
        print("-" * 20)

print(f"\n{'='*80}")
print(f"📊 RIEPILOGO: Generati {len(all_prompts)} prompt completi")
print(f"🎯 Target emotion: {args.target_polarity}")
print(f"📈 Samples per emotion: {args.n_samples}")
print(f"🤖 Modello utilizzato: {args.model}")
print("="*80)
