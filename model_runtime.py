"""Mesmo modelo, prompt e geração para CLI, avaliações e interface."""
import gc
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from support import ROOT, SYSTEM_PROMPT, parse_response, validate_prompt

# Definidos antes de importar Hugging Face; downloads permanecem no exercício.
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(ROOT / '.cache/huggingface/hub')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CONFIG = json.loads((ROOT / 'configs/experiments.json').read_text(encoding='utf-8-sig'))
torch.set_num_threads(4)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível. Execute setup.ps1 e check_gpu.py usando o Python de .venv.')


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'], revision=CONFIG['revision'])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def messages(prompt):
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]


def encode_example(tokenizer, row, max_length):
    conversation = messages(row['prompt'])
    prefix = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
    answer = json.dumps({'categoria': row['category'], 'acao_inicial': row['action']}, ensure_ascii=False)
    full = tokenizer.apply_chat_template(conversation + [{'role': 'assistant', 'content': answer}],
                                          tokenize=True, add_generation_prompt=False)
    if len(full) > max_length:
        raise ValueError(f"Exemplo {row.get('id', '?')} tem {len(full)} tokens; limite {max_length}.")
    if full[:len(prefix)] != prefix:
        raise ValueError('Template do tokenizer não preserva o prefixo do assistente.')
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prefix) + full[len(prefix):]}


def collate_examples(examples):
    tokenizer = get_tokenizer()
    length = max(len(row['input_ids']) for row in examples)
    return {
        'input_ids': torch.tensor([r['input_ids'] + [tokenizer.pad_token_id] * (length - len(r['input_ids']))
                                   for r in examples]),
        'attention_mask': torch.tensor([r['attention_mask'] + [0] * (length - len(r['input_ids']))
                                        for r in examples]),
        'labels': torch.tensor([r['labels'] + [-100] * (length - len(r['labels'])) for r in examples])
    }


def load_base_model():
    require_cuda()
    model = AutoModelForCausalLM.from_pretrained(CONFIG['model_id'], revision=CONFIG['revision'],
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda')
    if not all(parameter.device.type == 'cuda' for parameter in model.parameters()):
        raise RuntimeError('Há pesos fora da GPU.')
    return model


class Runtime:
    def __init__(self, adapter=None):
        require_cuda()
        if adapter:
            adapter = Path(adapter)
            marker = adapter / 'training_summary.json'
            if not marker.exists() or not (adapter / 'adapter_model.safetensors').exists():
                raise ValueError('Adaptador ausente ou treino incompleto. Execute experiments.py primeiro.')
            metadata = json.loads(marker.read_text(encoding='utf-8'))
            if metadata['revision'] != CONFIG['revision']:
                raise ValueError('Adaptador treinado com revisão diferente do modelo.')
        self.model = load_base_model()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload()
        self.model.eval()
        self.tokenizer = get_tokenizer()
        self.adapter = str(adapter) if adapter else None

    def generate(self, prompt, temperature=0, seed=42):
        prompt = validate_prompt(prompt)
        if temperature not in CONFIG['temperatures']:
            raise ValueError('Temperatura deve ser 0, 0.3 ou 0.7.')
        tokens = self.tokenizer.apply_chat_template(messages(prompt), tokenize=True, add_generation_prompt=True,
                                                    return_tensors='pt').to('cuda')
        if tokens.shape[1] > CONFIG['max_length']:
            raise ValueError(f'Prompt com {tokens.shape[1]} tokens; limite {CONFIG["max_length"]}. Reduza o chamado.')
        set_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        options = {'do_sample': temperature > 0, 'max_new_tokens': CONFIG['max_new_tokens'],
                   'pad_token_id': self.tokenizer.pad_token_id,
                   'temperature': temperature if temperature > 0 else 1.0,
                   'top_p': CONFIG['top_p'] if temperature > 0 else 1.0,
                   'top_k': 0 if temperature > 0 else 50}
        if temperature > 0:
            options.update(temperature=temperature, top_p=CONFIG['top_p'])
        with torch.inference_mode():
            output = self.model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), **options)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        raw = self.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True).strip()
        return {'raw': raw, 'parsed': parse_response(raw), 'seconds': seconds,
                'peak_vram_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
                'input_tokens': tokens.shape[1], 'output_tokens': output.shape[1] - tokens.shape[1]}

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
