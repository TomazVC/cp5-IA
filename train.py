"""Treina LoRA na GPU; smoke é separado dos experimentos avaliados."""
import argparse
import gc
import math
import time

from model_runtime import CONFIG, collate_examples, encode_example, get_tokenizer, load_base_model, torch
from support import ROOT, SYSTEM_PROMPT, file_hash, load_splits, prompt_hash, write_json
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed


def run_name(lr):
    return f'lr-{lr:g}'


def full_training_steps(n_rows, batch_size, accumulation, epochs):
    return math.ceil(math.ceil(n_rows / batch_size) / accumulation) * epochs


def train(lr, smoke=False):
    if lr not in CONFIG['learning_rates']:
        raise ValueError('Taxa de aprendizado não prevista em configs/experiments.json.')
    name = 'smoke' if smoke else run_name(lr)
    output = ROOT / 'artifacts' / name
    if output.exists():
        raise ValueError(f'{output} já existe. Não sobrescreveremos um experimento; mova-o antes de repetir.')
    splits = load_splits()
    tokenizer = get_tokenizer()
    # Validar comprimento de TODOS os dados antes de consumir GPU; sem consultar saídas de teste.
    encoded = {split: [encode_example(tokenizer, row, CONFIG['max_length']) for row in rows]
               for split, rows in splits.items()}
    set_seed(CONFIG['seed'])
    model = get_peft_model(load_base_model(), LoraConfig(r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'], target_modules=['q_proj', 'v_proj'],
        task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    parameter = next(p for p in model.parameters() if p.requires_grad and torch.count_nonzero(p).item() == 0)
    before = parameter.detach().clone()
    counts = {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'total': sum(p.numel() for p in model.parameters())}
    args = TrainingArguments(output_dir=str(output / 'checkpoints'), num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else CONFIG['gradient_accumulation_steps'],
        learning_rate=lr, fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=1, logging_steps=1 if smoke else 8, report_to=[],
        seed=CONFIG['seed'], data_seed=CONFIG['seed'], dataloader_num_workers=0,
        remove_unused_columns=False, optim='adamw_torch',
        # Transformers 4.51 subestima o último bloco parcial; orçamento explícito cobre três épocas inteiras.
        max_steps=2 if smoke else full_training_steps(len(encoded['train']), CONFIG['batch_size'],
                                                     CONFIG['gradient_accumulation_steps'], CONFIG['epochs']))
    trainer = Trainer(model=model, args=args,
        train_dataset=encoded['train'][:8] if smoke else encoded['train'],
        eval_dataset=encoded['validation'][:5] if smoke else encoded['validation'],
        data_collator=collate_examples)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = trainer.train()
    final_eval = trainer.evaluate()
    torch.cuda.synchronize()
    delta = (parameter.detach() - before).abs().max().item()
    if delta <= 0 or not math.isfinite(result.training_loss) or not math.isfinite(final_eval['eval_loss']):
        raise RuntimeError('Treino sem atualização válida de pesos ou loss não finita.')
    summary = {'name': name, 'smoke': smoke, 'learning_rate': lr, 'model_id': CONFIG['model_id'],
               'revision': CONFIG['revision'], 'config': CONFIG, 'parameters': counts,
               'system_prompt': SYSTEM_PROMPT, 'prompt_sha256': prompt_hash(),
               'train_loss': result.training_loss, 'eval_loss': final_eval['eval_loss'],
               'duration_seconds': time.perf_counter() - started,
               'peak_vram_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
               'adapter_weight_max_delta': delta, 'global_step': trainer.state.global_step,
               'best_checkpoint': trainer.state.best_model_checkpoint, 'completed_epochs': trainer.state.epoch,
               'device': str(parameter.device), 'gpu': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'data_sha256': {split: file_hash(ROOT / f'data/{split}.jsonl') for split in splits},
               'history': trainer.state.log_history}
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    write_json(output / 'training_summary.json', summary)
    write_json(ROOT / f'reports/{name}/training.json', summary)
    print(f'Treino concluído: {name}; steps={summary["global_step"]}; VRAM={summary["peak_vram_mb"]:.0f} MiB')
    del trainer, model, parameter, before
    gc.collect()
    torch.cuda.empty_cache()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--smoke', action='store_true', help='Apenas dois steps; não conta como experimento final.')
    options = parser.parse_args()
    train(options.lr, options.smoke)
