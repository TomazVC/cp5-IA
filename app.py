"""Frontend local, sequencial, com uma única variante do modelo na GPU."""
import json
import os
import threading

from support import ROOT, file_hash, validate_prompt
from model_runtime import CONFIG, Runtime

os.environ['GRADIO_TEMP_DIR'] = str(ROOT / '.cache/gradio')
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import gradio as gr

ENGINE = None
ENGINE_ADAPTER = None
LOCK = threading.Lock()


def selected_adapter():
    path = ROOT / 'reports/selection.json'
    if not path.exists():
        raise ValueError('Ainda não há modelo selecionado. Execute experiments.py primeiro.')
    selection = json.loads(path.read_text(encoding='utf-8'))
    adapter = (ROOT / selection['adapter']).resolve()
    if not adapter.is_relative_to((ROOT / 'artifacts').resolve()):
        raise ValueError('Caminho do adaptador fora da pasta artifacts.')
    if selection['config'] != CONFIG:
        raise ValueError('A seleção pertence a outros parâmetros. Confira os experimentos.')
    if not (adapter / 'adapter_model.safetensors').exists():
        raise ValueError('Pesos do adaptador não encontrados. Execute experiments.py.')
    if file_hash(adapter / 'adapter_model.safetensors') != selection['adapter_sha256']:
        raise ValueError('Pesos do adaptador mudaram desde a seleção.')
    return adapter


def respond(prompt, variant, temperature):
    global ENGINE, ENGINE_ADAPTER
    prompt = validate_prompt(prompt)
    if variant not in ('Original', 'Treinado'):
        raise ValueError('Selecione um modelo válido.')
    if temperature not in CONFIG['temperatures']:
        raise ValueError('Temperatura inválida.')
    adapter = selected_adapter() if variant == 'Treinado' else None
    with LOCK:
        if ENGINE is None or ENGINE_ADAPTER != adapter:
            if ENGINE is not None:
                ENGINE.close()
                ENGINE = None
            ENGINE = Runtime(adapter)
            ENGINE_ADAPTER = adapter
            ENGINE.generate('Esqueci a senha do portal.', 0)  # aquecimento
        result = ENGINE.generate(prompt, float(temperature), CONFIG['seed'])
    parsed = result['parsed']
    category = parsed['categoria'] if parsed else 'Formato inválido'
    action = parsed['acao_inicial'] if parsed else 'Veja a resposta bruta abaixo; não foi corrigida automaticamente.'
    details = (f"{CONFIG['model_id']} | {adapter.name if adapter else 'original'} | CUDA | "
               f"{result['seconds']:.2f} s | pico {result['peak_vram_mb']:.0f} MiB | "
               f"{result['output_tokens']} tokens gerados")
    return category, action, result['raw'], details


def ui_respond(prompt, variant, temperature):
    try:
        return respond(prompt, variant, temperature)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error


def build_app():
    selected = ROOT / 'reports/selection.json'
    default_temp = json.loads(selected.read_text(encoding='utf-8'))['temperature'] if selected.exists() else 0
    with gr.Blocks(title='CP5 — Triagem de suporte', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# Triagem de chamados\nCompare a LLM original com o ajuste local feito na sua GPU.')
        gr.Markdown('Categorias: **acesso**, **rede**, **hardware**, **software** e **outros**. '
                    'Demonstração acadêmica com dados sintéticos; confira as orientações antes de aplicá-las.')
        prompt = gr.Textbox(label='Descreva o chamado', lines=4, max_lines=8,
                            placeholder='Ex.: Esqueci a senha do portal e não consigo entrar.')
        with gr.Row():
            variant = gr.Radio(['Original', 'Treinado'], value='Treinado' if selected.exists() else 'Original', label='Modelo')
            temperature = gr.Dropdown(CONFIG['temperatures'], value=default_temp, label='Temperatura (0 = determinística)')
        button = gr.Button('Analisar chamado', variant='primary')
        with gr.Row():
            category = gr.Textbox(label='Categoria', interactive=False)
            action = gr.Textbox(label='Ação inicial', interactive=False, lines=3)
        raw = gr.Textbox(label='Resposta bruta do modelo', interactive=False, lines=4)
        details = gr.Textbox(label='Execução', interactive=False)
        gr.Examples(['Esqueci a senha do portal.', 'O Wi-Fi caiu em todos os computadores da sala.',
                     'O teclado está com uma tecla quebrada.', 'O editor fecha ao abrir um documento.',
                     'Não está funcionando.'], inputs=prompt)
        button.click(ui_respond, [prompt, variant, temperature], [category, action, raw, details],
                     api_name='predict', concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1', server_port=7860, share=False, show_error=True)
