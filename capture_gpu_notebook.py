"""Executa um notebook real com kernel da .venv e salva a saída para o VSCode."""
import os
import sys

from support import ROOT, write_json


def main():
    import nbformat
    from nbclient import NotebookClient

    kernel_dir = ROOT / '.cache/jupyter/kernels/cp5'
    write_json(kernel_dir / 'kernel.json', {
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'CP5 (.venv CUDA)', 'language': 'python'
    })
    os.environ['JUPYTER_PATH'] = str(ROOT / '.cache/jupyter')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.cache/jupyter/runtime')
    source = '''import sys, subprocess, torch
from datetime import datetime
from zoneinfo import ZoneInfo
print("CP5 — GPU local funcionando")
print("Data:", datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat())
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "| CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA indisponível"
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**2), "MiB")
a = torch.randn(1024, 1024, device="cuda")
b = a @ a
torch.cuda.synchronize()
assert torch.isfinite(b).all().item()
print("Operação real: matriz 1024x1024 | dispositivo:", b.device, "| resultado finito: True")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], text=True).strip())'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# CP5 — Evidência real de GPU\n'
            'Notebook executado pelo kernel Python de `.venv`; saída preservada para visualização no VSCode.'),
        nbformat.v4.new_code_cell(source)
    ])
    notebook.metadata['kernelspec'] = {'display_name': 'CP5 (.venv CUDA)', 'language': 'python', 'name': 'cp5'}
    NotebookClient(notebook, timeout=120, kernel_name='cp5', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, ROOT / 'evidence/gpu-vscode.ipynb')
    print('Notebook executado na GPU e salvo em evidence/gpu-vscode.ipynb')


if __name__ == '__main__':
    main()
