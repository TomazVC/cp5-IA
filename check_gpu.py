import json
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from model_runtime import require_cuda, torch
from support import ROOT, write_json


def main():
    require_cuda()
    matrix = torch.randn(1024, 1024, device='cuda')
    result = matrix @ matrix
    torch.cuda.synchronize()
    if result.device.type != 'cuda' or not torch.isfinite(result).all().item():
        raise RuntimeError('Operação real na GPU falhou.')
    evidence = {'date': datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(),
                'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
                'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0),
                'total_vram_mb': torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
                'operation': '1024x1024 matrix multiplication', 'result_device': str(result.device),
                'finite_result': True,
                'nvidia_smi': subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout}
    write_json(ROOT / 'evidence/gpu.json', evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
