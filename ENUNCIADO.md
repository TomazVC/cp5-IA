# CP5 — Enunciado

Enunciado fornecido pelo usuário em 14/10/2026.

Treinar uma LLM local e apresentar todas informações, incluindo o código no GitHub.
Selecionar/escolher LLM.

- Escolher um problema específico para a LLM treinar/resolver.
- Testar diferentes parâmetros.
- Verificar/avaliar o modelo (respostas).
- Fazer todos testes utilizando GPU.
- Criar um frontend para o usuário inserir/criar o prompt.
- Apresentar um print com as informações da GPU instalada e funcionando dentro do VSCode.
- Apresentar toda documentação e código no GitHub.

Obs. 1: plugins do VSCode que ajudam no monitoramento da GPU: GPU Environment e GPU Monitor.
Obs. 2: instalar a versão correta do CUDA para a respectiva placa de vídeo.

## Interpretação inicial

- Proposta: ajustar um modelo pré-treinado localmente (fine-tuning com LoRA). O enunciado não exige explicitamente treinamento do zero; confirmar com o professor se houver dúvida sobre essa interpretação.
- Experimentos de treinamento e inferência devem executar na GPU; não aceitar fallback silencioso para CPU.
- Comparar o modelo original com as versões treinadas em exemplos reservados e registrar também respostas ruins.
- Print do driver sozinho não comprova treinamento: juntar evidências de PyTorch CUDA, operação real na GPU e execução no terminal integrado do VSCode.
- Guardar código, dados publicáveis, configurações, resultados, prints e instruções de reprodução neste projeto. Publicar código e documentação no GitHub; definir o destino com o usuário.

## Diagnóstico inicial — 14/10/2026

- Pasta inicialmente vazia e sem repositório Git.
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU; VRAM: 6144 MiB; driver: 610.74.
- Python: 3.12.10.
- PyTorch do Python consultado: 2.5.1+cpu. Não serve para os experimentos exigidos na GPU.
- Transformers: 5.14.1; PEFT não encontrado nesse Python.
- VSCode encontrado no PATH; nvcc não encontrado no PATH (isso não comprova ausência de Toolkit).
- Uso da GPU pelo PyTorch ainda não validado.

## Proposta para discutir

Problema: classificar chamados de suporte em português e sugerir uma ação inicial.
Modelo candidato: Qwen/Qwen2.5-0.5B-Instruct.
Técnica: LoRA, com sequências curtas e batch pequeno; medir consumo de VRAM antes de ampliar.
Interface candidata: Gradio local, com prompt e comparação entre modelo original e treinado.

Separar treino, validação e teste, evitando variações quase idênticas entre conjuntos. Comparar o modelo original e pelo menos três configurações de treinamento. Alterar primeiro apenas a taxa de aprendizado, mantendo dados, seed e demais parâmetros constantes. Depois comparar parâmetros de geração no melhor checkpoint, escolhido pela validação. Usar o teste final somente para relatar o resultado.

Avaliação proposta: acerto da categoria, validade do formato, avaliação manual da ação sugerida, latência e pico de VRAM. Documentar origem/licença dos dados e critérios de avaliação.

Status: proposta ainda não aprovada; sem instalação, código ou treinamento nesta etapa.

## Referências oficiais

- Modelo: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- LoRA/PEFT: https://huggingface.co/docs/peft/en/quicktour
- Instalação PyTorch: https://pytorch.org/get-started/locally/
- CUDA reportado pelo driver: https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html

## Atualização — problema escolhido

O usuário confirmou chamados de suporte. O desenho detalhado está em docs/superpowers/specs/2026-10-14-cp5-design.md; demais escolhas ainda são propostas do assistente para revisão.
