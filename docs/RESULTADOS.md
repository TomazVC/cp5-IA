# Resultados reais — CP5

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU. Modelo: Qwen/Qwen2.5-0.5B-Instruct.
Revisão: `7ae557604adf67be50417f59c2c2f167def9a775`. Seed: 42.

Parâmetros totais (incluindo adaptadores): 494,573,440; treináveis: 540,672. Todos os três treinos completam três épocas/96 atualizações; o checkpoint salvo usa menor loss de validação.

## Validação — treinamento

| Modelo | Learning rate | Acurácia | JSON válido | Loss validação | Tempo treino (s) | VRAM treino (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Original | — | 37.3% | 100.0% | — | — | — |
| lr-5e-05 | 5e-05 | 58.7% | 100.0% | 1.5067 | 167.9 | 1382 |
| lr-0.0001 | 0.0001 | 72.0% | 100.0% | 1.3914 | 168.7 | 1382 |
| lr-0.0002 | 0.0002 | 88.0% | 100.0% | 1.3139 | 167.5 | 1382 |

## Validação — geração

| Temperatura | Acurácia | JSON válido | Tempo médio (s) |
|---:|---:|---:|---:|
| 0 | 88.0% | 100.0% | 0.83 |
| 0.3 | 88.0% | 100.0% | 0.81 |
| 0.7 | 82.7% | 100.0% | 0.80 |

Seleção somente na validação: **lr-0.0002**, temperatura **0**.

## Teste final reservado — 75 chamados

| Modelo | Acurácia | Macro-F1 | JSON válido | Latência média (s) | Pico VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| original | 33.3% | 0.284 | 100.0% | 0.85 | 976 |
| lr-0.0002 | 73.3% | 0.730 | 100.0% | 0.83 | 976 |

JSON inválido conta como erro de classificação e permanece no denominador. Macro-F1 inclui todas as cinco categorias. Matrizes de confusão e respostas brutas estão nos JSONs em reports/.

## Exemplos

### Melhoria

Chamado (acesso-13-0, esperado acesso): Tenho conta ativa, mas meu perfil não permite aprovar solicitações.

Original:
```text
{"categoria": "software", "acao_inicial": "Verifique as configurações do perfil e tente novamente."}
```
Treinado:
```text
{"categoria": "acesso", "acao_inicial": "Confira a configuração da sua conta e solicite ao administrador que autorize o processo."}
```

Regressão: nenhum caso nesta execução segundo acerto de categoria.

### Falha do treinado

Chamado (acesso-15-0, esperado acesso): O login corporativo diz que meu usuário pertence a uma organização diferente.

Original:
```text
{"categoria": "software", "acao_inicial": "Verifique se o sistema de login está configurado para usar o nome da organização como identificador único."}
```
Treinado:
```text
{"categoria": "outros", "acao_inicial": "Informe ao responsável pelo sistema a sua nova organização e solicite a atualização do login."}
```

## Revisão qualitativa por IA

O arquivo [reports/human_review.csv](../reports/human_review.csv) contém 25 casos e 150 notas preenchidas de 0 a 2. A autoria e as justificativas estão identificadas nas observações do CSV.

| Critério — média de 0 a 2 | Original | Treinado |
|---|---:|---:|
| Pertinência | 0,76 | 1,28 |
| Clareza | 1,28 | 1,52 |
| Segurança | 1,40 | 1,72 |

Rubrica: 0 inadequada, 1 parcialmente adequada, 2 adequada. Pertinência mede a adequação da ação ao problema; clareza, sua compreensão; segurança, as precauções e a ausência de orientações perigosas. Acertar a categoria não garante uma ação adequada.

A avaliação foi realizada por IA e não foi cega, pois as variantes dos modelos estavam identificadas. As notas são subjetivas e não constituem revisão humana independente.

## Limitações

Dados sintéticos: 80 cenários com cinco redações cada; os conjuntos têm famílias distintas, mas apenas 15 cenários independentes em cada conjunto de avaliação. As 75 respostas não são 75 cenários independentes. Não extrapolar a acurácia para chamados reais.

Uma única seed de treinamento e de amostragem: a comparação de temperaturas não mede estabilidade entre múltiplas seeds. Latência mede a chamada de geração com sincronização CUDA, excluindo tokenização, seed, decode e carregamento; varia com comprimento da resposta, temperatura da GPU e programas concorrentes.

As notas qualitativas complementam as métricas automáticas e não demonstram segurança para atendimento real.

## Evidências e publicação

Os prints estão em [evidence/](../evidence/) e incorporados ao [README](../README.md):

- [GPU funcionando no VSCode](../evidence/gpu-vscode.png).
- [Frontend com modelo original](../evidence/evidencia-front-original.png).
- [Frontend com modelo treinado](../evidence/evidencia-front-treinado.png).
- [Notebook de comprovação da GPU](../evidence/evidencia-cp5.png).

Os prints do frontend usam o mesmo chamado e temperatura 0. O notebook executado está em [gpu-vscode.ipynb](../evidence/gpu-vscode.ipynb).

Repositório de entrega: [TomazVC/cp5-IA](https://github.com/TomazVC/cp5-IA). O autor confirmou a verificação do acesso para entrega.
