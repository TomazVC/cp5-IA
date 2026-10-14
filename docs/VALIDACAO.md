# Validação da implementação

## Evidências executadas

- Windows / Python 3.12 / RTX 3050 Laptop, 6 GB.
- PyTorch 2.8.0+cu126 com CUDA runtime 12.6. Operação real de matriz em cuda:0: `evidence/gpu.json` e notebook executado `evidence/gpu-vscode.ipynb`.
- Captura real do notebook aberto no VSCode: `evidence/gpu-vscode.png`.
- Smoke: dois passos de otimização na GPU, atualização de pesos LoRA não nula e loss finita; pico aproximado de 1356 MiB. Smoke é separado dos três treinos completos.
- Suíte: 19 testes passaram. Inclui inferência real CUDA, máscara de prompt/padding, rejeição de truncamento, JSON inválido no denominador, separação de famílias, seleção somente na validação, preservação de notas humanas e invalidação de cache ao mudar o prompt.
- Git checkout: bytes dos dados e hashes permanecem iguais com autocrlf true/false; `.gitattributes` preserva LF.
- Ruff: checagem de erros/imports F sem problemas.
- Frontend verificado com cliente oficial Gradio contra o servidor real: modelo original e treinado executaram em CUDA, devolveram categoria/ação/resposta bruta/tempo, e entrada vazia foi rejeitada. Limites de entrada conferidos na configuração servida. Evidência: `evidence/frontend.json`.
- Oito avaliações de 75 respostas auditadas: métricas recalculadas a partir das respostas brutas coincidem com os relatórios. Três treinos com pesos atualizados, 96 passos e três épocas; CSV humano com 25 casos e notas em branco.
- Casos de falha verificados: adaptador ausente é rejeitado antes de carregar o modelo; com CUDA ocultada somente no processo de teste, a execução aborta com mensagem clara e não usa CPU.

## Achados da revisão independente e correções

1. Transformers 4.51.3 calculava 93 atualizações e encerrava em 2.928 épocas. O código agora define 96 atualizações explicitamente para cobrir três épocas. As execuções iniciais foram arquivadas em `.cache/initial-93steps/` e não compõem resultados finais. Todos os três treinos corrigidos confirmaram `global_step=96` e `completed_epochs=3.0`.
2. Prompt de sistema não participava da identidade dos artefatos. Agora seu hash é salvo e conferido em treino, avaliação, seleção e carregamento do adaptador. Resultados antigos não foram retroativamente renomeados como válidos.
3. CSV de avaliação humana podia pertencer a respostas antigas. Agora IDs, chamados, referências e respostas são comparados antes de reutilizar; mudanças exigem arquivar o CSV e notas existentes são preservadas quando correspondem à mesma avaliação.
4. Interface agora informa os limites de 1200 caracteres e 256 tokens antes do envio.

Adicionalmente, a primeira versão gerava JSONL com fins de linha dependentes do sistema. Agora o gerador usa LF e o Git preserva esses bytes, evitando invalidar hashes ao clonar em outro ambiente. Os conteúdos semânticos dos chamados não foram alterados.

## Decisões de execução

- Trabalho direto em cp5 e branch própria: pasta inicialmente sem Git, conforme destino pedido pelo usuário.
- Plano executado na sessão, conforme autorização de implementar e entregar instruções depois.
- Bookkeeping em PowerShell e ledger local, adequados ao ambiente Windows.
- Repetição dos treinos após corrigir épocas, com custo de tempo adicional de GPU, para manter evidência consistente com os requisitos.

## Limites de verificação

Testes de contrato/arquivos são verificações de software e usam CPU; todos os testes que carregam ou executam a LLM, todos os treinos e todas as avaliações do modelo usam CUDA. Não há fallback de modelo para CPU.

A avaliação semântica humana e a publicação GitHub não são substituídas por testes de código. O relatório final dos experimentos é `RESULTADOS.md`, produzido somente depois da execução completa.

A tentativa de visualização automática do frontend no navegador foi interrompida pela ferramenta por não conseguir confirmar a URL da janela com confiança. Não houve mais ações de UI após a interrupção. A verificação funcional do frontend foi feita pelo cliente oficial do Gradio contra o servidor local; não foi concluída uma inspeção visual do frontend no navegador.
