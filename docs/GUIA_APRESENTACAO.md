# Como apresentar o CP5

## Sequência da demonstração

1. Explique o problema: transformar chamados em português em uma categoria e uma ação inicial. Mostre as cinco categorias e o tratamento de casos ambíguos.
2. Explique a escolha: Qwen2.5-0.5B-Instruct é pequeno o suficiente para ajustar na RTX 3050 de 6 GB. LoRA atualiza adaptadores em q_proj/v_proj, preservando os pesos principais; isso é fine-tuning real, com gradientes e atualização de pesos.
3. Mostre `evidence/gpu-vscode.png` ou abra o notebook `evidence/gpu-vscode.ipynb` no VSCode. Se solicitado, execute `check_gpu.py` no terminal integrado usando `.venv`.
4. Mostre `data/manifest.json`: 400 registros, 250/75/75, famílias de cenários distintas e hashes. Explique que são dados sintéticos e que cada conjunto de avaliação tem apenas 15 cenários independentes.
5. Abra `configs/experiments.json` e a tabela de validação em `RESULTADOS.md`. Compare as três taxas de aprendizado mantendo os demais parâmetros constantes. Mostre loss, duração e VRAM registradas em `reports/*/training.json`.
6. Explique a seleção: escolher modelo e temperatura pela validação; só depois abrir o teste final. Uma saída com JSON inválido conta como erro. Acurácia mede categoria, não qualidade da ação.
7. Rode `app.py`, abra http://127.0.0.1:7860 e envie um chamado. Troque entre Original e Treinado com a mesma entrada/temperatura. Mostre categoria, ação, resposta bruta e execução CUDA.
8. Mostre exemplos de falhas e regressões do relatório; apresente as notas de `reports/human_review.csv` e suas médias em `RESULTADOS.md`, identificando a revisão como avaliação qualitativa por IA.
9. Mostre o repositório GitHub com código, dados, configurações, resultados e comandos de reprodução. Ambiente/cache/pesos não precisam estar no Git: são reconstruídos pelos comandos documentados.

## Chamados para experimentar na interface

- "Meu perfil não permite acessar a pasta do departamento."
- "O Wi-Fi parou em toda a sala, mas a rede cabeada funciona."
- "A tela do monitor sofreu uma rachadura."
- "O editor de texto fecha quando abro um arquivo."
- "Algo não está funcionando, mas não sei explicar."

Esses exemplos são para demonstração interativa; não entram retroativamente nas métricas reservadas. O modelo pode errar; preserve a resposta bruta para explicar limitações.

## Perguntas que podem surgir

**É treinamento do zero?** Não. É ajuste supervisionado local de uma LLM pré-treinada, com adaptadores LoRA. O enunciado foi interpretado dessa forma.

**Onde ocorre o treinamento?** Num processo Python independente, na GPU deste computador. Não usa tokens cobrados por uma API OpenAI. A assistência no chat é separada da execução do script.

**Como comprovar que treinou?** Histórico de loss, 96 passos de otimização, atualização de pesos não nula, adaptadores salvos, CUDA/VRAM registradas e comparação com o original.

**Por que o checkpoint salvo pode não ser da última época?** O treino cobre três épocas, mas salva o checkpoint com menor loss de validação.

**Por que não publicar todos os pesos?** O modelo original é obtido da revisão fixada no Hugging Face. Adaptadores/checkpoints são arquivos locais; código/configurações/dados permitem reproduzir o ajuste.

**Pode ser usado num suporte real?** Ainda não foi validado para isso. Os dados são sintéticos, a amostra independente é pequena e a qualidade das ações exige revisão humana.

## Antes da entrega

- Conferir o relatório de resultados reais e os exemplos de erro.
- Apresentar os 25 casos do CSV preenchido e as justificativas, mantendo explícita a autoria por IA.
- Explicar o fine-tuning, confirmado como o treinamento esperado para o exercício.
- Conferir que a branch de entrega no GitHub contém a documentação e as evidências mais recentes. O acesso foi verificado pelo autor.
