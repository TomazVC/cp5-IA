# Análise das respostas e limites do resultado

Esta análise foi feita pelo assistente sobre as respostas registradas. **Não substitui nem preenche a avaliação humana** em `reports/human_review.csv`.

No teste final, o original acertou 25/75 categorias (33,3%) e o treinado 55/75 (73,3%). Ambos geraram JSON válido em 75/75. A melhoria de classificação não comprova a qualidade das ações propostas.

## Exemplos reais observados

| Caso | Resposta/observação | Avaliação qualitativa do assistente |
|---|---|---|
| Perfil sem permissão para aprovar solicitações, acesso-13-0 | Original: software; treinado: acesso, pedindo ao administrador autorização. | Categoria corrigida e orientação compatível com revisão de permissão. |
| Pico de latência, rede-14-0 | Treinado: rede; “Anote a latência e solicite análise ao suporte da rede.” | Ação inicial pertinente e simples. |
| Notebook ligado após derramar líquido, hardware-14-0 | Treinado: outros; “Evite usar o dispositivo e solicite a instalação de limpeza.” | Categoria errada; ação mal formulada e sem orientar o desligamento indicado pela referência. Não usar como orientação técnica. |
| Conector de energia quebrado, hardware-15-0 | Treinado: hardware; “Imprima o status do conector e solicite substituição.” | Categoria correta, mas “imprima o status” não é uma ação inicial pertinente ao defeito físico. |
| Festa de aniversário no escritório, outros-13-0 | Treinado: rede, encaminhando detalhes da festa ao responsável pela rede. | Falha de escopo: classificou um pedido de evento como problema de rede. |
| Senha esquecida, demonstração do frontend | Treinado: acesso; pediu nome de usuário e data da última utilização. | Categoria correta; ação pouco pertinente, pois a referência recomenda recuperação oficial de senha. Registro em evidence/frontend.json. |

As respostas completas estão em `reports/lr-0.0002/test-temp-0.json` e `evidence/frontend.json`. A seleção de modelo foi concluída na validação antes de consultar esses casos de teste.

## O que concluir

- O ajuste aprendeu parte da classificação de domínio e melhorou a métrica escolhida.
- Ações podem continuar genéricas, mal formuladas ou sem precauções necessárias, mesmo com JSON e categoria corretos.
- JSON válido verifica estrutura, não pertinência nem segurança.
- A amostra de teste contém 15 famílias com cinco redações cada; erros nas redações da mesma família são correlacionados.
- A avaliação humana deve examinar ações independentemente do acerto da categoria. Não considerar este modelo pronto para atendimento real.

## Próximos experimentos possíveis

Somente como trabalho futuro: ampliar cenários/dados, avaliar ações também na validação e considerar um modelo maior se a GPU suportar. Um novo ciclo deve usar um novo teste reservado; não ajustar repetidamente os parâmetros olhando o teste que já foi utilizado. Nenhum desses experimentos adicionais foi executado nesta entrega.
