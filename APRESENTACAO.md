# Roteiro da apresentação (~10 min)

Cada bloco tem um responsável sugerido. Ajustem os nomes para que a contribuição de cada integrante fique visível, como pede o enunciado.

| # | Bloco | Tempo | O que mostrar | Responsável |
|---|---|---|---|---|
| 1 | Problema | 1 min | Uma equipe de atendimento recebe milhares de comentários e não lê todos. Quem usaria a solução e qual é o ganho: achar as reclamações primeiro. | _Nome 1_ |
| 2 | Dados | 1,5 min | Olist/Kaggle, 99 mil avaliações, 59% sem texto. Rótulo vindo da nota (1–2 e 4–5, sem a nota 3). Figura `01_notas.png`. | _Nome 1_ |
| 3 | Preparação | 1,5 min | Normalização de acentos ("ótimo"/"otimo"). Por que NÃO tirar stop words ("não recebi"). Divisão temporal e o motivo dela (`03_negativas_por_mes.png`). | _Nome 2_ |
| 4 | Modelagem | 1,5 min | Tabela da validação cruzada: baseline Dummy → regra de palavras-chave → modelos. Por que a Reg. Logística (empate técnico, mais simples, interpretável). PR-AUC como métrica. Limiar escolhido no treino. | _Nome 2_ |
| 5 | Resultados | 1,5 min | Teste: recall 93% e precisão 83%. Matriz de confusão. Próximo da validação cruzada, ou seja, sem overfitting. A regra de palavras-chave acha só 44%. | _Nome 3_ |
| 6 | Erros e limitações | 1,5 min | `reports/erros.md`: "Não recebi" com 5 estrelas, porque o rótulo é ruidoso. Comentários curtos (F1 0,68). Nota 3. | _Nome 3_ |
| 7 | **Demo ao vivo** | 2 min | `streamlit run app/app.py` (sequência abaixo). | _Nome 4_ |

## Sequência da demo

1. **"Elogio"** → positiva, 0%. Mostrar os termos "excelente" e "recomendo" puxando para elogio.
2. **"Reclamação de entrega"** → negativa, prioridade alta. Mostrar "nao recebi" e "passou do".
3. **"Misto (difícil)"** → o modelo erra e diz positiva. É o momento de mostrar a limitação: os pesos de "rápida" e "diferente" se anulam, e o modelo não entende o "mas".
4. **Comentário digitado pela turma.** Peçam uma frase para a plateia. Funciona bem como encerramento.

## Perguntas prováveis

- **Por que não usar acurácia?** O Dummy já tem 75% de acurácia sem acertar nenhuma reclamação.
- **Por que divisão temporal?** O modelo vai ser usado em avaliações futuras. Uma divisão aleatória mistura períodos e superestima o resultado.
- **Por que tirar a nota 3?** Ela é ambígua: a nota 3 tem tanto "Satisfeito" quanto "Não funcionou, parece queimado".
- **Como melhorar?** Rotular uma amostra à mão, usar o BERTimbau e criar uma classe neutra.
