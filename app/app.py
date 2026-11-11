"""Streamlit demo: type a review, get the sentiment, the priority and the why."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict  # noqa: E402

EXAMPLES = {
    "Escolha um exemplo…": "",
    "Elogio": "Produto excelente, chegou antes do prazo e muito bem embalado. Recomendo!",
    "Reclamação de entrega": "Até agora não recebi o produto e já passou do prazo. Ninguém responde.",
    "Produto com defeito": "Veio quebrado, quero devolver e receber meu dinheiro de volta",
    "Misto (difícil)": "Entrega rápida, mas o produto é bem diferente da foto",
    "Curto e ambíguo": "ok",
}

st.set_page_config(page_title="Triagem de avaliações", page_icon="💬", layout="centered")
st.title("Triagem de avaliações de clientes")
st.caption(
    "Modelo treinado com ~29 mil avaliações reais de e-commerce brasileiro (Olist, 2016–2018). "
    "Ele estima se o comentário é uma reclamação, para que o atendimento priorize quem está insatisfeito."
)

choice = st.selectbox("Exemplos", list(EXAMPLES))
text = st.text_area("Comentário do cliente", value=EXAMPLES[choice], height=110,
                    placeholder="Digite uma avaliação em português…")

if st.button("Classificar", type="primary", disabled=not text.strip()):
    result = predict(text)
    negative = result.label == "negativa"

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentimento", result.label.capitalize())
    col2.metric("Prob. de reclamação", f"{result.probability_negative:.0%}")
    col3.metric("Prioridade", result.priority.capitalize())
    st.progress(result.probability_negative)

    if negative:
        st.error(f"Reclamação provável — encaminhar ao atendimento (limiar {result.threshold:.2f}).")
    else:
        st.success(f"Avaliação positiva — sem ação necessária (limiar {result.threshold:.2f}).")

    if result.contributions:
        st.subheader("Por que o modelo decidiu assim")
        table = pd.DataFrame(result.contributions, columns=["termo", "peso"])
        table["direção"] = table["peso"].map(lambda w: "→ reclamação" if w > 0 else "→ elogio")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Peso = quanto o termo empurrou a decisão. Positivo aproxima de reclamação, negativo de elogio.")
    else:
        st.warning("Nenhuma palavra do comentário é conhecida pelo modelo; a resposta não é confiável.")

with st.expander("Resultados do modelo no teste (jun–ago/2018)"):
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        chosen = metrics["test"]["at_threshold"]
        st.write(f"**Modelo:** {metrics['chosen_model']} — {metrics['test_size']} avaliações nunca vistas no treino")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recall (reclamações achadas)", f"{chosen['recall_neg']:.0%}")
        c2.metric("Precisão dos alertas", f"{chosen['precision_neg']:.0%}")
        c3.metric("F1 (negativa)", f"{chosen['f1_neg']:.2f}")
        st.image(str(ROOT / "reports" / "figures" / "04_matriz_confusao.png"), width=360)
    st.caption("Limitação: o rótulo vem da nota (1–2 = negativa, 4–5 = positiva), não de uma leitura humana do texto.")
