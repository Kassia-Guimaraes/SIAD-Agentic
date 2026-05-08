import streamlit as st

from rag_sns24_gpt4o import inicializar_agent, perguntar_sns24


st.set_page_config(
    page_title="SNS24 Assistant - SIAD Prototype",
    page_icon="🩺",
    layout="centered"
)


TEXTOS = {
    "Português": {
        "titulo": "🩺 Assistente SNS24",
        "subtitulo": "Protótipo académico de apoio à triagem",
        "aviso": (
            "Este sistema é um protótipo académico desenvolvido para a UC de "
            "Sistemas Inteligentes de Apoio à Decisão. Não substitui avaliação médica profissional. "
            "Em caso de emergência, contacte o 112."
        ),
        "descricao": (
            "Descreva os sintomas de forma simples. "
            "Pode escrever em português, inglês ou espanhol. "
            "O assistente tentará responder na mesma língua usada pelo utilizador."
        ),

        "label_sintomas": "Descreva os sintomas:",
        "placeholder": "Ex.: Tenho dor no peito, suor frio e náuseas.",
        "botao": "Analisar sintomas",
        "erro_vazio": "Por favor, escreva os sintomas antes de analisar.",
        "spinner": "A analisar os sintomas com base nos protocolos SNS24...",
        "sucesso": "Análise concluída",
        "resultado": "Resultado",
        "erro_processamento": "Ocorreu um erro ao processar a análise.",
        "rodape": (
            "Arquitetura: RAG local com ChromaDB + embeddings via Ollama "
            "+ geração de resposta via IAedu/GPT-4o."
        ),
        "idioma": "Idioma da interface"
    },

    "English": {
        "titulo": "🩺 SNS24 Assistant",
        "subtitulo": "Academic triage support prototype",
        "aviso": (
            "This system is an academic prototype developed for the Decision Support Intelligent Systems course. "
            "It does not replace professional medical assessment. "
            "In an emergency, call 112."
        ),
        "descricao": (
            "Describe the symptoms in simple terms. "
            "You may write in Portuguese, English or Spanish. "
            "The assistant will try to answer in the same language used by the user."
        ),

        "label_sintomas": "Describe the symptoms:",
        "placeholder": "Example: I have chest pain, cold sweat and nausea.",
        "botao": "Analyse symptoms",
        "erro_vazio": "Please describe the symptoms before analysing.",
        "spinner": "Analysing symptoms based on SNS24 protocols...",
        "sucesso": "Analysis completed",
        "resultado": "Result",
        "erro_processamento": "An error occurred while processing the analysis.",
        "rodape": (
            "Architecture: local RAG with ChromaDB + embeddings via Ollama "
            "+ answer generation via IAedu/GPT-4o."
        ),
        "idioma": "Interface language"
    },

    "Español": {
        "titulo": "🩺 Asistente SNS24",
        "subtitulo": "Prototipo académico de apoyo a la triaje",
        "aviso": (
            "Este sistema es un prototipo académico desarrollado para la asignatura de "
            "Sistemas Inteligentes de Apoyo a la Decisión. No sustituye una evaluación médica profesional. "
            "En caso de emergencia, contacte con el 112."
        ),
        "descricao": (
            "Describa los síntomas de forma sencilla. "
            "Puede escribir en portugués, inglés o español. "
            "El asistente intentará responder en la misma lengua utilizada por el usuario."
        ),

        "label_sintomas": "Describa los síntomas:",
        "placeholder": "Ej.: Tengo dolor en el pecho, sudor frío y náuseas.",
        "botao": "Analizar síntomas",
        "erro_vazio": "Por favor, describa los síntomas antes de analizar.",
        "spinner": "Analizando los síntomas con base en los protocolos SNS24...",
        "sucesso": "Análisis concluido",
        "resultado": "Resultado",
        "erro_processamento": "Ocurrió un error al procesar el análisis.",
        "rodape": (
            "Arquitectura: RAG local con ChromaDB + embeddings vía Ollama "
            "+ generación de respuesta vía IAedu/GPT-4o."
        ),
        "idioma": "Idioma de la interfaz"
    }
}


@st.cache_resource
def carregar_sistema():
    """
    Inicializa o prompt e a base vetorial apenas uma vez.
    Isto evita recriar/recarregar a Chroma a cada interação.
    """
    prompt, base_dados = inicializar_agent()
    return prompt, base_dados


def main():
    idioma = st.sidebar.selectbox(
        "Language / Idioma",
        ["Português", "English", "Español"]
    )

    t = TEXTOS[idioma]

    st.title(t["titulo"])
    st.subheader(t["subtitulo"])

    st.warning(t["aviso"])

    st.markdown(t["descricao"])


    sintomas = st.text_area(
        t["label_sintomas"],
        height=140,
        placeholder=t["placeholder"]
    )

    analisar = st.button(t["botao"], type="primary")

    if analisar:
        if not sintomas.strip():
            st.error(t["erro_vazio"])
            return

        try:
            with st.spinner(t["spinner"]):
                prompt, base_dados = carregar_sistema()
                resposta = perguntar_sns24(sintomas, prompt, base_dados)

            st.success(t["sucesso"])

            st.markdown(f"### {t['resultado']}")
            st.markdown(resposta)

        except Exception as erro:
            st.error(t["erro_processamento"])
            st.exception(erro)

    st.divider()

    st.caption(t["rodape"])


if __name__ == "__main__":
    main()