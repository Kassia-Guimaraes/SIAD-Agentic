import streamlit as st
from rag_sns24_simples import inicializar_agent, perguntar_sns24

prompt, base_dados = inicializar_agent()

st.title("Chatbot SNS24")
st.write("Descreve os teus sintomas para obter orientação.")

if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

pergunta = st.chat_input("Escreve os teus sintomas")

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        try:
            with st.spinner("A processar resposta...", show_time=True):
                resposta_bot = perguntar_sns24(pergunta, prompt=prompt, base_dados=base_dados)
        except Exception as e:
            resposta_bot = f"Erro ao gerar resposta: {e}"

        st.write(resposta_bot)

    st.session_state.historico.append({"role": "assistant", "content": resposta_bot})