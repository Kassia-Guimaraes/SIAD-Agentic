import streamlit as st
from rag_sns24_gpt4o import inicializar_agent, perguntar_sns24, obter_todas_sessoes, obter_mensagens_sessao
import uuid


if "historico" not in st.session_state:
    st.session_state.historico = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# --- BARRA LATERAL: HISTÓRICO DE CONVERSAS ---
with st.sidebar:
    st.title("📂 Histórico")
    
    if st.button("➕ Nova Conversa"):
        st.session_state.historico = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    
    sessoes_antigas = obter_todas_sessoes()
    
    for s_id, titulo in sessoes_antigas:
    # a label é o Título dinâmico que a IA gerou
        if st.button(titulo, key=s_id, use_container_width=True):
            mensagens = obter_mensagens_sessao(s_id)
            st.session_state.session_id = s_id
            st.session_state.historico = []
            for p, r in mensagens:
                st.session_state.historico.append({"role": "user", "content": p})
                st.session_state.historico.append({"role": "assistant", "content": r})
            st.rerun()

# --- FIM DA BARRA LATERAL ---

prompt, base_dados = inicializar_agent()

st.title("Chatbot SNS24")
st.write("Descreve os teus sintomas para obter orientação.")


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
                resposta_bot = perguntar_sns24(pergunta, prompt_template=prompt, base_dados=base_dados, session_id=st.session_state.session_id)
        except Exception as e:
            resposta_bot = f"Erro ao gerar resposta: {e}"

        st.write(resposta_bot)

    st.session_state.historico.append({"role": "assistant", "content": resposta_bot})