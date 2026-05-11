import streamlit as st
from rag_sns24_gpt4o_memory import (
    inicializar_agent,
    perguntar_sns24,
    obter_todas_sessoes,
    obter_mensagens_sessao,
    guardar_avaliacao
)
import uuid
import unicodedata


prompt, base_dados = inicializar_agent()

if "historico" not in st.session_state:
    st.session_state.historico = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "mostrar_avaliacao" not in st.session_state:
    st.session_state.mostrar_avaliacao = False

if "avaliacao_enviada" not in st.session_state:
    st.session_state.avaliacao_enviada = False

if "nota_avaliacao" not in st.session_state:
    st.session_state.nota_avaliacao = None
    
if "mostrar_pergunta_final" not in st.session_state:
    st.session_state.mostrar_pergunta_final = False
    
if "ultima_pergunta" not in st.session_state:
    st.session_state.ultima_pergunta = None

if "ultima_resposta" not in st.session_state:
    st.session_state.ultima_resposta = None

# Dicionário com textos da interface
TEXTOS = {
    "Português": {
        "titulo": "Chatbot SNS24",
        "descricao": "Descreve os teus sintomas para obter orientação.",
        "placeholder": "Escreve os teus sintomas",
        "spinner": "A processar resposta...",
        "erro": "Erro ao gerar resposta:",
        "emergencia": "Situação de emergência",
        "avaliacao": "Como avalia o meu atendimento?",
        "avaliacao_1": "Precisa melhorar",
        "avaliacao_2": "Bom",
        "avaliacao_3": "Excelente",
        "obrigado": "Obrigado pela avaliação",
        "aviso_curto": "Protótipo académico · Não substitui avaliação médica profissional · Em emergência, contacte 112",
        "mais_info": "Deseja mais alguma informação?",
        "sim": "Sim",
        "nao": "Não",
        "escreva_extra": "Pode escrever a sua dúvida abaixo."
    },
    "English": {
        "titulo": "SNS24 Chatbot",
        "descricao": "Describe your symptoms to receive guidance.",
        "placeholder": "Write your symptoms",
        "spinner": "Processing response...",
        "erro": "Error generating response:",
        "emergencia": "Emergency situation",
        "avaliacao": "How would you rate this assistance?",
        "avaliacao_1": "Needs improvement",
        "avaliacao_2": "Good",
        "avaliacao_3": "Excellent",
        "obrigado": "Thank you for your rating",
        "aviso_curto": "Academic prototype · Does not replace professional medical assessment · In an emergency, call 112",
        "mais_info": "Would you like any further information?",
        "sim": "Yes",
        "nao": "No",
        "escreva_extra": "You can write your question below."
    },
    "Español": {
        "titulo": "Chatbot SNS24",
        "descricao": "Describe tus síntomas para recibir orientación.",
        "placeholder": "Escribe tus síntomas",
        "spinner": "Procesando respuesta...",
        "erro": "Error al generar respuesta:",
        "emergencia": "Situación de emergencia",
        "avaliacao": "¿Cómo evalúa esta atención?",
        "avaliacao_1": "Necesita mejorar",
        "avaliacao_2": "Bueno",
        "avaliacao_3": "Excelente",
        "obrigado": "Gracias por tu evaluación",
        "aviso_curto": "Prototipo académico · No sustituye una evaluación médica profesional · En caso de emergencia, contacte con el 112",
        "mais_info": "¿Desea más información?",
        "sim": "Sí",
        "nao": "No",
        "escreva_extra": "Puede escribir su duda abajo."
    }
}


# --- BARRA LATERAL: HISTÓRICO DE CONVERSAS ---
with st.sidebar:
    st.title("📂 Histórico")
    
    if st.button("➕ Nova Conversa"):
        st.session_state.historico = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.mostrar_avaliacao = False
        st.session_state.mostrar_pergunta_final = False
        st.session_state.avaliacao_enviada = False
        st.session_state.nota_avaliacao = None
        st.rerun()

    st.divider()
    
    sessoes_antigas = obter_todas_sessoes()
    
    for s_id, titulo in sessoes_antigas:
    # a label é o Título dinâmico que a IA gerou
        if st.button(titulo, key=s_id, use_container_width=True):
            mensagens = obter_mensagens_sessao(s_id)
            st.session_state.session_id = s_id
            st.session_state.historico = []
            st.session_state.mostrar_pergunta_final = False
            st.session_state.mostrar_avaliacao = False
            st.session_state.avaliacao_enviada = False
            st.session_state.nota_avaliacao = None
            
            for p, r in mensagens:
                st.session_state.historico.append({"role": "user", "content": p})
                st.session_state.historico.append({"role": "assistant", "content": r})
            st.rerun()
            
    # Caixa lateral para escolher o idioma da interface
    idioma = st.selectbox(
        "Idioma / Language",
        ["Português", "English", "Español"]
    )

    t = TEXTOS[idioma]


st.markdown(
    f"""
    <div style="
        text-align: center;
        margin: 35px auto 35px auto;
        padding: 28px 30px;
        max-width: 760px;
        border-radius: 22px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.22);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    ">
        <div style="font-size: 48px; margin-bottom: 8px;">🩺</div>
        <h1 style="
            margin-bottom: 8px;
            font-size: 38px;
            color: var(--text-color);
            font-weight: 800;
        ">
            {t["titulo"]}
        </h1>
        <p style="
            font-size: 20px;
            color: var(--text-color);
            opacity: 0.85;
            margin-bottom: 12px;
        ">
            {t["descricao"]}
        </p>
        <p style="
            font-size: 14px;
            color: var(--text-color);
            opacity: 0.65;
            margin-top: 10px;
        ">
            {t["aviso_curto"]}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- FIM DA BARRA LATERAL ---


def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def resposta_final(resposta):
    texto = resposta.lower()

    tem_encaminhamento = (
        "encaminhamento:" in texto
        or "referral:" in texto
        or "derivación:" in texto
        or "derivacion:" in texto
        or "derivacao:" in texto
        or "encaminamiento:" in texto
    )

    tem_justificacao = (
        "justificacao:" in texto
        or "justificação:" in texto
        or "justification:" in texto
        or "justificación:" in texto
    )

    tem_encaminhamento_clinico = (
        "112" in texto
        or "inem" in texto
        or "servico de urgencia" in texto
        or "serviço de urgência" in resposta.lower()
        or "centro de saude" in texto
        or "centro de saúde" in resposta.lower()
        or "auto-cuidado" in texto
        or "autocuidado" in texto
        or "tratamento em casa" in texto
    )

    return (tem_encaminhamento and tem_justificacao) or tem_encaminhamento_clinico


def e_emergencia(resposta):
    texto = normalizar_texto(resposta)

    linhas_encaminhamento = [
        linha.strip()
        for linha in texto.splitlines()
        if (
            "encaminhamento:" in linha
            or "referral:" in linha
            or "derivacao:" in linha
            or "derivacion:" in linha
        )
    ]

    for linha in linhas_encaminhamento:
        if "112" in linha or "INEM" in linha:
            return True

    return False


for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant" and e_emergencia(msg["content"]):
            st.image("ligar-inem-112.jpg", width=90)
            st.markdown(
                f"<h6 style='color:#ff4b4b;'>🚨 {t['emergencia']}</h6>",
                unsafe_allow_html=True
            )

        st.write(msg["content"])

pergunta = st.chat_input(t["placeholder"])


if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        try:
            with st.spinner(t["spinner"], show_time=True):
                resposta_bot = perguntar_sns24(pergunta, prompt_template=prompt, base_dados=base_dados, session_id=st.session_state.session_id)
        except Exception as e:
            resposta_bot = f"{t['erro']} {e}"

        if e_emergencia(resposta_bot):
            st.image("ligar-inem-112.jpg", width=90)
            st.markdown(
                f"<h6 style='color:#ff4b4b;'>🚨 {t['emergencia']}</h6>",
                unsafe_allow_html=True)
        
        st.write(resposta_bot)
        
        if resposta_final(resposta_bot):
            st.session_state.ultima_pergunta = pergunta
            st.session_state.ultima_resposta = resposta_bot

            st.session_state.mostrar_pergunta_final = True
            st.session_state.mostrar_avaliacao = False
            st.session_state.avaliacao_enviada = False
            st.session_state.nota_avaliacao = None
                    
    st.session_state.historico.append({"role": "assistant", "content": resposta_bot})
    
    
# --- PERGUNTA FINAL: MAIS INFORMAÇÃO OU AVALIAÇÃO ---

if st.session_state.mostrar_pergunta_final and not st.session_state.mostrar_avaliacao:
    st.divider()
    st.markdown(f"##### {t['mais_info']}")
      
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] button {
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
            background-color: #198754 !important;
            color: white !important;
            border: 1px solid #198754 !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
            background-color: #dc3545 !important;
            color: white !important;
            border: 1px solid #dc3545 !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
            background-color: #157347 !important;
            color: white !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
            background-color: #bb2d3b !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_sim, col_nao = st.columns(2)
    
    with col_sim:
        if st.button(t["sim"], key=f"mais_info_sim_{st.session_state.session_id}"):
            st.session_state.mostrar_pergunta_final = False
            st.session_state.mostrar_avaliacao = False
            st.info(t["escreva_extra"])

    with col_nao:
        if st.button(t["nao"], key=f"mais_info_nao_{st.session_state.session_id}"):
            st.session_state.mostrar_pergunta_final = False
            st.session_state.mostrar_avaliacao = True
            st.rerun()


# --- AVALIAÇÃO DO ATENDIMENTO ---

if st.session_state.mostrar_avaliacao:
    st.divider()

    if not st.session_state.avaliacao_enviada:
        st.markdown(f"##### {t['avaliacao']}")

        st.markdown(
            """
            <style>
            .st-key-avaliacao_excelente button,
            .st-key-avaliacao_bom button,
            .st-key-avaliacao_melhorar button {
                background-color: #6b7280 !important;
                color: white !important;
                border: 1px solid #6b7280 !important;
                border-radius: 12px !important;
                font-weight: 700 !important;
                width: 100% !important;
                padding: 0.7rem 1rem !important;
                margin-bottom: 0.35rem !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16) !important;
                transition: all 0.15s ease-in-out !important;
            }

            .st-key-avaliacao_excelente button:hover,
            .st-key-avaliacao_bom button:hover,
            .st-key-avaliacao_melhorar button:hover {
                background-color: #4b5563 !important;
                color: white !important;
                border: 1px solid #4b5563 !important;
                transform: translateY(-1px);
                box-shadow: 0 5px 14px rgba(0, 0, 0, 0.20) !important;
            }

            .st-key-avaliacao_excelente button:active,
            .st-key-avaliacao_bom button:active,
            .st-key-avaliacao_melhorar button:active {
                background-color: #374151 !important;
                color: white !important;
                border: 1px solid #374151 !important;
                transform: translateY(0px);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        espaco_esq, col_centro, espaco_dir = st.columns([2, 2, 2])

        with col_centro:
            with st.container(key="avaliacao_excelente"):
                if st.button(
                    t["avaliacao_3"],
                    key=f"avaliacao_3_{st.session_state.session_id}",
                    use_container_width=True
                ):
                    st.session_state.nota_avaliacao = 3
                    st.session_state.avaliacao_enviada = True

                    guardar_avaliacao(
                        session_id=st.session_state.session_id,
                        nota=3,
                        pergunta_utente=st.session_state.ultima_pergunta,
                        resposta_sistema=st.session_state.ultima_resposta
                    )

                    st.rerun()

            with st.container(key="avaliacao_bom"):
                if st.button(
                    t["avaliacao_2"],
                    key=f"avaliacao_2_{st.session_state.session_id}",
                    use_container_width=True
                ):
                    st.session_state.nota_avaliacao = 2
                    st.session_state.avaliacao_enviada = True

                    guardar_avaliacao(
                        session_id=st.session_state.session_id,
                        nota=2,
                        pergunta_utente=st.session_state.ultima_pergunta,
                        resposta_sistema=st.session_state.ultima_resposta
                    )

                    st.rerun()

            with st.container(key="avaliacao_melhorar"):
                if st.button(
                    t["avaliacao_1"],
                    key=f"avaliacao_1_{st.session_state.session_id}",
                    use_container_width=True
                ):
                    st.session_state.nota_avaliacao = 1
                    st.session_state.avaliacao_enviada = True

                    guardar_avaliacao(
                        session_id=st.session_state.session_id,
                        nota=1,
                        pergunta_utente=st.session_state.ultima_pergunta,
                        resposta_sistema=st.session_state.ultima_resposta
                    )

                    st.rerun()

    else:
        st.success(f"{t['obrigado']}: {st.session_state.nota_avaliacao}")