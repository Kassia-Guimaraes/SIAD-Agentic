# rag_sns24_simples.py

from pathlib import Path
import os
import json
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings


load_dotenv()

# API IAedu
IAEDU_URL = os.getenv("IAEDU_URL")
IAEDU_API_KEY = os.getenv("IAEDU_API_KEY")
IAEDU_CHANNEL_ID = os.getenv("IAEDU_CHANNEL_ID")

# Ollama local apenas para embeddings
OLLAMA_LOCAL_URL = "http://localhost:11434"

# Escolhe o modelo de embeddings local
# Opções: "nomic-embed-text", "bge-m3", "nomic-embed-text-v2-moe"
EMBEDDING_MODEL = "bge-m3"

PASTA_DB = Path("chroma_sns24_bge-m3")


def run_chunks():
    documentos = TextLoader("sns24_kb.txt", encoding="utf-8").load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(documentos)

    #print(f"\n -> {len(chunks)} chunks prontos")
    return chunks


def criar_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
És um assistente de triagem clínica baseado nos protocolos SNS24.

O teu objetivo é:
1) conversar com o utilizador para perceber melhor a situação clínica;
2) aplicar os protocolos SNS24 fornecidos no contexto;
3) só depois disso dar uma recomendação final de urgência e encaminhamento.

Dados obrigatórios:
- Idade do utente (se o utente não fornecer a idade, pergunte!).
- Só passe medicação se for em autocuidados.
- Se passar algum medicamenteo (que não seja de prescrição médica), você deve dizer a dosagem, frequência de uso e duração máxima para utilização, e adequar a medicação de acordo com a idade do utente.

Regra de prioridade absoluta:

Se a última mensagem do utilizador indicar qualquer uma das situações abaixo, classifica imediatamente como EMERGÊNCIA, sem pedir idade, género, duração adicional ou outros dados:

- Hemorragia ativa: perda de sangue ativa há 10 minutos ou mais.
- Perda de consciência, desmaio ou inconsciência.
- Perda de força num membro, fraqueza súbita de um lado do corpo ou dificuldade na fala.
- Dor no peito grave: dor em aperto, pressão ou peso no peito há mais de 20 minutos.
- Dificuldade respiratória grave: lábios roxos, respiração ofegante, apneia ou respiração ruidosa.
- Exposição a substâncias tóxicas, como lixívia, pesticidas, produtos químicos ou venenos.

Nestes casos, a resposta final deve ser imediata:
- urgencia: EMERGÊNCIA
- encaminhamento: 112 / INEM
- tempo: Agora
- justificacao: Explica em 2 a 4 frases que existe sinal de alarme grave e que deve contactar imediatamente o 112 / INEM.

Não faças perguntas adicionais nestes casos.

Regras de idioma:
- Responde sempre na mesma língua predominante usada pelo utilizador.
- Se o utilizador escrever em português, responde em português.
- Se escrever em inglês, responde em inglês.
- Se escrever em espanhol, responde em espanhol.
- Se escrever noutra língua, tenta responder nessa língua se for possível.
- Não mistures idiomas na mesma resposta.

Regras clínicas e conversacionais:
- Fala em linguagem simples, respeitosa e clara.
- Faz apenas UMA pergunta de cada vez.
- Usa o histórico da conversa para evitar repetir perguntas já respondidas.
- Se o utilizador der informação espontânea, aproveita-a.
- Não inventes dados que o utilizador não forneceu.
- Se houver sinais claros de emergência, não prolongues a entrevista: dá imediatamente a recomendação de emergência.
- Se faltarem dados importantes, continua a fazer perguntas antes da resposta final.
- Este sistema é académico e não substitui avaliação clínica real.

Dados mínimos que deves obter antes da resposta final:
- Sintomas principais.
- Caracterização dos sintomas (duração, aspecto, frequência, cor - quando cabível)
- Há quanto tempo começaram os sintomas.
- Se estão a piorar, a melhorar ou estáveis.
- Doenças importantes, gravidez ou medicação habitual relevante.
- Estado atual: consegue andar, falar, respirar, beber líquidos, manter-se consciente, etc.
- Sinais de alarme relacionados com o protocolo recuperado.

Protocolos SNS24 recuperados:
{context}

Última mensagem do utilizador:
{question}

Tarefa:
Decide se já há informação suficiente para classificar a situação.

Se ainda faltarem dados importantes:
- Faz UMA pergunta objetiva.
- Não dês ainda a resposta final.

Se já houver informação suficiente OU se existirem sinais claros de emergência:
responde no formato adequado à língua do utilizador.

Formato em português:
- encaminhamento: [112 - INEM / Serviço de Urgência (SU) / Centro de Saúde (CSP) / Autocuidado (Tratamento em casa + reavaliação)]
- justificacao: [2 a 4 frases com base nos sintomas, sinais de alarme e protocolos]

Formato em inglês:
- referral: [112 - INEM / Emergency Service (ED) / Health Center (CSP) / Self-care (Home treatment + reassessment)]
- justification: [2 to 4 sentences based on symptoms, warning signs and protocols]

Formato em espanhol:
- derivación: [112 - INEM / Servicio de Urgencias (SU) / Centro de Salud (CSP) / Autocuidados (Tratamiento domiciliario + reevaluación)]
- justificación: [2 a 4 frases basadas en los síntomas, signos de alarma y protocolos]
"""

    )


def inicializar_base_dados():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_LOCAL_URL
    )

    if PASTA_DB.exists() and PASTA_DB.is_dir():
        #print("\n -> A carregar base vetorial existente")
        base_dados = Chroma(
            persist_directory=str(PASTA_DB),
            embedding_function=embeddings
        )
    else:
        #print("\n -> A criar nova base vetorial")
        base_dados = Chroma.from_documents(
            documents=run_chunks(),
            embedding=embeddings,
            persist_directory=str(PASTA_DB)
        )

    #print("\n -> Base de dados pronta")
    return base_dados


def extrair_texto_iaedu(resposta):
    """
    Extrai o texto final da resposta em streaming da IAedu.
    A IAedu devolve várias linhas JSON. A resposta final costuma vir em:
    {"type": "message", "content": {"content": "..."}}
    """

    texto_tokens = []
    texto_message = None

    for linha in resposta.text.splitlines():
        linha = linha.strip()

        if not linha:
            continue

        try:
            dados = json.loads(linha)
        except Exception:
            continue

        tipo = dados.get("type")
        conteudo = dados.get("content")

        # Resposta final completa
        if tipo == "message" and isinstance(conteudo, dict):
            if isinstance(conteudo.get("content"), str):
                texto_message = conteudo["content"]

        # Tokens parciais
        elif tipo == "token" and isinstance(conteudo, str):
            texto_tokens.append(conteudo)

    if texto_message:
        return texto_message.strip()

    if texto_tokens:
        return "".join(texto_tokens).strip()

    return resposta.text.strip()


def chamar_iaedu(prompt, session_id):

    headers = {
        "x-api-key": IAEDU_API_KEY,
        
    }

    data = {
        "message": prompt,
        "thread_id": session_id,
        "channel_id": IAEDU_CHANNEL_ID,
        "user_info": json.dumps({
            "id": "user_001",
            "name": "Vanessa",
            "email": "vanessag@example.com"
        })
    }

    #print("\n -> A chamar IAedu...")
    #print(" -> IAEDU_URL:", IAEDU_URL)
    #print(" -> API key carregada?", IAEDU_API_KEY is not None)
    #print(" -> Channel ID carregado?", IAEDU_CHANNEL_ID is not None)
    #print(" -> Envio em formato: form-urlencoded")
    

    resposta = requests.post(
        IAEDU_URL,
        headers=headers,
        data=data,
        timeout=300
    )

    #print("\n -> Status IAedu:", resposta.status_code)

    if resposta.status_code >= 400:
        print("\nResposta de erro da IAedu:")
        print(resposta.text[:2000])
        resposta.raise_for_status()

    return extrair_texto_iaedu(resposta)

def gerar_titulo_com_ia(pergunta, resposta, session_id):
    """Pede à IA para resumir a triagem num título de 3 a 5 palavras."""
    prompt_resumo = f"""
    Baseado nesta triagem, gera um título muito curto e profissional (máximo 5 palavras).
    NÃO uses pontuação final. Sê direto.
    
    Pergunta: {pergunta}
    Resposta: {resposta}
    
    Título:"""
    
    try:
        # Chamamos a mesma função de IA 
        titulo = chamar_iaedu(prompt_resumo, session_id)
        # Limpamos possíveis aspas ou lixo que a IA envie
        return titulo.replace('"', '').replace('.', '').strip()
    except:
        return "Nova Triagem"
    
def registar_na_bd(pergunta, resposta, session_id):
    conn = sqlite3.connect('historico_triagem.db')
    cursor = conn.cursor()
    
    # 1. Tabela de Mensagens (Histórico detalhado)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            data_hora TEXT,
            pergunta_utente TEXT,
            resposta_sistema TEXT
        )
    ''')

    # 2. Tabela de Sessões (Para a barra lateral)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessoes (
            session_id TEXT PRIMARY KEY,
            titulo TEXT,
            data_criacao TEXT
        )
    ''')

    # Verifica se a sessão já tem título; se não, gera um
    # Verifica se a sessão é nova
    cursor.execute("SELECT titulo FROM sessoes WHERE session_id = ?", (session_id,))
    if not cursor.fetchone():
        # ---  USAMOS A IA PARA O TÍTULO ---
        titulo_ia = gerar_titulo_com_ia(pergunta, resposta, session_id)
        
        cursor.execute(
            "INSERT INTO sessoes (session_id, titulo, data_criacao) VALUES (?, ?, ?)",
            (session_id, titulo_ia, datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
    
    # Guarda a interação normal
    cursor.execute(
        "INSERT INTO interacoes (session_id, data_hora, pergunta_utente, resposta_sistema) VALUES (?, ?, ?, ?)",
        (session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pergunta, resposta)
    )
    conn.commit()
    conn.close()

def perguntar_sns24(sintomas, prompt_template, base_dados, session_id):
    retriever = base_dados.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(sintomas)

    contexto = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt_final = prompt_template.format(
        context=contexto,
        question=sintomas
    )

    # 1. Obtém a resposta da IA
    resposta = chamar_iaedu(prompt_final, session_id)
    
    # 2. guardar na bd
    try:
        registar_na_bd(sintomas, resposta, session_id)
        
    except Exception as e:
        print(f"Erro ao guardar na BD: {e}")

    return resposta

def obter_todas_sessoes():
    conn = sqlite3.connect('historico_triagem.db')
    cursor = conn.cursor()
    try:
        # Lê o título e o ID da nova tabela 'sessoes'
        cursor.execute("SELECT session_id, titulo FROM sessoes ORDER BY data_criacao DESC")
        sessoes = cursor.fetchall()
    except:
        sessoes = []
    conn.close()
    return sessoes

def obter_mensagens_sessao(session_id):
    """Retorna todas as perguntas e respostas de uma sessão específica."""
    conn = sqlite3.connect('historico_triagem.db')
    cursor = conn.cursor()
    cursor.execute("SELECT pergunta_utente, resposta_sistema FROM interacoes WHERE session_id = ?", (session_id,))
    mensagens = cursor.fetchall()
    conn.close()
    return mensagens

def inicializar_agent():
    prompt = criar_prompt()
    base_dados = inicializar_base_dados()
    return prompt, base_dados


if __name__ == "__main__":
    prompt, base_dados = inicializar_agent()

    print("\n" + "=" * 100)
    print("Chatbot SNS24")
    print("Escreve 'sair' para terminar | Write 'exit' to finish | Escribe 'salir' para finalizar")
    print("=" * 100)
    print("Descreva os seus sintomas: | Describe your symptoms: | Describe tus síntomas:")

    while True:
        sintomas = input("\nUtente: | User: | Usuario:")

        if sintomas.lower().strip() == "sair":
            break
        if sintomas.lower().strip() == "exit":
            break
        if sintomas.lower().strip() == "salir":
            break

        resposta = perguntar_sns24(sintomas, prompt, base_dados)

        print("\nSNS24-Assistente Virtual: | Virtual Assistant: | Asistente Virtual:")
        print(resposta)