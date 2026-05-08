# rag_sns24_simples.py

from pathlib import Path
import os
import json
import requests

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
1) interpretar os sintomas descritos pelo utilizador;
2) usar apenas os protocolos SNS24 fornecidos no contexto;
3) classificar a urgência e indicar o encaminhamento adequado.

Regras:
- Responde na mesma língua usada pelo utilizador. Se o utilizador escrever em português, responde em português. Se escrever em inglês, responde em inglês.
- Não inventes dados que o utilizador não forneceu.
- Se houver sinais de emergência, dá prioridade à segurança.
- Se faltarem dados essenciais, faz UMA pergunta objetiva antes de classificar.
- Usa linguagem simples, respeitosa e clara.
- Este sistema é académico e não substitui avaliação clínica real.

Protocolos SNS24 recuperados:
{context}

Sintomas do utilizador:
{question}

Analisa:
1. Que sintomas foram reportados?
2. Existem sinais de alarme?
3. Qual a urgência clínica?
4. Qual o encaminhamento adequado?

Responde SEMPRE neste formato:

- urgencia: [emergência / urgente / consulta / autocuidado]
- encaminhamento: [112 / Serviço de Urgência / Médico de família ou SNS24 / autocuidado com vigilância]
- justificacao: [2 a 4 frases com base nos sintomas e nos protocolos]
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


def chamar_iaedu(prompt):

    headers = {
        "x-api-key": IAEDU_API_KEY,
        
    }

    data = {
        "message": prompt,
        "thread_id": "siad-sns24-thread-001",
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


def perguntar_sns24(sintomas, prompt_template, base_dados):
    retriever = base_dados.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(sintomas)

    contexto = "\n\n---\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt_final = prompt_template.format(
        context=contexto,
        question=sintomas
    )

    resposta = chamar_iaedu(prompt_final)

    return resposta


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