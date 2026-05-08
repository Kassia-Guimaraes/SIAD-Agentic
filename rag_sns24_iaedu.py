from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

from pathlib import Path
import requests
import os
import json
from dotenv import load_dotenv


load_dotenv()

# IAedu
IAEDU_URL = os.getenv("IAEDU_URL")
IAEDU_API_KEY = os.getenv("IAEDU_API_KEY")
IAEDU_CHANNEL_ID = os.getenv("IAEDU_CHANNEL_ID")

# Ollama local apenas para embeddings
base_url = "http://localhost:11434"

# Escolha do modelo de embeddings
EMBEDDING_MODEL = "bge-m3"
# Alternativas:
# EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL = "nomic-embed-text-v2-moe"

PASTA_DB = Path("chroma_sns24_bge_m3")


def run_chuncks():
    documentos = TextLoader("sns24_kb.txt", encoding="utf-8").load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(documentos)

    print("\n -> chunks prontos")
    return chunks


def criar_prompt():
    return PromptTemplate(
        input_variables=["context", "question", "historico"],
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

Histórico da conversa:
{historico}

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
- [emergência / muito urgente / urgente / pouco urgente / aconselhamento / autocuidado]
- encaminhamento: [112 / Serviço de Urgência / ADC / consulta SNS24 / contacto médico / autocuidado com vigilância]
- justificacao: [2 a 4 frases com base nos sintomas, sinais de alarme e protocolos]

Formato em inglês:
- [emergency / very urgent / urgent / less urgent / advice / self-care]
- referral: [112 / Emergency Department / clinical assessment / SNS24 consultation / medical contact / self-care with monitoring]
- justification: [2 to 4 sentences based on symptoms, warning signs and protocols]

Formato em espanhol:
- [emergencia / muy urgente / urgente / poco urgente / orientación / autocuidado]
- derivación: [112 / Servicio de Urgencias / evaluación clínica / consulta SNS24 / contacto médico / autocuidado con vigilancia]
- justificación: [2 a 4 frases basadas en los síntomas, signos de alarma y protocolos]
"""
    )


def inicializar_agent():
    # Testa se o Ollama local está ativo
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()

    prompt = criar_prompt()

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=base_url
    )

    if PASTA_DB.exists() and PASTA_DB.is_dir():
        base_dados = Chroma(
            persist_directory=str(PASTA_DB),
            embedding_function=embeddings
        )
    else:
        base_dados = Chroma.from_documents(
            documents=run_chuncks(),
            embedding=embeddings,
            persist_directory=str(PASTA_DB)
        )

    print("\n -> base de dados pronta")
    return prompt, base_dados


def extrair_texto_iaedu(resposta):
    """
    A IAedu devolve streaming em várias linhas JSON.
    Esta função extrai a mensagem final.
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

        if tipo == "message" and isinstance(conteudo, dict):
            if isinstance(conteudo.get("content"), str):
                texto_message = conteudo["content"]

        elif tipo == "token" and isinstance(conteudo, str):
            texto_tokens.append(conteudo)

    if texto_message:
        return texto_message.strip()

    if texto_tokens:
        return "".join(texto_tokens).strip()

    return resposta.text.strip()


def chamar_iaedu(prompt_final):
    if not IAEDU_URL:
        raise ValueError("IAEDU_URL não está definida no ficheiro .env")

    if not IAEDU_API_KEY:
        raise ValueError("IAEDU_API_KEY não está definida no ficheiro .env")

    if not IAEDU_CHANNEL_ID:
        raise ValueError("IAEDU_CHANNEL_ID não está definido no ficheiro .env")

    headers = {
        "x-api-key": IAEDU_API_KEY
    }

    data = {
        "message": prompt_final,
        "thread_id": "siad-sns24-thread-001",
        "channel_id": IAEDU_CHANNEL_ID,
        "user_info": json.dumps({
            "id": "user_001",
            "name": "Teste SIAD",
            "email": "teste@example.com"
        })
    }

    resposta = requests.post(
        IAEDU_URL,
        headers=headers,
        data=data,
        timeout=300
    )

    if resposta.status_code >= 400:
        print("\nResposta de erro da IAedu:")
        print(resposta.text[:2000])
        resposta.raise_for_status()

    return extrair_texto_iaedu(resposta)


def perguntar_sns24(sintomas, prompt, base_dados, historico=""):
    retriever = base_dados.as_retriever(search_kwargs={"k": 3})

    # Recupera protocolos com base na última mensagem do utilizador
    docs = retriever.invoke(sintomas)

    contexto = "\n\n---\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt_final = prompt.format(
        context=contexto,
        question=sintomas,
        historico=historico
    )

    resposta = chamar_iaedu(prompt_final)

    return resposta


if __name__ == "__main__":
    prompt, base_dados = inicializar_agent()

    print("\n" + "=" * 100)
    print("Chatbot SNS24")
    print("Escreve 'sair' para terminar / Write 'exit' to finish / Escribe 'salir' para finalizar")
    print("=" * 100)

    while True:
        sintomas = input("\nUtente / User / Usuario: ")

        if sintomas.lower().strip() in ["sair", "exit", "salir"]:
            break

        resposta = perguntar_sns24(sintomas, prompt, base_dados)

        print("\nSNS24 - Assistente Virtual:")
        print(resposta)