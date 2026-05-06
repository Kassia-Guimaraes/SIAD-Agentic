# rag_sns24_simples.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import requests

base_url = "http://localhost:11434"
chatbot = None

def run_chuncks():
    documentos = TextLoader("sns24_kb.txt", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(documentos)
    print("\n -> chunks prontos")

    return chunks


def inicializar_agent():
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                És um assistente de triagem clínica baseado nos protocolos SNS24. 
                O teu objetivo é:
                1) conversar com o utilizador para perceber melhor a situação clínica
                2) aplicar os protocolos SNS24 fornecidos em {context}
                3) só depois disso dar uma recomendação final de urgência e encaminhamento.

                Regra geral:
                - Fala em linguagem simples e respeitosa.
                - Faz uma pergunta de cada vez.
                - Se o utilizador der informação espontânea, aproveita-a e evita repetir perguntas.

                Dados mínimos que deves tentar obter ANTES da resposta final:
                - Idade do utente
                - Sintomas principais (já em {question}, mas podes clarificar)
                - Há quanto tempo começaram os sintomas
                - Se os sintomas estão a piorar, a melhorar ou estáveis
                - Medicação habitual e doenças crónicas importantes (ex.: coração, pulmões, diabetes, gravidez)
                - Situação atual: consegue andar, falar em frases completas, beber líquidos, etc.

                Fluxo da conversa:

                1) Confirmar queixas principais
                - Reescreve brevemente o que o utilizador disse e confirma: 
                    "Percebi que está com {question}. É isso?"

                2) Recolher dados essenciais
                - Pergunta, em passos:
                    - "Qual é a sua idade?"
                    - "Há quanto tempo começaram estes sintomas?"
                    - "Os sintomas estão a piorar, a melhorar ou mantêm-se iguais?"
                    - "Tem algum problema de saúde importante (ex.: do coração, pulmões, rins, diabetes, está grávida, ou outro que considere relevante)?"

                3) Procurar sinais de alarme (conforme protocolos SNS24 {context})
                - Coloca perguntas específicas de alarme (exemplos genéricos, adaptar ao protocolo):
                    - Dificuldade em respirar, dor no peito, alteração de consciência, febre muito alta e persistente, rigidez da nuca, dor súbita intensa, etc.
                - Se identificar qualquer sinal de alarme grave, podes encurtar a entrevista e avançar mais depressa para a classificação de urgência.

                4) Só depois disto faz:
                Raciocínio:
                1. Que sintomas foram reportados?
                2. Existem sinais de alarme?
                3. Qual a urgência clínica?

                Resposta final (usa SEMPRE este formato, sem texto extra):
                - urgencia: [por exemplo: emergência / muito urgente / urgente / pouco urgente / aconselhamento / autocuidado]
                - encaminhamento: [por exemplo: 112 / SU hospitalar / ADC / consulta SNS24 / contacto médico / autocuidado com vigilância]
                - justificacao: [explica, em 2-4 frases, com base nos sintomas, sinais de alarme e protocolos {context}]

                Se considerares que ainda faltam dados importantes para classificar pela grelha SNS24, continua a fazer perguntas antes da "Resposta final".
                Nunca inventes dados que o utilizador não forneceu.
                Se o utilizador disser explicitamente que não quer responder a algo, segue com o melhor possível com a informação disponível e menciona a limitação na justificação.
                """
    )

    pasta_db = Path(".chroma_sns24")

    if pasta_db.exists() and pasta_db.is_dir():
        base_dados = Chroma(
            persist_directory=str(pasta_db),
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=base_url
            )
        )
    else:
        base_dados = Chroma.from_documents(
            documents=run_chuncks(),
            embedding=OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=base_url
            ),
            persist_directory=str(pasta_db)
        )
    print("\n -> base de dados pronta")

    return prompt, base_dados

def inicializar_chatbot(prompt=None, base_dados=None):
    global chatbot

    if chatbot is not None:
        return chatbot

    chatbot = RetrievalQA.from_chain_type(
        llm=Ollama(model="tinyllama", temperature=0.1),
        retriever=base_dados.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return chatbot

def perguntar_sns24(sintomas, prompt, base_dados):
    bot = inicializar_chatbot(prompt, base_dados)
    resposta = bot.invoke({"query": sintomas})
    return resposta["result"]