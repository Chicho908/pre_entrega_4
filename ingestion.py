# ingestion.py: Pipeline de Ingesta y Gestion de Metadatos. Carga los PDF
# de /data (politicas propias de una empresa ficticia, EmpresaFlomat, escritas
# para este proyecto), los fragmenta y los sube a Pinecone con metadata
# avanzada (fuente, pagina, categoria, numero de chunk). El texto original
# se guarda DENTRO de la metadata de cada vector (esto lo hace
# automaticamente PineconeVectorStore, en la clave "text") para no depender
# de una base de datos relacional aparte cuando despues se quiere mostrar
# el contenido recuperado.

# logging deja registro de cuantos documentos/chunks se procesan y suben.
import logging

# os.path.basename limpia el nombre de archivo (sin el path completo) para
# guardarlo como metadata legible.
import os

# time.sleep espera entre reintentos si el upsert a Pinecone falla de forma transitoria.
import time

# List tipa la lista de chunks que devuelve cargar_y_fragmentar.
from typing import List

# DirectoryLoader recorre /data cargando cada PDF con PyPDFLoader; PyPDFLoader
# devuelve un Document POR PAGINA, con la pagina ya en la metadata (clave
# "page", 0-indexada) -- de ahi sale el campo "pagina" que pide la consigna.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Document es el tipo que devuelven el loader y el splitter (texto + metadata).
from langchain_core.documents import Document

# HuggingFaceEmbeddings genera los vectores localmente (sin API key ni costo).
from langchain_huggingface import HuggingFaceEmbeddings

# PineconeVectorStore conecta LangChain con el indice de Pinecone (upsert y busqueda).
from langchain_pinecone import PineconeVectorStore

# RecursiveCharacterTextSplitter corta respetando una jerarquia de
# separadores (parrafo -> linea -> oracion -> palabra), igual que en el Modulo 2.
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE

logger = logging.getLogger(__name__)


# Lee los PDF de data_dir (un Document por pagina) y los corta en chunks
# con metadata enriquecida. from_tiktoken_encoder mide chunk_size/chunk_overlap
# en TOKENS (no caracteres), que es lo que realmente limita al modelo de
# embeddings. split_documents conserva la metadata del Document original
# (incluida la pagina) en cada chunk que sale de el.
def cargar_y_fragmentar(data_dir: str = DATA_DIR) -> List[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documentos_crudos = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos_crudos)

    # Metadata avanzada: fuente (nombre de archivo, sin el path completo),
    # pagina (1-indexada; PyPDFLoader la entrega 0-indexada), categoria
    # (nombre de archivo sin extension, legible) y un indice de chunk
    # global para trazabilidad.
    for i, chunk in enumerate(chunks):
        nombre_archivo = os.path.basename(chunk.metadata["source"])
        chunk.metadata["source"] = nombre_archivo
        chunk.metadata["pagina"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["categoria"] = os.path.splitext(nombre_archivo)[0].replace("_", " ")
        chunk.metadata["chunk_id"] = i

    logger.info(
        "%d paginas cargadas de %s -> %d chunks generados",
        len(documentos_crudos), data_dir, len(chunks),
    )
    return chunks


# Modelo de embeddings UNICO para todo el sistema: se usa tanto para indexar
# (ingesta) como para consultar (retriever). EMBEDDING_MODEL vive en
# config.py y se importa siempre desde aca, asi el error #1 de un RAG
# (indexar con un modelo y consultar con otro) no puede pasar.
def _crear_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Genera embeddings y sube los chunks a Pinecone (crea/actualiza los vectores).
# El upsert es una llamada de red que puede fallar de forma transitoria
# (timeout, corte de conexion), asi que se reintenta unas pocas veces con
# un log por intento fallido antes de darse por vencido.
def ingestar(chunks: List[Document], index_name: str = INDEX_NAME, namespace: str = NAMESPACE) -> tuple:
    embeddings = _crear_embeddings()

    vectorstore = None
    intentos = 3
    for intento in range(1, intentos + 1):
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name,
                namespace=namespace,
            )
            break
        except Exception as error:
            if intento == intentos:
                logger.error("Fallaron los %d intentos de subir a Pinecone: %s", intentos, error)
                raise
            espera = 2.0 * intento
            logger.warning(
                "Intento %d/%d de upsert fallo (%s). Reintentando en %.0fs...",
                intento, intentos, error, espera,
            )
            time.sleep(espera)

    logger.info(
        "%d chunks subidos a Pinecone (indice='%s', namespace='%s')",
        len(chunks), index_name, namespace,
    )
    return vectorstore, embeddings


# Se conecta a un indice YA ingestado, sin volver a subir nada. Se usa en
# evaluate.py y main.py: no tiene sentido re-embeber y re-subir los
# documentos cada vez que se quiere consultar o evaluar el sistema.
def conectar_vectorstore(index_name: str = INDEX_NAME) -> tuple:
    embeddings = _crear_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore, embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    chunks = cargar_y_fragmentar()
    print(f"Fragmentos generados: {len(chunks)}")
    print("Metadata de ejemplo:", chunks[0].metadata)
