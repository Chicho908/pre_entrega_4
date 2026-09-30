# rag_system.py: Recuperador Hibrido - combina BM25 (lexico) con la busqueda
# vectorial de Pinecone mediante un EnsembleRetriever (que internamente
# fusiona los rankings con RRF, Reciprocal Rank Fusion). BM25 aporta en
# terminos tecnicos, siglas y nombres propios ("2FA") que un embedding puede
# diluir; la busqueda vectorial aporta en parafraseos y sinonimos ("dias de
# descanso" ~ "vacaciones"). El EnsembleRetriever combina ambos rankings por
# posicion, no por score, porque BM25 y la similitud coseno no comparten escala.

# re separa la query/documentos en palabras, ignorando puntuacion.
import re

# unicodedata.normalize permite sacar las tildes para que el tokenizador de
# BM25 no distinga "envio" de "envío" (ver _tokenizar mas abajo).
import unicodedata

# Tipos para documentar que recibe y que devuelve cada metodo.
from typing import Dict, List

# EnsembleRetriever fusiona varios retrievers en un unico ranking (RRF).
from langchain_classic.retrievers import EnsembleRetriever

# BM25Retriever es la mitad lexica del hibrido (ranking por keywords, en memoria).
from langchain_community.retrievers import BM25Retriever

# Document es el tipo que devuelven los retrievers de LangChain.
from langchain_core.documents import Document

from config import NAMESPACE, PESOS_ENSEMBLE, TOP_K


# El preprocess_func por defecto de BM25Retriever es literalmente
# "texto.split()": sin minusculas, sin sacar tildes ni puntuacion. Eso hace
# que "envios" (como lo escribe un usuario real) nunca matchee con "envío",
# "Envíos" o "envíos:" en el documento, y la mitad lexica del hibrido casi
# no aporte nada. Esta funcion normaliza (minusculas + sin tildes) antes de
# tokenizar, para que BM25 compare palabras de verdad, no strings exactos.
def _tokenizar(texto: str) -> List[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"\w+", texto)


# Encapsula el retriever hibrido ya armado y expone un metodo simple para
# pedir el top-k de una consulta, con el formato que despues usan
# evaluate.py y main.py.
class RAGSystem:
    def __init__(self, retriever, k: int = TOP_K):
        self.retriever = retriever
        self.k = k

    # Devuelve los k fragmentos mas relevantes para la query, con su fuente,
    # pagina y categoria (metadata), sin exponer el objeto Document de
    # LangChain hacia afuera.
    def obtener_top_k(self, query: str) -> List[Dict]:
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "pagina": d.metadata.get("pagina", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
            }
            for d in docs
        ]


# Arma el EnsembleRetriever a partir de los chunks (para BM25, que vive en
# memoria) y el vectorstore de Pinecone (para la busqueda semantica).
def construir_retriever_hibrido(
    chunks: List[Document],
    vectorstore,
    k: int = TOP_K,
    weights: tuple = PESOS_ENSEMBLE,
    namespace: str = NAMESPACE,
) -> EnsembleRetriever:
    retriever_bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenizar)
    retriever_bm25.k = k

    retriever_vectorial = vectorstore.as_retriever(
        search_kwargs={"k": k, "namespace": namespace},
    )

    return EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vectorial],
        weights=list(weights),
    )
