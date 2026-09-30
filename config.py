# config.py: variables de entorno y constantes centrales del proyecto. Todo
# lo que otro modulo necesita (nombre del indice, namespace, modelo de
# embeddings, tamano de chunk, etc.) se define ACA UNA SOLA VEZ y se importa
# desde aca. Esto evita el error mas comun de la consigna (el "mismatch de
# dimensiones"): EMBEDDING_DIM se fija una sola vez y se reutiliza tanto al
# crear el indice de Pinecone como al generar los embeddings, en vez de
# hardcodear el numero en mas de un lugar.

# os lee las variables de entorno (la API key, el nombre del indice, etc.)
# y arma las rutas a data/ y golden_set.json relativas a este archivo.
import os

# load_dotenv lee el .env y carga esas variables antes de leerlas con os.getenv.
from dotenv import load_dotenv

load_dotenv()

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError(
        "Falta PINECONE_API_KEY en el archivo .env. "
        "Copia .env.example a .env y completa tu API key "
        "(gratis en https://app.pinecone.io)."
    )

INDEX_NAME = os.getenv("INDEX_NAME", "empresaflomat-rag-hibrido")
NAMESPACE = os.getenv("NAMESPACE", "politicas-empresaflomat")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- Embeddings ---
# Modelo local (no requiere API key ni costo): HuggingFace / sentence-transformers.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension real de este modelo -> debe coincidir con el indice

# --- Chunking ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHUNK_SIZE = 600      # tokens, dentro del rango 500-800 sugerido por la consigna
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 5
PESOS_ENSEMBLE = (0.5, 0.5)  # (BM25, vectorial) -> mismo peso para ambas senales

# --- Golden set / evaluacion ---
GOLDEN_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
