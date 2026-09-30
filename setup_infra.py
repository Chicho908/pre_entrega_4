# setup_infra.py: Infraestructura Cloud - crea (o conecta con) el indice
# Serverless de Pinecone. Cubre el criterio "Configuracion e Infraestructura
# Cloud" de la rubrica: inicializacion correcta del indice serverless +
# manejo seguro de la API key (siempre via variable de entorno, nunca
# hardcodeada en el codigo).

# logging deja registro de si el indice se crea de cero o ya existia.
import logging

# time.sleep espera a que el indice recien creado quede "ready" antes del primer upsert.
import time

# Pinecone es el cliente oficial; ServerlessSpec define la nube/region del indice serverless.
from pinecone import Pinecone, ServerlessSpec

from config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

logger = logging.getLogger(__name__)


# Reintenta una llamada de red a Pinecone si falla (timeout, corte de
# conexion, error transitorio del lado del servicio), con backoff simple y
# un log por cada intento fallido -- para tener visibilidad de que paso
# cuando una llamada externa falla, en vez de que el script corte en seco
# con un traceback sin contexto.
def _con_reintentos(func, *args, intentos: int = 3, espera_base: float = 2.0, **kwargs):
    for intento in range(1, intentos + 1):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            if intento == intentos:
                logger.error("Fallaron los %d intentos contra Pinecone: %s", intentos, error)
                raise
            espera = espera_base * intento
            logger.warning(
                "Intento %d/%d fallo (%s). Reintentando en %.0fs...",
                intento, intentos, error, espera,
            )
            time.sleep(espera)


# Crea el indice si no existe (esperando a que quede 'ready') y lo conecta.
# Devuelve (cliente Pinecone, objeto Index) listos para usar.
def setup_vector_infrastructure(index_name: str = INDEX_NAME, dimension: int = EMBEDDING_DIM) -> tuple:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    indices_existentes = [i["name"] for i in _con_reintentos(pc.list_indexes)]

    if index_name not in indices_existentes:
        logger.info("Creando indice '%s' (dimension=%d, metric=cosine)...", index_name, dimension)
        _con_reintentos(
            pc.create_index,
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # El indice puede tardar unos segundos en quedar listo antes del primer upsert.
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        logger.info("Indice '%s' listo.", index_name)
    else:
        logger.info("El indice '%s' ya existe, no se vuelve a crear.", index_name)

    indice = pc.Index(index_name)
    return pc, indice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _, indice = setup_vector_infrastructure()
    print(indice.describe_index_stats())
