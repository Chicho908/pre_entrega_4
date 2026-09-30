# ingest.py: Pipeline de ingesta completo - infraestructura -> carga ->
# chunking -> upsert en Pinecone. Cubre los criterios "Infraestructura Cloud"
# e "Ingesta y Metadatos" de la rubrica. Es un script de proceso (no
# interactivo), asi que usa logging en vez de print: si el namespace ya
# tiene vectores no vuelve a subir nada, salvo que se pida --forzar.
#
# Uso:
#   python ingest.py            -> ingesta solo si el namespace esta vacio
#   python ingest.py --forzar   -> borra el namespace y reingesta desde cero

# argparse lee el flag --forzar desde la terminal.
import argparse

# logging deja registro de cada paso de la ingesta (indice, chunks, subida).
import logging

from config import EMBEDDING_DIM, INDEX_NAME, NAMESPACE
from ingestion import cargar_y_fragmentar, ingestar
from setup_infra import setup_vector_infrastructure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# Silencia el ruido de las librerias de terceros (descargas de HuggingFace, HTTP interno, etc.)
for _ruidoso in ("httpx", "sentence_transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(_ruidoso).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Orquesta el pipeline completo: crea/conecta el indice, y si el namespace
# ya tiene vectores lo salta (a menos que forzar=True, en cuyo caso lo
# vacia antes) -- asi cambiar el corpus de /data no deja vectores viejos
# mezclados con los nuevos.
def main(forzar: bool = False) -> None:
    logger.info("1) Verificando/creando indice '%s' (dimension=%d)...", INDEX_NAME, EMBEDDING_DIM)
    _, indice = setup_vector_infrastructure()

    vectores_existentes = indice.describe_index_stats()["namespaces"].get(NAMESPACE, {}).get("vector_count", 0)
    if vectores_existentes > 0 and not forzar:
        logger.info(
            "El namespace '%s' ya tiene %d vectores: se salta la ingesta. "
            "Usa 'python ingest.py --forzar' para reingestar desde cero.",
            NAMESPACE, vectores_existentes,
        )
        return

    if forzar and vectores_existentes > 0:
        logger.info("2) Borrando los %d vectores existentes del namespace '%s'...", vectores_existentes, NAMESPACE)
        indice.delete(delete_all=True, namespace=NAMESPACE)

    logger.info("3) Cargando y fragmentando documentos de data/...")
    chunks = cargar_y_fragmentar()
    logger.info("   %d fragmentos generados. Metadata de ejemplo: %s", len(chunks), chunks[0].metadata)

    logger.info("4) Generando embeddings y subiendo a Pinecone (namespace='%s')...", NAMESPACE)
    ingestar(chunks)

    stats = indice.describe_index_stats()
    vectores_namespace = stats["namespaces"].get(NAMESPACE, {})
    logger.info("5) Listo. Namespace '%s': %s", NAMESPACE, vectores_namespace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sube los documentos de data/ a Pinecone.")
    parser.add_argument("--forzar", action="store_true", help="Borra el namespace y reingesta desde cero.")
    args = parser.parse_args()
    main(forzar=args.forzar)
