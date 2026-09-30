# evaluate.py: Script de Evaluacion - calcula Precision@5 y Recall@5 sobre
# el Golden Set. Cubre el criterio "Evaluacion Cuantitativa de Metricas" de
# la rubrica. A diferencia de ingest.py, esto no es un log de proceso sino
# el REPORTE final que se entrega (igual que main.py), asi que usa print()
# en vez de logging para que quede prolijo y facil de leer en la consola.
#
# Requiere haber corrido "python ingest.py" antes al menos una vez (este
# script se conecta al indice ya ingestado, no vuelve a subir nada).
#
# Uso:
#   python evaluate.py

# json carga el golden set de preguntas con su documento_id_esperado.
import json

from config import GOLDEN_SET_PATH
from evaluator import evaluar
from ingestion import cargar_y_fragmentar, conectar_vectorstore
from rag_system import RAGSystem, construir_retriever_hibrido


# Reconstruye el retriever hibrido (BM25 necesita los chunks en memoria de
# nuevo; el vectorstore se reconecta al indice ya ingestado) y corre el
# golden set completo, imprimiendo el detalle por pregunta y el promedio final.
def main() -> None:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    print("Reconstruyendo el retriever hibrido (BM25 local + Pinecone ya ingestado)...")
    chunks = cargar_y_fragmentar()  # BM25 es en memoria: necesita los chunks de nuevo
    vectorstore, _ = conectar_vectorstore()  # se conecta al indice, no re-sube nada

    retriever_hibrido = construir_retriever_hibrido(chunks, vectorstore)
    rag_system = RAGSystem(retriever_hibrido)

    print(f"Evaluando {len(golden_set)} preguntas del golden set...\n")
    reporte = evaluar(rag_system, golden_set)

    print("=" * 80)
    for r in reporte["detalle"]:
        estado = "OK" if r["recall@5"] == 1.0 else "FALLO"
        print(f"[{estado}] {r['pregunta']}")
        print(f"        Esperado: {r['esperado']} | Recuperados: {r['recuperados']}")
        print(f"        Recall@5: {r['recall@5']:.0%} | Precision@5: {r['precision@5']:.0%}\n")

    print("=" * 80)
    print(f"RECALL@5 PROMEDIO:    {reporte['recall@5_promedio']:.1%}")
    print(f"PRECISION@5 PROMEDIO: {reporte['precision@5_promedio']:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
