# evaluator.py: Evaluacion Cuantitativa de Metricas - Precision@5 y Recall@5
# sobre un Golden Set de preguntas con su documento_id_esperado conocido de
# antemano (golden_set.json).

# Tipos para documentar que recibe y que devuelve evaluar().
from typing import Dict, List

from rag_system import RAGSystem


# Corre cada pregunta del golden set contra el sistema y calcula Precision@k
# y Recall@k (k = rag_system.k) por pregunta y en promedio.
def evaluar(rag_system: RAGSystem, golden_set: List[Dict]) -> Dict:
    resultados_por_pregunta = []

    for caso in golden_set:
        top_k = rag_system.obtener_top_k(caso["pregunta"])
        fuentes_recuperadas = [r["fuente"] for r in top_k]

        # Recall@k: 1.0 si el documento esperado aparece entre los recuperados, si no 0.0.
        recall = 1.0 if caso["documento_id_esperado"] in fuentes_recuperadas else 0.0

        # Precision@k: que porcentaje de los recuperados coincide con la fuente esperada.
        coincidencias = sum(1 for f in fuentes_recuperadas if f == caso["documento_id_esperado"])
        precision = coincidencias / len(top_k) if top_k else 0.0

        resultados_por_pregunta.append(
            {
                "pregunta": caso["pregunta"],
                "esperado": caso["documento_id_esperado"],
                "recuperados": fuentes_recuperadas,
                "recall@5": recall,
                "precision@5": precision,
            }
        )

    recall_promedio = sum(r["recall@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)
    precision_promedio = sum(r["precision@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)

    return {
        "detalle": resultados_por_pregunta,
        "recall@5_promedio": recall_promedio,
        "precision@5_promedio": precision_promedio,
    }
