# main.py: modo interactivo por consola para probar el sistema RAG hibrido
# (BM25 + Pinecone) a mano. Corre "python ingest.py" una vez antes para
# poblar el indice.
#
# Uso:
#   python main.py

# textwrap.fill reparte el contenido completo de cada fragmento en lineas
# cortas y con sangria, para que se lea espaciado en la consola en vez de
# un bloque de texto pegado (sin recortar ni una palabra del contenido).
import textwrap

from ingestion import cargar_y_fragmentar, conectar_vectorstore
from rag_system import RAGSystem, construir_retriever_hibrido

ANCHO_CONSOLA = 88


# Loop de preguntas: arma el retriever hibrido una sola vez y despues, por
# cada pregunta que escribe el usuario, muestra el top-k de fragmentos
# recuperados con su fuente, hasta que escriba "salir".
def main() -> None:
    print("Cargando documentos y conectando con Pinecone...")
    chunks = cargar_y_fragmentar()
    vectorstore, _ = conectar_vectorstore()
    retriever_hibrido = construir_retriever_hibrido(chunks, vectorstore)
    rag_system = RAGSystem(retriever_hibrido)

    print("\nSistema RAG hibrido (BM25 + Pinecone). Escribi 'salir' para terminar.\n")

    while True:
        pregunta = input("Vos: ").strip()

        if pregunta.lower() in ("salir", "exit", "quit", ""):
            print("\nListo, terminamos la sesion.")
            break

        resultados = rag_system.obtener_top_k(pregunta)

        if not resultados:
            print("No se recupero ningun fragmento.\n")
            continue

        print(f"\nTop {len(resultados)} fragmentos recuperados:")
        for i, r in enumerate(resultados, 1):
            # Contenido completo del chunk (sin recortar ninguna palabra),
            # pero repartido en lineas cortas con sangria para que se lea
            # espaciado en la consola en vez de un bloque de texto pegado.
            texto_formateado = textwrap.fill(
                r["contenido"],
                width=ANCHO_CONSOLA,
                initial_indent="     ",
                subsequent_indent="     ",
            )
            print(f"\n  {i}. [{r['fuente']} (pag. {r['pagina']}) | {r['categoria']}]\n")
            print(texto_formateado)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
