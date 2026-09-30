# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

Modulo de Recuperacion Escalable (Modulo 4, CoderHouse - Desarrollo de
aplicaciones con LLMs). Sube un corpus de documentos tecnicos en PDF a un
indice **Pinecone Serverless** con metadata avanzada (fuente, pagina y
categoria), combina busqueda **vectorial + BM25** en un recuperador
hibrido, y mide su calidad con **Precision@5 / Recall@5** sobre un Golden
Set. La base vectorial vive en la nube (Pinecone Serverless), asi que el
sistema esta pensado para escalar mas alla de una sola maquina.

El corpus son 4 politicas internas en PDF de una empresa de distribucion
ficticia, **EmpresaFlomat** (envios, devoluciones, descuentos por volumen y
vacaciones) escritas para este proyecto — simples y faciles de verificar a
mano, y no son una copia de la guia del curso (ver "Decisiones de diseno").

## Que hace

1. **`config.py`** — punto unico de configuracion: variables de entorno
   (`.env`) y constantes compartidas (`INDEX_NAME`, `NAMESPACE`,
   `EMBEDDING_DIM`, tamano de chunk). Todo lo que otro modulo necesita se
   define aca una sola vez, para evitar el "mismatch de dimensiones" de
   tener el mismo numero hardcodeado en mas de un lugar.
2. **`setup_infra.py`** — `setup_vector_infrastructure()`: crea el
   indice Serverless si no existe (esperando a que quede `ready` antes del
   primer upsert) o se conecta al existente. La API key sale siempre de
   `.env`, nunca hardcodeada.
3. **`ingestion.py`** — `cargar_y_fragmentar()` (`PyPDFLoader` por pagina +
   `RecursiveCharacterTextSplitter`, chunks de 600 tokens con 100 de
   overlap, metadata de `source`/`pagina`/`categoria`/`chunk_id`) e
   `ingestar()` (genera embeddings locales y hace el upsert a Pinecone).
   `conectar_vectorstore()` se conecta a un indice ya ingestado sin volver
   a subir nada.
4. **`rag_system.py`** — `construir_retriever_hibrido()` arma un
   `EnsembleRetriever` (BM25 + Pinecone, pesos 0.5/0.5). La clase
   `RAGSystem` expone `obtener_top_k(query)`, que devuelve los `k`
   fragmentos mas relevantes ya formateados (contenido, fuente, pagina,
   categoria).
5. **`evaluator.py`** — `evaluar(rag_system, golden_set)`: corre cada
   pregunta del Golden Set contra el sistema y calcula Precision@5 y
   Recall@5, por pregunta y en promedio.
6. **`ingest.py`** — script de ingesta end-to-end: infraestructura →
   carga → chunking → embeddings → upsert. Si el namespace ya tiene
   vectores, se salta la ingesta (no reprocesa todo de nuevo); con
   `python ingest.py --forzar` borra el namespace y reingesta desde cero.
   Usa `logging`, porque es un script de proceso.
7. **`evaluate.py`** — script de evaluacion: reconstruye el recuperador
   hibrido (sin volver a subir nada a Pinecone) y corre `evaluar()` sobre
   `golden_set.json`, imprimiendo el reporte en consola. Usa `print()`
   porque el resultado ES el reporte que se entrega, no un log de proceso.
8. **`main.py`** — modo interactivo por consola para consultar el sistema
   a mano. Muestra el contenido completo de cada fragmento recuperado (sin
   recortar ninguna palabra), con su fuente, pagina y categoria; el texto
   se reparte con `textwrap` en lineas cortas con sangria para que se lea
   espaciado en la consola. Usa `print()` porque la salida interactiva ES
   la respuesta que se lee, no un log.

## Como correrlo

```bash
python -m venv venv
venv\Scripts\activate          # en Windows
# source venv/bin/activate     # en Linux/Mac
pip install -r requirements.txt
copy .env.example .env         # completar con tu PINECONE_API_KEY real
python ingest.py               # crea el indice y sube los documentos (una sola vez)
python evaluate.py             # calcula Precision@5 y Recall@5 sobre el golden set
python main.py                 # modo interactivo
```

Si ya corriste `ingest.py` antes con un `INDEX_NAME`/corpus distinto (por
ejemplo, una version anterior de este repo) y queres reemplazar los
vectores viejos por los nuevos en el mismo indice, corre
`python ingest.py --forzar` en vez de `python ingest.py`. Si preferis
arrancar de cero sin tocar el indice viejo, alcanza con dejar el
`INDEX_NAME` nuevo (`empresaflomat-rag-hibrido`) del `.env.example`: se crea un
indice aparte y el viejo queda intacto (se puede borrar a mano desde
[app.pinecone.io](https://app.pinecone.io) si ya no hace falta).

`PINECONE_API_KEY` es la key que efectivamente usa el codigo — es gratis
en [app.pinecone.io](https://app.pinecone.io). Los embeddings se generan
localmente, asi que no hace falta una key de OpenAI ni de Anthropic para
que el sistema funcione (ver "Decisiones de diseno"); el `.env.example`
igual incluye `OPENAI_API_KEY` vacia porque la consigna la pide
explicitamente en el paso 1 del entregable.

La primera vez, `ingest.py` descarga el modelo de embeddings local
(~90 MB) y crea el indice en Pinecone; las siguientes corridas detectan
que el indice ya existe y no lo vuelven a crear.

## Estructura del repositorio

```
.
├── data/                              # corpus propio: 4 politicas de EmpresaFlomat (PDF)
│   ├── politica_envios.pdf
│   ├── politica_devoluciones.pdf
│   ├── politica_descuentos.pdf
│   └── politica_vacaciones.pdf
├── config.py                          # variables de entorno + constantes compartidas
├── setup_infra.py                     # setup_vector_infrastructure() (indice Pinecone)
├── ingestion.py                       # carga + chunking + embeddings + upsert
├── rag_system.py                      # RAGSystem + EnsembleRetriever (BM25 + Pinecone)
├── evaluator.py                       # evaluar(): Precision@5 / Recall@5
├── golden_set.json                    # 5 preguntas con su documento_id_esperado
├── ingest.py                          # script de ingesta end-to-end
├── evaluate.py                        # script de evaluacion (Precision@5 / Recall@5)
├── main.py                            # modo interactivo por consola
├── requirements.txt                   # dependencias con versiones fijadas
├── .env.example                       # plantilla de variables de entorno (sin key real)
└── .gitignore                         # excluye .env, venv/ y __pycache__/
```

Estructura plana, sin carpeta `src/`.

## Ejemplo de salida esperada

`evaluate.py` corre las 5 preguntas del golden set contra las 4 politicas
de EmpresaFlomat (una pagina cada una, 4 chunks en total). Con un corpus tan
chico y `top_k=5`, se recuperan los 4 documentos disponibles en cada
consulta — por eso el patron esperado es **Recall@5 = 100%** (el documento
correcto siempre esta entre los recuperados) y **Precision@5 = 25%** (1 de
cada 4 recuperados es el relevante). Esto no es un error: es el mismo
patron que se ve en el golden set de ejemplo del Modulo 4, y es intencional
— el corpus se mantuvo chico y simple a proposito, para que las respuestas
sean faciles de verificar a mano.

```
[OK] ¿Cuántos días de vacaciones corresponden a un empleado con 7 años de antigüedad?
        Esperado: politica_vacaciones.pdf | Recuperados: ['politica_vacaciones.pdf', 'politica_devoluciones.pdf', ...]
        Recall@5: 100% | Precision@5: 25%

...

================================================================================
RECALL@5 PROMEDIO:    100.0%
PRECISION@5 PROMEDIO: 25.0%
================================================================================
```

## Decisiones de diseno

- **Embeddings locales (`sentence-transformers/all-MiniLM-L6-v2`) en vez de
  OpenAI**: no requieren otra API key ni tienen costo por uso — solo hace
  falta `PINECONE_API_KEY`. Su dimension (384) se define una sola vez en
  `config.py` y se reutiliza tanto al crear el indice como al generar los
  vectores, para no repetir el error de mismatch de dimensiones que
  menciona la consigna. `OPENAI_API_KEY` esta en `.env.example` porque el
  paso 1 del entregable la pide explicitamente, pero ningun modulo la lee
  (no hace falta completarla para que el sistema funcione).
- **`EMBEDDING_DIM` como constante unica**: si mañana se cambia de modelo de
  embeddings, se cambia en un solo lugar (`config.py`) y tanto el indice
  como la ingesta quedan consistentes automaticamente.
- **Esperar a que el indice quede `ready`**: `setup_vector_infrastructure()`
  no asume que el indice esta listo apenas se crea — sondea
  `describe_index(name).status["ready"]` antes del primer upsert, evitando
  una condicion de carrera real de Pinecone Serverless.
- **`conectar_vectorstore()` separado de `ingestar()`**: `evaluate.py` y
  `main.py` no necesitan volver a generar embeddings ni re-subir nada cada
  vez que se corren — solo `ingest.py` escribe en Pinecone.
- **Namespace (`politicas-empresaflomat`)**: aisla estos datos de cualquier otro
  dato que se suba mas adelante al mismo indice (multi-tenancy / separar
  entornos), tal como pide la consigna explicitamente ("ignorar el
  namespace" es uno de los errores a evitar). Se usa igual al subir
  (`ingestar()`) y al consultar (`construir_retriever_hibrido()`), asi que
  nunca hay una busqueda que lo ignore.
- **`ingest.py --forzar`**: como el namespace no se resetea solo, cambiar
  los documentos de `data/` y correr `ingest.py` de nuevo dejaria los
  vectores viejos mezclados con los nuevos. `--forzar` borra el namespace
  completo antes de reingestar.
- **Corpus propio en PDF (politicas de EmpresaFlomat, una empresa ficticia) en vez
  de copiar el corpus de la guia del curso**: la consigna pide PDF como uno
  de los formatos validos para la ingesta. Se eligio un dominio simple
  (envios, devoluciones, descuentos, vacaciones) en vez de documentacion
  tecnica de una libreria para que las preguntas y sus respuestas sean
  faciles de entender y verificar a mano, sin perder generalidad: son
  politicas internas con datos concretos (dias, porcentajes, montos), igual
  de validas como "documentos tecnicos" que la consigna menciona.
- **`PyPDFLoader` en vez de un loader de texto plano**: carga un `Document`
  por pagina, con el numero de pagina ya en la metadata (`page`,
  0-indexada). De ahi sale el campo `pagina` (1-indexada) que pide la
  consigna en la metadata avanzada ("fuente, pagina, etiquetas de
  categoria") — no es un dato inventado, es la pagina real del PDF de la
  que salio cada chunk.
- **Metadata con el texto incluido**: `PineconeVectorStore` guarda el
  `page_content` dentro de la metadata de cada vector (clave `text`) —
  evita depender de una base de datos relacional aparte para mostrar el
  contenido recuperado.
- **`EnsembleRetriever` con pesos 0.5/0.5**: punto de partida neutral entre
  la señal lexica (BM25) y la semantica (Pinecone). Si se detecta que un
  tipo de pregunta rinde mejor con una señal que con la otra (por ejemplo,
  siglas exactas como "2FA" favorecen a BM25), es el parametro a ajustar
  primero (`PESOS_ENSEMBLE` en `config.py`).
- **Tokenizador propio para BM25 (`_tokenizar` en `rag_system.py`)**: el
  `preprocess_func` por defecto de `BM25Retriever` es literalmente
  `texto.split()` (sin minusculas, sin sacar tildes ni puntuacion), asi que
  una consulta real como "como son los envios" no matcheaba con "envío" o
  "Envíos:" en el documento y la mitad lexica del hibrido casi no aportaba.
  `_tokenizar` normaliza (minusculas + sin tildes) antes de tokenizar, para
  que BM25 compare palabras de verdad.
- **Reintentos con logging en las llamadas a Pinecone**: tanto
  `setup_vector_infrastructure()` (`setup_infra.py`) como `ingestar()`
  (`ingestion.py`) reintentan unas pocas veces con backoff si la llamada de
  red falla de forma transitoria, dejando un log por cada intento fallido
  en vez de cortar en seco con un traceback sin contexto.
- **`chunk_size=600` / `chunk_overlap=100`**: dentro del rango 500-800
  tokens que sugiere la consigna, medidos en tokens reales
  (`from_tiktoken_encoder`) y no en caracteres.
- **Golden Set con un solo documento esperado por pregunta**: por eso
  Recall@5 solo puede dar 0 o 1 (no hay termino medio matematicamente
  posible con un unico documento relevante) — no es una limitacion de la
  metrica, es consecuencia de como esta armado este golden set en
  particular.
- **Versiones fijadas en `requirements.txt`**: las mismas probadas en este
  entorno, para que el repo sea reproducible.
- **`.env` fuera del repo**: la API key real nunca se sube (esta en
  `.gitignore`); solo se versiona `.env.example` como plantilla.
- **Estructura plana y comentarios linea por linea**: sin carpeta `src/`
  (todos los `.py` en la raiz) y con un comentario arriba de cada import y
  de cada funcion/clase explicando el porque. `ingest.py` usa `logging`
  porque es un script de proceso; `evaluate.py` y `main.py` usan `print()`
  porque su salida es el resultado que se lee (un reporte o una respuesta
  interactiva), no un log.
