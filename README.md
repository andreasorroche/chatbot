# Chatbot

Este chatbot implementado con LangChain tiene las siguientes funcionalidades:
- Interfaz de chatbot con Streamlit
- Uso como base de conocimiento de un RAG varios documentos PDF largos (un libro de Física y otro de Astronomía)
, utilizando una base de datos vectorial (FAISS).
- Memoria dinámica que mantiene la conversación y cuando esta pasa de X tokens se resume de forma automática.

## Prerrequisitos
- Python
- Docker compose

## Uso
En la carpeta config se encuentra un archivo settings en el que se pueden modificar dos parámetros. Por un lado se puede 
especificar el modelo que se quiere utilizar para el chat (por defecto está gpt-4) y el máximo de tokens a partir
del cual se resume la memoria.
Debes añadir un archivo .env en la carpeta docker con la variable de entorno tal y como se especifica
en el archivo template.env.
Entonces, desde el directorio de docker debes lanzar el comando:

docker compose up --build


# Puntos teóricos

1. Diferencias entre 'completion' y 'chat' models.
   
Los modelos de 'completion' generan texto en base a un prompt, mientras que los modelos de 'chat' están optimizados
para mantener una conversación con contextos de turno.

2. ¿Cómo forzar a que el chatbot responda 'sí' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?

Dándole la orden en el prompt de que únicamente responda con 'sí' o 'no' forzaría al chatbot a responder de esa forma.
Para parsear la salida y que siga un formato determinado se puede utilizar Output Parsers de LangChain. De esta forma puedes
generar cualquier forma de datos estructurados. Hay muchos tipos de Output Parsers en LangChain, entre ellos JSON y
Pydantic, y muchos de ellos admiten streaming.

3. Ventajas e inconvenientes de RAG vs fine-tunning.
   
Entre las ventajas del RAG podemos encontrar:
- Mayor precisión ya que tiene acceso a base de conocimiento externo.
- Reducción de alucinaciones y sesgos al acceder a fuentes con información factual.
- Adaptabilidad a nuevos datos fácilmente.
- Identificación de la fuente de la respuesta del modelo, dando mayor calidad y confianza.
- Requiere menos datos etiquetados y recursos en comparación con el fine-tuning.
  
Sin embargo, también tenemos los siguientes inconvenientes de RAG:
- Menor personalización del modelo, ya que, aunque incorpore información externa, puede no personalizar completamente 
el estilo de escritura o el comportamiento del modelo.
- Puede estar más limitado por la información que recupera, en comparación con los modelos completamente generativos.
- Mayor complejidad por tener que gestionar el modelo generador y el componente de recuperación, en lugar de únicamente
cargar el modelo.

Por otro lado, encontramos las siguientes ventajas del fine-tuning:
- Mayor rendimiento en tareas específicas en las que se entrene.
- Modelos más creativos y flexibles ya que no se cierran tanto a contestar respecto a la información extra en el caso de RAG.
- Implementación más sencilla ya que implica menos componentes, facilitanto la integración.
  
Los inconvenientes del fine-tuning son:
- Se requiere de más datos etiquetados y recursos computacionales para entrenar el modelo.
- Los modelos fine-tuning son más estáticos que RAG, por lo que no se pueden adaptar a datos cambiantes.
- Puede ser menos transparente en cuanto al proceso de ajuste de parámetros y pesos del modelo a diferencia de RAG.
  
4. ¿Cómo evaluar el desempeño de un bot Q&A?¿Cómo evaluar el desempeño de un RAG?
   
Para evaluar el desempeño de un bot Q&A, se pueden considerar varios enfoques. Uno de los 
métodos comunes implica la creación de un conjunto de datos de validación que contiene preguntas representativas y 
respuestas esperadas. Este conjunto de datos se puede generar manualmente o automáticamente utilizando técnicas como 
el uso de modelos de lenguaje para generar respuestas de referencia. Luego, se pueden aplicar métricas de 
evaluación, como ROUGE, BLEU, F1-score, entre otras, para comparar las respuestas generadas por el bot con las 
respuestas esperadas en el conjunto de datos de validación. Además, se pueden llevar a cabo pruebas de usuario y 
análisis de retroalimentación para evaluar la precisión, la coherencia y la utilidad percibida del bot en 
situaciones del mundo real. La combinación de técnicas automatizadas y evaluaciones humanas proporciona una 
visión holística del desempeño del bot Q&A y ayuda a identificar áreas de mejora. 
En el caso de un RAG, además de evaluar el componente de generación también hay que evaluar la recuperación de 
información de su conocimiento extra. Esto se logra mediante la comparación de las respuestas generadas por el modelo 
con el contenido del conocimiento base, como documentos o corpus de texto. Se utilizan métricas de similitud y 
relevancia para determinar qué tan bien se alinean las respuestas generadas con la información presente en el conocimiento base. 
