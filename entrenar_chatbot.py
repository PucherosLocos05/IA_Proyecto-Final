import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Dataset simple de intenciones
intents = [
    {"tag": "saludo", "patterns": ["hola", "buenas", "buenos dias", "qué onda", "hola buenas"]},
    {"tag": "despedida", "patterns": ["adios", "hasta luego", "nos vemos", "gracias, adios"]},
    {"tag": "precio", "patterns": ["cuanto cuestan las donas", "precio de las donas", "precios", "cuanto vale una dona", "cuanto cuestan"]},
    {"tag": "menu", "patterns": ["que sabores tienes", "menu de donas", "sabores de donas", "que venden", "quiero ver el menú"]},
    {"tag": "horario", "patterns": ["cual es su horario", "a que hora abren", "a que hora cierran", "horarios", "cuando estan abiertos"]},
    {"tag": "ubicacion", "patterns": ["donde estan", "cuál es su ubicacion", "direccion de la tienda", "en donde se encuentran"]}
]

# 2. Construir listas de frases (X) y etiquetas (y)
frases = []
etiquetas = []
tags = []

for intent in intents:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        frases.append(pattern)
        etiquetas.append(tag)

# 3. Codificar etiquetas a números
tags_unicos = sorted(list(set(tags)))
tag_a_id = {tag: idx for idx, tag in enumerate(tags_unicos)}
id_a_tag = {idx: tag for tag, idx in tag_a_id.items()}

y = np.array([tag_a_id[e] for e in etiquetas])

# 4. Tokenizar texto
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(frases)
sequences = tokenizer.texts_to_sequences(frases)

maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen, padding="post")

vocab_size = len(tokenizer.word_index) + 1  # +1 por OOV

print("Frases de entrenamiento:", len(frases))
print("Tamaño vocabulario:", vocab_size)
print("Longitud máxima de secuencia:", maxlen)
print("Clases:", tags_unicos)