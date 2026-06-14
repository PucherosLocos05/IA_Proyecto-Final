
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# 1. Cargar la base de conocimientos y el modelo entrenado
# Nota: Si tus archivos .pkl o .h5 tienen nombres ligeramente distintos, cámbialos aquí
intents = json.loads(open('intents_donas.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def limpiar_texto(texto):
    palabras_texto = nltk.word_tokenize(texto)
    palabras_texto = [lemmatizer.lemmatize(palabra.lower()) for palabra in palabras_texto]
    return palabras_texto

def bolsa_palabras(texto, palabras):
    palabras_texto = limpiar_texto(texto)
    bolsa = [0] * len(palabras)
    for s in palabras_texto:
        for i, palabra in enumerate(palabras):
            if palabra == s:
                bolsa[i] = 1
    return np.array(bolsa)


def predecir_intencion(texto):
    bolsa = bolsa_palabras(texto, words)
    res = model.predict(np.array([bolsa]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    resultados = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    
    if resultados:
        return_list = []
        for r in resultados:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        tag = return_list[0]['intent']
        conf = float(return_list[0]['probability'])
        return tag, conf
    else:
        return "no_entendido", 0.0

def obtener_respuesta(tag, intents_json):
    lista_intents = intents_json['intents']
    for i in lista_intents:
        if i['tag'] == tag:
            resultado = random.choice(i['responses'])
            break
    return resultado

def iniciar_chat():
    print("Chatbot Donas CETI (Deep Learning) 🍩")
    print("Escribe 'salir' para terminar.\n")
    while True:
        msg = input("Tú: ")
        if msg.lower() == 'salir':
            print("Bot: ¡Adiós! Vuelve pronto por más donas. 🍩")
            break
        
     
        tag, conf = predecir_intencion(msg)
        
        if conf > 0.5 and tag != "no_entendido":
            respuesta = obtener_respuesta(tag, intents)
            print(f"Bot: {respuesta}\n")
        else:
            print("Bot: Lo siento, no entendí tu pregunta. ¿Podrías repetirla de otra forma?\n")

if __name__ == "__main__":
    iniciar_chat()