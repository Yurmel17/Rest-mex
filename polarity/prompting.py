import google.generativeai as genai
import os
import sys
import pandas as pd
import time
import random
import re
import asyncio
from common.prompting_utils import *

invalid_reviews = 0
api_processed = 0
rule_processed = 0


def build_prompt_few_shot(review_to_classify, polarities):
    """Builds the prompt with instructions and few-shot examples."""
    few_shot_examples = [
        {
            "review": "Mi Lugar Favorito!!!!, Excelente lugar para comer y pasar una buena noche!!! El servicio es de primera y la comida exquisita!!!",
            "town": "Sayulita"
        },
        {
            "review": "Lugar impresionante en la playa,Fui con mis suegros durante una semana alrededor de la víspera de año nuevo y me encantó! Soy bastante particular viajero - tengo el listón muy alto, pero no soy un esnob, jajaja. Las habitaciones están genial; la playa es genial, la comida es excelente, el personal es muy agradable, aunque algunos de los chicos en el restaurante/bar lento, pero las señoras mantenga presionada la fortaleza. No puedo decir lo mucho que me encantó el desayuno y el almuerzo, opciones como las bebidas!  El secreto parece ser, pero espero poder conseguir una habitación aquí la próxima vez que venga a Tulum!",
            "town": "Tulum"
        },
        {
            "review": "La mejor comida en Isla Mujeres,Sergio siempre tiene los mejores filetes y platos de langosta en esta hermosa isla. Sus acompañamientos son excelentes y su presentación y la increíble carne que ofrece son las mejores. Sus precios son muy razonables y el servicio impecable. Desde sus ofrendas de langosta hasta...pescado, camarones y bistec, todos son increíbles. No olvides el postre, el vino y los cócteles. Una noche que te costaría fácilmente entre 250 y 300 en los estados, te costará alrededor de 125 aquí. ¡Él saca un producto increíble!Más",
            "town": "Isla_Mujeres"
        },
        {
            "review": """No apto para cardíacos ,"La entrada tiene un costo de $70 a menos que seas estudiante, maestro o menor de 12 años es gratis. Para llegar hasta las pirámides nosotros tomamos los autobuses Teotihuacanos en la central del norte. El costo fue de $104 por persona por el viaje redondo. Las pirámides están increíbles. La del sol la puedes subir hasta lo más alto. La de la luna hasta la mitad. Las vistas son increíbles y caminar por la calzada de los muertos es una experiencia padrísima. La zona está muy bien cuidado y restaurada. Vale muchísimo la pena. Mi recomendación es que lleven su propia agua ya que no hay nada de sombra y es súper cansado estar subiendo y bajando las pirámides, dentro del lugar los costos de agua son un poco más altos. (Un powerade de 1L estaba en $40) justo a la salida hay otra tiendita y el powerade de litro nos salió en $30. Lleven gorra, o sombrero o pueden comprar uno a la entrada, hay de muchos precios y gustos. """,
            "town": "Teotihuacan"
        },
        {
            "review": """Qué bonito lugar, me encanto","Qué bonito lugar, me encanto hoy que fui, tienen excelentes medidas de seguridad para cuidarnos todos. Me sentí demasiado bien, ya que tenía meses sin salir, se los recomiendo demasiado. Les recomiendo los delfin empanizados. Y su nueva remodelación! No se diga maravillosa. :)""",
            "town": "Ixtapan_de_la_Sal"
        }
    ]
    prompt = """
    Your task is to identify the polarity of the following review in Spanish
    You have to answer just with the polarity you think it is.  \n
    """

    prompt += "Also consider the following examples to understand the format and the type of expected answer:\n\n"

    for ex in few_shot_examples:
        prompt += f"--- EXAMPLE ---\n"
        prompt += f"Review: \"{ex['review']}\"\n"
        prompt += f"Pueblo Mágico: {ex['town']}\n\n"

    prompt += f"--- REVIEW TO CLASSIFY ---\n"
    prompt += f"Review: \"{review_to_classify}\"\n"
    prompt += f"Pueblo Mágico:"  # We leave this at the end to guide the model

    return prompt


# --- Main entry point ---
if __name__ == "__main__":
    # --- Configurable Parameters ---
    prompting_parameters = PromptingParameters(
        data_path='../data/Rest-Mex_2025_train.csv',
        real_column='Polarity',
        model='gemini-2.5-flash-preview-04-17',
        reviews=2000,
        output='../results/town_prediction_results_async.csv',
        prompt_builder=build_prompt_few_shot
    )

    # --- Run prompting ---
    prompting(prompting_parameters)
