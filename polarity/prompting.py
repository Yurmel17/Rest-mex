import os

from common.prompting_utils import *

def build_prompt_few_shot(review_to_classify):
    """Builds the prompt with instructions and few-shot examples."""
    few_shot_examples = [
        {
            "review": "Mi Lugar Favorito!!!!, Excelente lugar para comer y pasar una buena noche!!! El servicio es de primera y la comida exquisita!!!",
            "polarity": "5.0"
        },
        {
            "review": "Bonito lugar, Las olas son preciosas, playa. Mejor ciudad para ir de compras y caminar, Punta de Mita. Un montÃ³n de vendedores y puedes conseguir un mensaje en la playa. No soy un surfista pero nos gustÃ³ ver surf de otros.",
            "polarity": "4.0"
        },
        {
            "review": "Gran ubicación! El servicio es lento, muy lento,Lugar de primera, pero 10 minutos de tomar nuestro pedido, 10 minutos para cada ronda de cervezas, y a 20 minutos de la hora después de que pedí, era lentísimo para dar una mejor clasificación.Sólo tuvimos 2 cervezas cada uno, pero disfrutamos el lugar!",
            "polarity": "3.0"
        },
        {
            "review": """No apto para cardíacos ,"La entrada tiene un costo de $70 a menos que seas estudiante, maestro o menor de 12 años es gratis. Para llegar hasta las pirámides nosotros tomamos los autobuses Teotihuacanos en la central del norte. El costo fue de $104 por persona por el viaje redondo. Las pirámides están increíbles. La del sol la puedes subir hasta lo más alto. La de la luna hasta la mitad. Las vistas son increíbles y caminar por la calzada de los muertos es una experiencia padrísima. La zona está muy bien cuidado y restaurada. Vale muchísimo la pena. Mi recomendación es que lleven su propia agua ya que no hay nada de sombra y es súper cansado estar subiendo y bajando las pirámides, dentro del lugar los costos de agua son un poco más altos. (Un powerade de 1L estaba en $40) justo a la salida hay otra tiendita y el powerade de litro nos salió en $30. Lleven gorra, o sombrero o pueden comprar uno a la entrada, hay de muchos precios y gustos. """,
            "polarity": "2.0"
        },
        {
            "review": "Lejos de este sitio,lugar horrible. Ten cuidado, en lugares como México, con buenos granos de café, de lugares como el Café de Yara. No sabía nada de café (nada).  Todavía están sirviendo versión de 1980 de un mexicano capuchino (horrible expreso + Condensated y leche sobrecalentadas + +...Canela en la parte de arriba de cristal).  En general, alojarse lejos de lugares como éste en México o tendrás una idea equivocada sobre nuestro café",
            "polarity": "1.0"
        }
    ]
    prompt = """
    Your task is to identify the polarity of the following review in Spanish
    You have to answer just with the polarity you think it has. Polarity goes from 1.0 very bad, to 5.0 very positive.\n
    The possible values are 1.0, 2.0, 3.0, 4.0 and 5.0.
    """

    prompt += "Also consider the following examples to understand the format and the type of expected answer:\n\n"

    for ex in few_shot_examples:
        prompt += f"--- EXAMPLE ---\n"
        prompt += f"Review: \"{ex['review']}\"\n"
        prompt += f"Polarity: {ex['polarity']}\n\n"

    prompt += f"--- REVIEW TO CLASSIFY ---\n"
    prompt += f"Review: \"{review_to_classify}\"\n"
    prompt += f"Polarity:"  # We leave this at the end to guide the model

    return prompt


# --- Main entry point ---
if __name__ == "__main__":
    # --- Configurable Parameters ---
    prompting_parameters = PromptingParameters(
        data_path='../data/Rest-Mex_2025_train.csv',
        real_column='Polarity',
        model='gemini-2.5-flash-preview-04-17',
        reviews=2000,
        output='../results/Rest-Mex_2025_test_results_Polarity_prompting.csv',
        prompt_builder=build_prompt_few_shot,
        api_key= os.environ.get('API_KEY'),
        ratio=1500
    )

    # --- Run prompting ---
    prompting(prompting_parameters)
