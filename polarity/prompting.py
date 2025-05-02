import google.generativeai as genai
import os
import sys
import pandas as pd
import time
import random
import re
import asyncio

invalid_reviews = 0
api_processed = 0
rule_processed = 0

# --- 1. Configuration ---

def configure_api():
    """Configures the Gemini API key."""
    api_key = "api_key_here"
    try:
        genai.configure(api_key=api_key)
        print("API Key configured.")
        return True
    except Exception as e:
        print(f"Error configuring the API: {e}")
        sys.exit("API Key configuration failed.")


# --- 2. Data Loading ---

def load_data(csv_path, review_column, real_town_column=None):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded from {csv_path} with {len(df)} rows.")
        # Verify if the review column exists
        if review_column not in df.columns:
            raise ValueError(f"The review column '{review_column}' was not found in the CSV.")
        # Verify if the real town column exists (if provided)
        if real_town_column and real_town_column not in df.columns:
            print(
                f"Warning: The real town column '{real_town_column}' was not found. Accuracy evaluation will not be possible.")
            real_town_column = None  # Nullify if it doesn't exist to avoid later errors
        return df, real_town_column
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit("Error loading data.")
    except ValueError as ve:
        print(f"Error: {ve}")
        sys.exit("Error in CSV columns.")
    except Exception as e:
        print(f"Unexpected error loading the CSV: {e}")
        sys.exit("Error loading data.")


# --- 3. Few-Shot Examples ---
# These are examples! You should replace them with real and representative examples from YOUR data.
# Make sure they cover different towns and review styles.
FEW_SHOTS_EXAMPLES = [
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


def get_unique_towns(df, town_column='Town'):
    """
    Extracts all unique towns from a specified column in a CSV dataset.

    Args:
        csv_path (str): The path to the CSV file.
        town_column (str): The name of the column containing the town names.
                           Defaults to 'Town'.

    Returns:
        list: A list of unique town names found in the specified column,
              or None if the file is not found or the column doesn't exist.
              :param df: dataset
              :param town_column: column name with towns
    """
    try:
        if town_column in df.columns:
            unique_towns = df[town_column].unique().tolist()
            return unique_towns
        else:
            print(f"Error: Column '{town_column}' not found in the CSV file.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- 4. Prompt Construction ---

def build_prompt_few_shot(review_to_classify, towns, few_shot_examples):
    """Builds the prompt with instructions and few-shot examples."""
    prompt = """
    Your task is to identify the Mexican Magic Town (Pueblo Mágico) described in the following review in Spanish
    You have to answer just with the magic town you think it is. If the text of the review contains one of the possible
    towns just answer with that town, otherwise answer with the town you guess it is based on the title and the review.
    If the review is very generic and it's impossible to guess the city, just answer with the most famous town in the list.
    These are the possible towns:\n
    """
    for town in towns:
        prompt += town + '\n'

    prompt += "Also consider the following examples to understand the format and the type of expected answer:\n\n"

    for ex in few_shot_examples:
        prompt += f"--- EXAMPLE ---\n"
        prompt += f"Review: \"{ex['review']}\"\n"
        prompt += f"Pueblo Mágico: {ex['town']}\n\n"

    prompt += f"--- REVIEW TO CLASSIFY ---\n"
    prompt += f"Review: \"{review_to_classify}\"\n"
    prompt += f"Pueblo Mágico:"  # We leave this at the end to guide the model

    return prompt


# --- 5. Asynchronous API Call ---

async def guess_town_async(prompt, gemini_model, retries=3, delay=5):
    """Asynchronously calls the Gemini API and extracts the prediction."""
    global api_processed, invalid_reviews
    try:
        model = genai.GenerativeModel(gemini_model)
        for attempt in range(retries):
            try:
                response = await model.generate_content_async(prompt)
                if hasattr(response, 'text'):
                    parts = response.text.split(':')
                    prediction = parts[-1].strip()
                    prediction = prediction.replace('"', '').replace("'", "")
                    prediction = re.sub(r'\.$', '', prediction).strip()
                    api_processed = api_processed + 1
                    return prediction
                else:
                    invalid_reviews = invalid_reviews + 1
                    print(
                        f"Warning: No text received in the response for the prompt. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                    return None
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed. API Error: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    invalid_reviews = invalid_reviews + 1
                    print("Maximum number of retries reached.")
                    return None
        return None
    except Exception as e:
        print(f"Error creating the model or calling the API: {e}")
        return None


# --- 6. Asynchronous Processing ---



async def process_reviews_async(df, towns, title_column, review_column, few_shot_examples, gemini_model, num_reviews=None):
    """Asynchronously iterates over the DataFrame, generates prompts, calls the API, and saves predictions."""
    predictions = [None] * len(df)  # Initialize a list to store predictions

    if num_reviews:
        df_to_process = df.head(num_reviews)
        print(f"Processing the first {num_reviews} reviews asynchronously...")
    else:
        df_to_process = df
        print(f"Processing all {len(df)} reviews asynchronously...")

    async def process_row(index, row):
        current_review = str(row[title_column]) + ' ' + str(row[review_column])
        global invalid_reviews
        global rule_processed
        global api_processed

        # Hardcoded rule, if we find the town in the text we avoid calling the API
        for town in towns:
            if town in current_review:
                rule_processed = rule_processed + 1
                predictions[index] = town
                print(f"Row {index + 1}: Town '{town}' found in review. Skipping API call.")
                return

        if pd.isna(current_review) or not current_review.strip():
            invalid_reviews = invalid_reviews + 1
            print(f"Review in row {index + 1} is empty or invalid. Skipping API call.")
            return

        prompt = build_prompt_few_shot(current_review, towns, few_shot_examples)
        print(f"\n--- Prompt for row {index + 1} ---")
        # print(prompt)
        # print("------")

        await asyncio.sleep(1)  # Optional pause
        prediction = await guess_town_async(prompt, gemini_model)
        predictions[index] = prediction
        # print(f"Row {index + 1}/{len(df_to_process)}: Review processed. Prediction: {prediction}")

    tasks = [process_row(index, row) for index, row in df_to_process.iterrows()]
    await asyncio.gather(*tasks)

    df['gemini_prediction'] = predictions
    return df


# --- 7. Evaluation (Optional) ---

def evaluate_predictions(df_results, real_town_column):
    """Calculates the accuracy if the real values are available."""
    global invalid_reviews, api_processed, rule_processed
    if real_town_column is None or real_town_column not in df_results.columns:
        print("The column with the real town was not provided or does not exist. Accuracy cannot be calculated.")
        return

    correct = (df_results[real_town_column].str.strip().str.lower() ==
               df_results['gemini_prediction'].str.strip().str.lower())

    valid_predictions = df_results['gemini_prediction'].notna().sum()
    correct_valid_predictions = correct[df_results['gemini_prediction'].notna()].sum()

    if valid_predictions > 0:
        accuracy = (correct_valid_predictions / valid_predictions) * 100
        print(f"\n--- Evaluation ---")
        print(f"Total processed reviews with prediction: {valid_predictions}")
        print(f"Failed to guess rows: {invalid_reviews}")
        print(f"Correct predictions: {correct_valid_predictions}")
        print(f"Api guesses: {api_processed}")
        print(f"Rule guesses: {rule_processed}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo valid predictions were obtained for evaluation.")


# --- Main entry point ---
if __name__ == "__main__":
    print("--- Starting Asynchronous Magic Town Review Classification Program ---")

    # --- Configurable Parameters ---
    CSV_PATH = '../data/Rest-Mex_2025_train.csv'  # CHANGE THIS!
    TITLE_COLUMN = 'Title'  # CHANGE THIS to the actual name of your review column!
    REVIEW_COLUMN = 'Review'  # CHANGE THIS to the actual name of your review column!
    REAL_TOWN_COLUMN = 'Town'  # CHANGE THIS to the actual name if you have it, or set to None!
    GEMINI_MODEL = 'gemini-2.5-flash-preview-04-17'  # You can try 'gemini-1.5-pro'
    NUM_REVIEWS_TO_PROCESS = 2_000  # Set to None to process all, or a number for quick testing
    OUTPUT_CSV_PATH = '../results/town_prediction_results_async.csv'  # Optional: to save results

    # 1. Configure API
    if not configure_api():
        sys.exit("API configuration failed.")

    # 2. Load Data
    original_df, REAL_TOWN_COLUMN = load_data(CSV_PATH, REVIEW_COLUMN, REAL_TOWN_COLUMN)
    possible_cities = get_unique_towns(original_df, REAL_TOWN_COLUMN)

    # 3. (Optional) Select Few-Shots
    print(f"Using {len(FEW_SHOTS_EXAMPLES)} few-shot examples.")

    # 4. Process Reviews Asynchronously
    results_df = asyncio.run(process_reviews_async(original_df.copy(),  # Pass a copy to avoid potential side effects
                                                    possible_cities,
                                                    TITLE_COLUMN,
                                                    REVIEW_COLUMN,
                                                    FEW_SHOTS_EXAMPLES,
                                                    GEMINI_MODEL,
                                                    NUM_REVIEWS_TO_PROCESS))

    # 5. Show some results
    print("\n--- First rows with predictions ---")
    print(results_df[[REVIEW_COLUMN, REAL_TOWN_COLUMN, 'gemini_prediction']].head())

    # 6. Evaluate (if applicable)
    if REAL_TOWN_COLUMN:
        evaluate_predictions(results_df, REAL_TOWN_COLUMN)

    # 7. Save Results (Optional)
    try:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nResults saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

    print("\n--- Asynchronous Program Finished ---")