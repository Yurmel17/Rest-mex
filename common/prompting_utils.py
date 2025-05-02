import math

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

class PromptingParameters:
    TITLE_COLUMN = 'Title'
    REVIEW_COLUMN = 'Review'

    def __init__(self, data_path, real_column, model, reviews, output, prompt_builder):
        self.csv_path = data_path  # CHANGE THIS!
        self.column = real_column  # CHANGE THIS to the actual name if you have it, or set to None!
        self.model = model  # You can try 'gemini-1.5-pro'
        self.num_reviews = reviews  # Set to None to process all, or a number for quick testing
        self.output_path = output  # Optional: to save results
        self.build_prompt = prompt_builder

def configure_api():
    """Configures the Gemini API key."""
    api_key = "AIzaSyA4IDp67D0CM7mR5WAgOlhUXbdocbfbYC4"
    try:
        genai.configure(api_key=api_key)
        print("API Key configured.")
        return True
    except Exception as e:
        print(f"Error configuring the API: {e}")
        sys.exit("API Key configuration failed.")


def load_data(params: PromptingParameters):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(params.csv_path)
        print(f"Dataset loaded from {params.csv_path} with {len(df)} rows.")
        # Verify if the review column exists
        if params.REVIEW_COLUMN not in df.columns:
            raise ValueError(f"The review column '{params.REVIEW_COLUMN}' was not found in the CSV.")
        # Verify if the real town column exists (if provided)
        if params.column and params.column not in df.columns:
            print(
                f"Warning: The real town column '{params.column}' was not found. Accuracy evaluation will not be possible.")
            params.column = None  # Nullify if it doesn't exist to avoid later errors
        return df, params.column
    except FileNotFoundError:
        print(f"Error: CSV file not found at {params.csv_path}.")
        sys.exit("Error loading data.")
    except ValueError as ve:
        print(f"Error: {ve}")
        sys.exit("Error in CSV columns.")
    except Exception as e:
        print(f"Unexpected error loading the CSV: {e}")
        sys.exit("Error loading data.")


# --- 5. Asynchronous API Call ---
async def guess_row_async(prompt, gemini_model, retries=3, delay=20):
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
async def process_reviews_async(df, labels, params: PromptingParameters):
    """
    Asynchronously processes reviews in batches, with a 1-minute delay between batches.
    """
    predictions = [None] * len(df)  # Initialize a list to store predictions

    if params.num_reviews:
        df_to_process = df.head(params.num_reviews)
        print(f"Processing the first {params.num_reviews} reviews...")
    else:
        df_to_process = df
        print(f"Processing all {len(df)} reviews...")

    total_reviews_to_process = len(df_to_process)
    rate_limit_per_minute = 1500
    batch_size = rate_limit_per_minute # Each batch is up to the rate limit
    num_batches = math.ceil(total_reviews_to_process / batch_size)

    print(f"Total reviews to process: {total_reviews_to_process}")
    print(f"Processing in batches of up to {batch_size}.")
    print(f"Estimated number of batches: {num_batches}")


    async def process_row(row):
        global invalid_reviews
        global rule_processed
        global api_processed

        # Use the original index from the full DataFrame
        original_index = row.name

        current_review = str(row[params.TITLE_COLUMN]) + ' ' + str(row[params.REVIEW_COLUMN])
        if pd.isna(current_review) or not current_review.strip():
            invalid_reviews += 1
            print(f"Review in row {original_index + 1} is empty or invalid. Skipping API call.")
            return

        prompt = params.build_prompt(current_review, labels)
        # print(f"\n--- Prompt for row {original_index + 1} ---")

        prediction = await guess_row_async(prompt, params.model)
        predictions[original_index] = prediction # Use original_index here
        print(f"Row {original_index + 1}: Review processed. Prediction: {prediction}") # Simplified logging per row


    # --- Batch Processing Loop ---
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, total_reviews_to_process)
        batch_df = df_to_process.iloc[start_index:end_index]

        print(f"\n--- Processing Batch {i + 1}/{num_batches} (Reviews {start_index + 1} to {end_index}) ---")
        batch_start_time = time.monotonic()

        # Create and run tasks for the current batch
        tasks = [process_row(row) for _, row in batch_df.iterrows()] # Pass None for index, use row.name inside
        await asyncio.gather(*tasks)

        batch_end_time = time.monotonic()
        batch_duration = batch_end_time - batch_start_time
        print(f"Batch {i + 1} finished in {batch_duration:.2f} seconds.")

        # Wait for the remainder of the minute if it's not the last batch
        if i < num_batches - 1:
            time_to_wait = 60.0 - batch_duration
            if time_to_wait > 0:
                print(f"Waiting for {time_to_wait:.2f} seconds before the next batch...")
                await asyncio.sleep(time_to_wait)
            else:
                print("Batch processing took longer than 60 seconds. No additional wait needed.")
    df['gemini_prediction'] = predictions
    return df


# --- 7. Evaluation (Optional) ---

def evaluate_predictions(df_results, real_column):
    """Calculates the accuracy if the real values are available."""
    global invalid_reviews, api_processed, rule_processed
    if real_column is None or real_column not in df_results.columns:
        print("The column with the real town was not provided or does not exist. Accuracy cannot be calculated.")
        return

    correct = (df_results[real_column].str.strip().str.lower() ==
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


def get_labes_from_column(df, column):
    """ Extracts all unique labels from a specified column in a CSV dataset. """
    try:
        if column in df.columns:
            unique_towns = df[column].unique().tolist()
            return unique_towns
        else:
            print(f"Error: Column '{column}' not found in the CSV file.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- Main prompting function ---
def prompting(params: PromptingParameters):
    print("--- Starting Asynchronous Magic Town Review Classification Program ---")

    # 1. Configure API
    if not configure_api():
        sys.exit("API configuration failed.")

    # 2. Load Data
    original_df, real_column = load_data(params)
    labels = get_labes_from_column(original_df, real_column)

    # 4. Process Reviews Asynchronously
    results_df = asyncio.run(process_reviews_async(original_df.copy(), labels, params))

    # 5. Show some results
    print("\n--- First rows with predictions ---")
    print(results_df[[params.REVIEW_COLUMN, real_column, 'gemini_prediction']].head())

    # 6. Evaluate (if applicable)
    if real_column:
        evaluate_predictions(results_df, real_column)

    # 7. Save Results (Optional)
    try:
        results_df.to_csv(params.csv_path, index=False, encoding='utf-8')
        print(f"\nResults saved to: {params.csv_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

    print("\n--- Asynchronous Program Finished ---")

