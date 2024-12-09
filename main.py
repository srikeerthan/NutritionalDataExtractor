import json
import mimetypes
import re
import sys

import requests
import pandas as pd
from tabula import read_pdf
from sentence_transformers import SentenceTransformer, util

from images_utils import download_all_images, extract_table_from_image, table_to_dataframe
from utils import validate_url

# Target nutrition keys
target_keys = {
    "item": ["item", "NaN"],
    "calories": ["calories"],
    "protein_g": ["protein"],
    "fat_g": ["fat", "total fat"],
    "carbs_g": ["carbs", "net carbs"]
}

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def map_key_to_target(key):
    """Map a key to the closest target key based on similarity."""
    key_embedding = model.encode(key.lower())
    max_similarity = -1
    best_match = None
    for target, synonyms in target_keys.items():
        for synonym in synonyms:
            synonym_embedding = model.encode(synonym.lower())
            similarity = util.cos_sim(key_embedding, synonym_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = target
    return best_match


def is_google_drive_url(link):
    """Check if the URL is a Google Drive link."""
    return "drive.google.com" in link


def get_google_drive_file_download_link(link):
    """Constructs Google Drive file download link."""
    # Extract the file ID from the URL
    try:
        file_id = link.split("/d/")[1].split("/")[0]
    except IndexError:
        print("Invalid Google Drive URL format.")
        return

    # Construct the direct download URL
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def get_formatted_link(link):
    if is_google_drive_url(link):
        return get_google_drive_file_download_link(link)
    return link


def guess_format_of_link(link, strict=True):
    # Guess the type based on the file extension
    link_type, _ = mimetypes.guess_type(link)

    # If no type is guessed and strict mode is enabled, fetch headers
    if link_type is None and strict:
        try:
            response = requests.get(link, allow_redirects=True)
            link_type = response.headers.get('Content-Type')

            # finding the type of file in cases like Google Drive
            content_disposition = response.headers.get('content-disposition')
            if content_disposition and re.findall(r'filename="([^"]+)"', content_disposition):
                file_type = re.findall(r'filename="([^"]+)"', content_disposition)[0].split('.')[
                    -1]  # Get the part after the last dot
                return file_type

        except Exception as e:
            print(f"Error fetching URL {link}: {e}")
            link_type = "Unknown"

    return link_type


def download_file_to_local(link, filename, file_extension):
    try:
        # Send a GET request to the download URL
        response = requests.get(link, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an error for unsuccessful requests (4xx/5xx)

        output_file = filename + '.' + file_extension

        # Write the file to disk in binary mode
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File successfully downloaded: {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return


def does_second_match_reference(reference, to_check, threshold=0.7):
    """
    Compare the second string against the reference to see if it holds the same meaning.

    Args:
    - reference (str): The reference string (the one you want).
    - to_check (str): The string to check for the same meaning as the reference.
    - threshold (float): A similarity threshold to determine if meanings are "same". Default is 0.8.

    Returns:
    - bool: True if the `to_check` string has the same meaning as `reference`, False otherwise.
    - float: The semantic similarity score (between 0 and 1).
    """
    # Encode the strings into embeddings
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    to_check_embedding = model.encode(to_check, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(reference_embedding, to_check_embedding).item()

    # Return whether the similarity exceeds the threshold
    return similarity_score >= threshold


def clean_and_convert_row_to_string(dataframe):
    # Extract the first row
    first_row = dataframe.iloc[0]  # Get the first row as a Series

    # Drop NaN values
    cleaned_row = first_row.dropna()

    # Convert to a string without index
    row_string = ' '.join(cleaned_row.astype(str).tolist())

    return row_string


def extract_nutritional_facts_from_pdf(file_path):
    heading_to_find = "Nutritional Facts"  # Example: looking for this heading

    # Extract all tables from the PDF
    tabular_data = read_pdf(file_path, pages='all', multiple_tables=True,
                            pandas_options={"header": None})  # Treat all rows as data

    # List to store all matching tables
    nutritional_facts_tables = []

    # Iterate over the tables and check if any table contains the heading
    for i, table in enumerate(tabular_data):
        first_row_string = clean_and_convert_row_to_string(table)
        if does_second_match_reference(heading_to_find,
                                       first_row_string):  # Convert table to string to search for the heading

            # Drop rows with fewer than 2 non-null values
            table = table.dropna(thresh=2).reset_index(drop=True)

            # Set the first row as the header
            table.columns = table.iloc[0]  # Set the first row as the header
            table = table[1:].reset_index(drop=True)  # Remove the header row from the data

            # # Check for NaN in the first column name and set it to 'item' if necessary
            table.columns.values[0] = 'item'  # Rename the first column to 'item'

            nutritional_facts_tables.append(table)  # Add the matching table to the list

    # If we have any matching tables, merge them into a single DataFrame
    if nutritional_facts_tables:
        combined_table = pd.concat(nutritional_facts_tables, ignore_index=True)
        # Convert to JSON
        json_result = combined_table.to_json(orient="records")  # Convert to JSON format
        return json_result
    else:
        print(f"No table found with the heading '{heading_to_find}'")
        return []


def get_formatted_nutrition_json(restaurant, nutritional_facts_json_str):
    # Convert input JSON to desired format
    items_nutrition_list = []

    nutritional_facts_json = json.loads(nutritional_facts_json_str)
    for entry in nutritional_facts_json:
        item_name = entry.get("item")
        nutrition_data = {}

        for key, value in entry.items():
            if key.lower() == "item":
                continue  # Skip the "item" key
            matched_key = map_key_to_target(key)
            if matched_key:
                try:
                    nutrition_data[matched_key] = float(value) if value else 0
                except ValueError:
                    if value in ['o', 'O']:
                        nutrition_data[matched_key] = 0

        items_nutrition_list.append({
            "name": item_name,
            "nutrition": nutrition_data
        })

    return {"source": restaurant, "items": items_nutrition_list}


def get_nutrition_data_from_pdf(restaurant, link):
    # Download the PDF file to local
    download_file_to_local(link, "nutrition_data", "pdf")

    # Extract Nutritional Data from the PDF
    nutritional_facts_json = extract_nutritional_facts_from_pdf("nutrition_data.pdf")
    nutrition_json = get_formatted_nutrition_json(restaurant, nutritional_facts_json)
    return nutrition_json


def get_nutritional_facts_from_images(downloaded_images):
    for image_path in downloaded_images:
        table_data = extract_table_from_image(image_path)
        print("Extracted Table Data:")
        print(table_data)
        df = table_to_dataframe(table_data)
        print(df)


def get_nutrition_data_from_html_page(restaurant, link):
    keyword_filters = ['nutrition', 'table', 'facts', 'nutritional']  # Customize keywords as needed
    downloaded_images = download_all_images(url, keyword_filters=keyword_filters)
    if downloaded_images:
        get_nutritional_facts_from_images(downloaded_images)
    else:
        return {}

    return {}


def main(restaurant, link):
    # Get the formatted link
    link = get_formatted_link(link)

    link_type = guess_format_of_link(link)
    if "html" in link_type.lower():
        print("Given URL is a html web page")
        nutritional_facts_json = get_nutrition_data_from_html_page(restaurant, link)
    elif "pdf" in link_type.lower():
        print("Given URL is a pdf file")
        nutritional_facts_json = get_nutrition_data_from_pdf(restaurant, link)
        print("\nNutritional Facts:\n")
        print(nutritional_facts_json)
    elif "image" in link_type.lower():
        print("Given URL is a Image file")
    else:
        print(f"Given URL is not a supported format: {link_type}", )
    return


if __name__ == '__main__':
    # Prompt user for input
    restaurant_name = input("Enter the restaurant name: ").strip()
    url = input("Enter the restaurant's URL: ").strip()

    # Validate URL
    if not validate_url(url):
        print("Error: Invalid URL format. Please provide a valid URL.")
        sys.exit(1)

    main(restaurant_name, url)
