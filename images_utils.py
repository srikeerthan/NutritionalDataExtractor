import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
import json
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'


def download_all_images(url, output_folder="images", keyword_filters=None):
    """
    Download all images from a webpage and filter based on keywords in the attributes.

    Args:
    - url (str): The webpage URL to crawl.
    - output_folder (str): Folder to save the downloaded images.
    - keyword_filters (list): List of keywords to filter images by (e.g., ['nutrition', 'table']).

    Returns:
    - list: A list of file paths for the downloaded images.
    """
    try:
        # Send a GET request to fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the response is not 200
        soup = BeautifulSoup(response.text, 'html.parser')

        # Prepare output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Find all <img> tags
        img_tags = soup.find_all('img')
        if not img_tags:
            print("No images found on the page!")
            return []

        downloaded_images = []
        for img_tag in img_tags:
            # Get the image URL
            img_src = img_tag.get('src')
            if not img_src:
                continue  # Skip if the <img> tag doesn't have a source

            if "http" in img_src or "https" in img_src:
                img_url = img_src
            else:
                img_url = urljoin(url, img_src)  # Handle relative paths

            # Check for keyword filters
            if keyword_filters:
                alt_text = img_tag.get('alt', '').lower()
                src_text = img_src.lower()
                if not any(keyword in alt_text or keyword in src_text for keyword in keyword_filters):
                    continue  # Skip images that don't match the filters

            # Get the image data
            img_data = requests.get(img_url).content

            # Extract image filename from the URL
            img_filename = os.path.basename(img_url)
            output_path = os.path.join(output_folder, img_filename)

            # Save the image locally
            with open(output_path, 'wb') as img_file:
                img_file.write(img_data)

            downloaded_images.append(output_path)
            print(f"Image downloaded: {output_path}")

        if not downloaded_images:
            print("No images matched the specified filters.")
        return downloaded_images

    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    Args:
    - image_path (str): Path to the image file.

    Returns:
    - np.array: Preprocessed image.
    """

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to make text stand out
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image (Tesseract works better with black text on white background)
    img_bin = 255 - img_bin

    # Use dilation to make table lines more prominent
    kernel = np.ones((2, 2), np.uint8)
    img_dilated = cv2.dilate(img_bin, kernel, iterations=1)

    return img_dilated


def extract_table_from_image(image_path):
    """
    Extract text from tables in an image.
    Args:
    - image_path (str): Path to the image file.

    Returns:
    - str: Extracted table data.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Save the processed image to a temporary file
    temp_image_path = "temp_image.jpg"  # Choose a temporary file name
    cv2.imwrite(temp_image_path, processed_image)

    img = Image.open(temp_image_path)
    # Use Tesseract to perform OCR on the image
    custom_config = r'--psm 6 -c tessedit_create_tsv=1'  # Page segmentation mode for tables
    table_data = pytesseract.image_to_data(img, config=custom_config, lang='eng', output_type=pytesseract.Output.DICT)

    print(table_data)

    # Convert positional data into a table
    rows = {}
    for i in range(len(table_data['text'])):
        text = table_data['text'][i].strip()
        if text:  # Ignore empty text entries
            # Group text by their `top` coordinate (row position)
            top = table_data['top'][i]
            left = table_data['left'][i]
            if top not in rows:
                rows[top] = []
            rows[top].append((left, text))

    # Sort rows by `top` coordinate
    sorted_rows = sorted(rows.items())

    print(sorted_rows)

    # Build the table as a list of rows
    table = []
    for _, row in sorted_rows:
        # Sort text within the row by `left` coordinate (column position)
        sorted_row = [text for _, text in sorted(row)]
        table.append(sorted_row)

    print(table)

    return table


def table_to_dataframe(table):
    """
    Convert the extracted table into a Pandas DataFrame.
    """
    # Normalize rows to the same number of columns
    max_columns = max(len(row) for row in table)
    normalized_table = [row + [''] * (max_columns - len(row)) for row in table]

    # Convert to a DataFrame
    df = pd.DataFrame(normalized_table)
    return df


