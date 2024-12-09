# Nutritional Data Extractor

This is a Python Script to Extract the Nutritional Facts of different dishes or items serverd by the Restuarants.

It takes the Restaurant name and URL containing Nutritional Facts data as Inputs and Provides the Nutrtional Facts data as Json in the below Format:
```
{
  "source": "Kitava",
  "items": [
    {
      "name": "Evolved Caesar Salad",
      "nutrition": {
        "item": 357.8,
        "calories": 1442.2,
        "protein_g": 11.4,
        "fat_g": 7.8,
        "carbs_g": 14.4
      }
    }
  ]
}
```

Currently the Project supports Nutritional Facts Data Extraction from the PDF files only.
