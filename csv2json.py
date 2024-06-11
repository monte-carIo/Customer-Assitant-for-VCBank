import csv
import json
from googletrans import Translator
import time

def translate_vietnamese_to_english(text):
    translator = Translator()
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            translation = translator.translate(text, src='vi', dest='en')
            return translation.text
        except Exception as e:
            print(f"Translation failed: {e}")
            print(f"Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(1)  # Add a delay before retrying
    print("Translation failed after multiple retries. Returning original text.")
    return text

def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            instruction_vi = row['question']
            # breakpoint()
            output_vi = row['answer']
            
            # Translate Vietnamese text to English
            instruction_en = translate_vietnamese_to_english(instruction_vi)
            output_en = translate_vietnamese_to_english(output_vi)
            
            item = {
                "instruction": instruction_en,
                "input": "",
                "output": output_en,
                "system": "",
                "history": []
            }
            data.append(item)
    
    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=2)

# Replace 'questions_answers.csv' and 'output.json' with your file paths
csv_to_json('questions_answers.csv', 'output_en.json')
