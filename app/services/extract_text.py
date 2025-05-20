



import re
import os
from pdf2image import convert_from_path
import pytesseract

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata/'

# Function to clean up extracted Nepali text
def clean_nepali_text(text):
    # Remove unnecessary content like URLs (if any)
    cleaned_text = re.sub(r'www\.[a-zA-Z0-9./]+', '', text)
    
    # Remove numbered bullets like "1.", "२.", etc.
    # cleaned_text = re.sub(r'^\d+\.\s', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove bullets like "(क)", "(ख)", etc.
    # cleaned_text = re.sub(r'\([\u0900-\u097F]\)(?=\s|$)', '', cleaned_text)
    
    # Remove isolated page numbers or standalone digits
    cleaned_text = re.sub(r'^\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove excessive line breaks and clean up whitespace
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text).strip()
    
    return cleaned_text

# Paths
pdf_path = '../../data/nepali/raw/data_act_2080.pdf'  # Replace with your PDF file path
output_folder = '../../data/nepali/processed/extracted'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Extract file name from the PDF path
file_name = os.path.splitext(os.path.basename(pdf_path))[0]

# Output text file path
output_text_path = os.path.join(output_folder, f"{file_name}.txt")

# Convert PDF to images
images = convert_from_path(pdf_path)

# Process each page
all_text = ""
for image in images:
    # Extract text from image using pytesseract (Nepali language config)
    custom_config = r'--psm 6 -l nep'  # 'nep' is the language code for Nepali
    extracted_text = pytesseract.image_to_string(image, config=custom_config)
    
    # Clean the extracted text
    cleaned_page_text = clean_nepali_text(extracted_text)
    
    # Append the cleaned text without any page headers
    all_text += cleaned_page_text + "\n"

# Save the cleaned text to a file
with open(output_text_path, 'w', encoding='utf-8') as file:
    file.write(all_text)

print(f"Cleaned text has been saved to: {output_text_path}")