import re
from pypdf import PdfReader
import json
import os

# Map Nepali digits to English digits
nepali_to_english_digit = str.maketrans('०१२३४५६७८९', '0123456789')

def convert_nepali_number_to_english(num_str: str) -> int:
    """Convert Nepali numeral string to an integer."""
    return int(num_str.translate(nepali_to_english_digit))

def split_into_smaller_chunks(text: str, max_words: int = 250) -> list:
    sentences = re.split(r'(?<=[।!?])\s+', text)  # '।' is the Nepali full-stop
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def chunk_nepali_legal_text(text: str, max_words: int = 250, max_sections: int = None):
    """
    Chunks Nepali legal text into sections based on Nepali numbered headings.
    Only process up to max_sections sections if specified.
    """
    section_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>.+?)\s+(?P<first_line>.*)'

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    chunks = []
    current_chunk = None
    current_content = []
    sections_processed = 0

    for line in lines:
        match = re.match(section_pattern, line)
        
        if match:
            if max_sections is not None and sections_processed >= max_sections:
                break

            if current_chunk is not None:
                full_content = ' '.join(current_content).strip()
                word_count = len(full_content.split())
                if word_count > max_words:
                    current_chunk['content'] = split_into_smaller_chunks(full_content, max_words)
                else:
                    current_chunk['content'] = [full_content]
                chunks.append(current_chunk)

            section_num_nepali = match.group('section_num')
            section_num = section_num_nepali
            title = match.group('title').strip()
            first_line = match.group('first_line').strip()

            current_chunk = {
                'section_num': section_num,
                'title': title,
                'content': []
            }
            current_content = [f"{section_num_nepali}. {title} {first_line}"]
            sections_processed += 1

        elif current_chunk is not None:
            current_content.append(line)
    
    if current_chunk is not None and (max_sections is None or sections_processed <= max_sections):
        full_content = ' '.join(current_content).strip()
        word_count = len(full_content.split())
        if word_count > max_words:
            current_chunk['content'] = split_into_smaller_chunks(full_content, max_words)
        else:
            current_chunk['content'] = [full_content]
        chunks.append(current_chunk)

    return chunks

def process_nepali_pdf_to_chunks(filename: str, output_dir: str = '../../data/nepali/processed/json', max_sections: int = None):
    """
    Process a Nepali PDF file and save chunked content as JSON.
    """
    document = PdfReader(f"../../data/nepali/raw/{filename}.pdf")
    text = ""
    for page_obj in document.pages:
        text += page_obj.extract_text()
    
    # Remove unnecessary content like URLs (if any)
    cleaned_text = re.sub(r'www\.[a-zA-Z0-9./]+', '', text)

    extracted_dir = os.path.abspath("../../data/nepali/processed/extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    hello_txt_path = os.path.join(extracted_dir, f"{filename}.txt")
    with open(hello_txt_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)
    
    chunks = chunk_nepali_legal_text(cleaned_text, max_sections=max_sections)

    for chunk in chunks:
        chunk['content'] = [content.replace('\n', '') for content in chunk['content']]
    
    output_path = f'{output_dir}/{filename}.json'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    
    print(f"Processed {len(chunks)} sections (max_sections={max_sections})")
    print(f"Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    filename = 'data_act_2080'
    process_nepali_pdf_to_chunks(filename, max_sections=32)  # example: only process first 50 sections
