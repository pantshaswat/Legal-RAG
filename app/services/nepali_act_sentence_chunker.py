import re
from pypdf import PdfReader
import json
import os

# Map Nepali digits to English digits
nepali_to_english_digit = str.maketrans('०१२३४५६७८९', '0123456789')

def convert_nepali_number_to_english(num_str: str) -> int:
    """Convert Nepali numeral string to an integer."""
    return int(num_str.translate(nepali_to_english_digit))

def split_into_sentences(text: str) -> list:
    """Split text into sentences using Nepali full-stop (।)."""
    # Split by Nepali full-stop, but keep the delimiter with the sentence
    sentences = re.split(r'(।)', text)
    
    # Rejoin the sentences with their delimiters
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    
    # Handle any remaining text (if the text doesn't end with a delimiter)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])
    
    return [sentence.strip() for sentence in result if sentence.strip()]

def chunk_nepali_legal_text(text: str, max_sections: int = None):
    """
    Chunks Nepali legal text into sections and sentences.
    Each sentence becomes its own chunk within a section.
    Only process up to max_sections sections if specified.
    
    Improved to detect full titles that may end with ':' or Nepali characters like 'षाः'.
    """
    # Updated pattern to better capture section numbers and titles
    # This pattern looks for a number followed by a dot, then captures everything until
    # either a '।' (Nepali full-stop) or before content that looks like a new paragraph
    section_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>.*?)(?=\s+\d+\.|\s*।|\s*$)'
    
    # Alternative pattern to capture the title until a colon or specific Nepali characters
    title_end_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>.*?(?::|षाः|ः))\s+(?P<first_line>.*)'
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    chunks = []
    current_chunk = None
    current_content = []
    sections_processed = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Try the title end pattern first (for titles that end with : or special chars)
        match = re.match(title_end_pattern, line)
        
        if not match:
            # If that doesn't match, try the regular section pattern
            match = re.match(section_pattern, line)
        
        if match:
            if max_sections is not None and sections_processed >= max_sections:
                break
                
            # Process previous section if it exists
            if current_chunk is not None:
                full_content = ' '.join(current_content).strip()
                current_chunk['content'] = split_into_sentences(full_content)
                chunks.append(current_chunk)
            
            section_num_nepali = match.group('section_num')
            section_num = section_num_nepali
            title = match.group('title').strip()
            
            # Get the first line content that follows the title
            # If using the title_end_pattern, we already have first_line
            # Otherwise, the first line is what remains after the title in the current line
            if 'first_line' in match.groupdict():
                first_line = match.group('first_line').strip()
            else:
                # Extract what remains after the title
                title_pattern = re.escape(title)
                first_line = re.sub(f'^{section_num_nepali}\.\s+{title_pattern}\s*', '', line).strip()
            
            current_chunk = {
                'section_num': section_num,
                'title': title,
                'content': []
            }
            
            full_first_line = f"{section_num_nepali}. {title}"
            if first_line:
                full_first_line += f" {first_line}"
                
            current_content = [full_first_line]
            sections_processed += 1
                
        elif current_chunk is not None:
            current_content.append(line)
            
        i += 1
    
    # Process the last section
    if current_chunk is not None and (max_sections is None or sections_processed <= max_sections):
        full_content = ' '.join(current_content).strip()
        current_chunk['content'] = split_into_sentences(full_content)
        chunks.append(current_chunk)
    
    return chunks

def process_nepali_pdf_to_chunks(filename: str, output_dir: str = '../../data/nepali/processed/json', max_sections: int = None):
    """
    Process a Nepali PDF file and save chunked content as JSON.
    Each sentence in a section becomes its own chunk.
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

    # Clean up any newlines within sentences
    for chunk in chunks:
        chunk['content'] = [content.replace('\n', '') for content in chunk['content']]
    
    output_path = f'{output_dir}/{filename}.json'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    
    print(f"Processed {len(chunks)} sections (max_sections={max_sections})")
    total_sentences = sum(len(chunk['content']) for chunk in chunks)
    print(f"Total sentences: {total_sentences}")
    print(f"Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    filename = 'data_act_2080'
    process_nepali_pdf_to_chunks(filename, max_sections=32)  # example: only process first 32 sections