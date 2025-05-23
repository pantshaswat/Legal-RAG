import re
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
    section_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>.+?[ः:])(?=\s+\d+\.|\s*।|\s*$)'

    # Alternative pattern to capture the title until a colon or specific Nepali characters
    # Modified to handle spaces before colons
    title_end_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>.*?(?:\s*:|षाः|ः))\s+(?P<first_line>.*)'

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

def process_extracted_text_to_chunks(filename: str, output_dir: str = 'data/nepali/processed/json', max_sections: int = None):
    """
    Process an already extracted text file and save chunked content as JSON.
    Each sentence in a section becomes its own chunk.
    """
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct absolute paths
    text_path = os.path.join(project_root, 'data', 'nepali', 'processed', 'extracted', f"{filename}.txt")
    output_dir = os.path.join(project_root, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read text file
        print(f"Reading text file from: {text_path}")
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        if not text.strip():
            print("Error: Text file is empty")
            return
            
        # Process the text into chunks
        chunks = chunk_nepali_legal_text(text, max_sections=max_sections)

        # Clean up any newlines within sentences
        for chunk in chunks:
            chunk['content'] = [content.replace('\n', '') for content in chunk['content']]
        
        output_path = os.path.join(output_dir, f'{filename}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        
        print(f"Processed {len(chunks)} sections (max_sections={max_sections})")
        total_sentences = sum(len(chunk['content']) for chunk in chunks)
        print(f"Total sentences: {total_sentences}")
        print(f"Output saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        raise

def process_all_extracted_files(input_dir: str = None, output_dir: str = None, max_sections: int = None):
    """
    Process all text files in the extracted directory.
    """
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct absolute paths
    if input_dir is None:
        input_dir = os.path.join(project_root, 'data', 'nepali', 'processed', 'extracted')
    else:
        input_dir = os.path.join(project_root, input_dir)
        
    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'nepali', 'processed', 'json')
    else:
        output_dir = os.path.join(project_root, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all text files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt') and not f.endswith('_raw.txt')]
    
    if not files:
        print(f"No text files found in {input_dir}")
        return
    
    print(f"Found {len(files)} text files to process")
    
    # Process each file
    for file in files:
        filename = os.path.splitext(file)[0]  # Remove the .txt extension
        print(f"\nProcessing file: {filename}")
        process_extracted_text_to_chunks(filename, output_dir=output_dir, max_sections=max_sections)
    
    print(f"\nAll files processed. Results saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Process a single file
    filename = 'bankoffence'
    process_extracted_text_to_chunks(filename, max_sections=28)
    
    # Or process all files in the extracted directory
    process_all_extracted_files(max_sections=None)  # Set to None to process all sections