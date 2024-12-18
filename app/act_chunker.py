import re
from pypdf import PdfReader
import json

def split_into_smaller_chunks(text: str, max_words: int = 250) -> list:
    """
    Split a text into smaller chunks of approximately max_words words.
    
    Args:
        text (str): Text to split
        max_words (int): Maximum number of words per chunk
        
    Returns:
        list: List of text chunks
    """
    # Split the text into sentences (assuming sentences end with . ? or !)
    sentences = re.split('(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed max_words, save current chunk and start new one
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_legal_text(text: str, max_words: int = 250):
    """
    Chunks legal text into sections based on numbered headings.
    Content longer than max_words is split into an array of smaller chunks.
    
    Args:
        text (str): The legal text to be chunked
        max_words (int): Maximum number of words per content chunk
        
    Returns:
        list[dict]: List of dictionaries containing section number, title, and content
    """
    section_pattern = r'(?P<section_num>\d+)\.\s+(?P<title>[^:]+):\s*(?P<first_line>.*)'
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    chunks = []
    current_chunk = None
    current_content = []
    
    for line in lines:
        match = re.match(section_pattern, line)
        
        if match:
            # Save previous chunk if it exists
            if current_chunk is not None:
                full_content = '\n'.join(current_content).strip()
                
                # Check if content needs to be split
                word_count = len(full_content.split())
                if word_count > max_words:
                    current_chunk['content'] = split_into_smaller_chunks(full_content, max_words)
                else:
                    current_chunk['content'] = [full_content]
                    
                chunks.append(current_chunk)
            
            # Create new chunk
            section_num = int(match.group('section_num'))
            title = match.group('title').strip()
            first_line = match.group('first_line').strip()
            
            current_chunk = {
                'section_num': section_num,
                'title': title,
                'content': []
            }
            
            current_content = [f"{section_num}. {title}: {first_line}"]
        elif current_chunk is not None:
            current_content.append(line)
    
    # Save the last chunk
    if current_chunk is not None:
        full_content = '\n'.join(current_content).strip()
        
        # Check if content needs to be split
        word_count = len(full_content.split())
        if word_count > max_words:
            current_chunk['content'] = split_into_smaller_chunks(full_content, max_words)
        else:
            current_chunk['content'] = [full_content]
            
        chunks.append(current_chunk)
    
    return chunks

def process_pdf_to_chunks(filename: str, output_dir: str = '../data/processed/json/Finance'):
    """
    Process a PDF file and save chunked content as JSON.
    
    Args:
        filename (str): Name of the PDF file (without .pdf extension)
        output_dir (str): Directory to save the JSON output
    """
    # Read PDF
    document = PdfReader(f"../data/raw/Finance/{filename}.pdf")
    text = ""
    for page_obj in document.pages:
        text += page_obj.extract_text()
    
    # Create chunks
    chunks = chunk_legal_text(text)
    
    # Save as JSON
    output_path = f'{output_dir}/{filename}.json'
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=4)
    
    print(f"Processed {len(chunks)} sections")
    print(f"Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    filename = 'Patent Design and Trademark Act'
    process_pdf_to_chunks(filename)