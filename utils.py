import re
from io import BytesIO
import pypdf
import docx

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from PDF or DOCX files."""
    text = ""
    if filename.lower().endswith('.pdf'):
        pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif filename.lower().endswith('.docx'):
        doc = docx.Document(BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file_bytes.decode('utf-8', errors='ignore')
    
    return text.strip()

def clean_text(text: str) -> str:
    """Clean text by stripping non-alphanumeric characters and normalizing spacing."""
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()
