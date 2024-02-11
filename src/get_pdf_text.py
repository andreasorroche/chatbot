from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os


def get_pdf_text(pdf_docs):
    """
    Extract text content from each page of a list of PDF documents.

    This function takes a list of PDF document filenames and extracts text content from each page
    of each document. It concatenates the extracted text from all pages of all provided PDFs and
    returns it as a single string.

    Args:
        pdf_docs (list of str): A list of filenames for PDF documents.

    Returns:
        str: A single string containing the concatenated text from all pages of all input PDFs.
    """

    text = ""
    for pdf in pdf_docs:
        pdf_path = os.path.join("/app/data", pdf)
        if os.path.exists(pdf_path):
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            print(f'El archivo PDF "{pdf}" no existe en el directorio "/app/data".')
    return text


def get_text_chunks(text: str):
    """
    Split a text string into smaller chunks using specific configuration parameters.

    This function receives a text string and splits it into smaller chunks. It utilizes a
    CharacterTextSplitter object with defined parameters to control the splitting behavior.
    The parameters include the separator character, chunk size, chunk overlap, and the length
    measuring function.

    Args:
        text (str): The text string to be split into chunks.

    Returns:
        list of str: A list of text chunks derived from the input text string.

    Configuration Parameters:
        - separator (str): Newline character ("\n"), used to split the text into chunks.
        - chunk_size (int): Desired size for each text chunk, set to 1500 characters.
        - chunk_overlap (int): Overlap between consecutive chunks, set to 200 characters.
        - length_function (function): Function to measure text length, set to `len`.
    """

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
