"""Vector store file splitter utilities.

This module provides functionality to split various file types into documents
for vector store indexing. It includes automatic file type detection and
specialized splitters for different document formats.
"""

import base64
import csv
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Type

import fitz  # PyMuPDF
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from PIL import Image

FILE_SPLITTERS: Dict[str, Type["FileSplitter"]] = {}

# Load environment variables
load_dotenv(find_dotenv())


def get_splitter(file_path: str) -> Type["FileSplitter"]:
    """Retrieve the appropriate FileSplitter for a given file path.

    This function matches the file path against registered file patterns
    in FILE_SPLITTERS and returns the FileSplitter with the highest priority.

    :param file_path: Path to the file for which a splitter is needed.
    :return: The FileSplitter class with the highest priority that matches the file path.
    :raises ValueError: If no matching splitter is found for the given file path.
    """
    ret = []
    for _, obj in FILE_SPLITTERS.items():
        if re.match(obj.file_pattern_re, file_path):
            ret.append([obj.priority, obj])
    if not ret:
        raise AttributeError(
            f"No splitter found for file: '{file_path}'. Supported splitters: {list(FILE_SPLITTERS.keys())}"
        )

    return sorted(ret, key=lambda x: x[0])[-1][1]


@dataclass
class FileSplitter:
    """Base class for file splitters.

    This class serves as a base for specific file splitters. Subclasses are automatically
    registered in the `FILE_SPLITTERS` dictionary unless their class name starts with '_'.

    :param file_pattern_re: Regular expression pattern for matching file types.
    :param priority: Priority of the file splitter.
    """

    file_pattern_re: str
    priority: int

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in `FILE_SPLITTERS`.

        Subclasses with names not starting with '_' are added to the global `FILE_SPLITTERS`
        dictionary, allowing for easy access and management of different file splitters.
        """
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("_"):
            if cls.__name__ == "PdfLlmSplitter" and not (
                os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_KEY")
            ):
                # If .env is not set, skip the PdfLlmSplitter
                return
            FILE_SPLITTERS[cls.__name__] = cls

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a file into documents.

        This method should be implemented by subclasses to define specific splitting logic.

        :param file_path: Path to the file to be split.
        :return: A list of Document objects resulting from the split.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


@dataclass(eq=False)
class PdfSplitter(FileSplitter):
    """Splits PDF files into documents.

    This class provides functionality to load and split PDF files into smaller
    document chunks using a text splitter.
    """

    file_pattern_re = r".+\.pdf"
    priority: int = 1

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a PDF file into documents.

        Loads a PDF file and splits it into smaller chunks using a character-based
        text splitter for further processing.

        :param file_path: Path to the PDF file to be split.
        :return: A list of Document objects resulting from the split.
        """
        loader = PyPDFLoader(file_path, extraction_mode="plain", extract_images=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
        return loader.load_and_split(text_splitter=text_splitter)


@dataclass(eq=False)
class TxtSplitter(FileSplitter):
    """Splits text and log files into documents.

    This class provides functionality to load and split text-based files
    into smaller document chunks using a text splitter.
    """

    file_pattern_re = r".+\.(txt|log)"
    priority: int = 1

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a text or log file into documents.

        Loads a text or log file and splits it into smaller chunks using a character-based
        text splitter for further processing.

        :param file_path: Path to the text or log file to be split.
        :return: A list of Document objects resulting from the split.
        """
        loader = TextLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)
        return loader.load_and_split(text_splitter=text_splitter)


@dataclass(eq=False)
class MdSplitter(FileSplitter):
    """Splits markdown files into documents.

    This class provides functionality to load and split markdown files
    into smaller document chunks using a markdown-aware text splitter.
    """

    file_pattern_re = r".+\.md"
    priority: int = 1

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a text or log file into documents.

        Loads a text or log file and splits it into smaller chunks using a character-based
        text splitter for further processing.

        :param file_path: Path to the text or log file to be split.
        :return: A list of Document objects resulting from the split.
        """
        loader = TextLoader(file_path)
        text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
        return loader.load_and_split(text_splitter=text_splitter)


@dataclass(eq=False)
class PdfLlmSplitter(FileSplitter):
    """Splits PDF files using LLM vision model for structured markdown extraction.

    This class converts PDF pages to images and uses Azure OpenAI vision model
    to extract structured markdown content, which is then split into documents.
    Processes up to 10 pages in parallel for efficiency.
    """

    file_pattern_re = r".+\.pdf"
    priority: int = 10  # Higher priority than regular PdfSplitter

    # LLM prompt for structured markdown extraction
    EXTRACTION_PROMPT = (Path(__file__).parent / "image_to_md_prompt.md").read_text()

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a PDF file into documents using LLM-based vision extraction.

        Converts PDF pages to images, processes them with Azure OpenAI vision model
        to extract structured markdown, then splits the result into documents.

        :param file_path: Path to the PDF file to be split.
        :return: A list of Document objects resulting from the split.
        """
        # Convert PDF to images
        images = cls._convert_pdf_to_images(file_path)

        # Process all pages with LLM
        combined_markdown = cls._process_all_pages(images)

        # Split the combined markdown using MarkdownTextSplitter
        text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)

        # Create a temporary document to split
        temp_doc = Document(page_content=combined_markdown, metadata={"source": file_path})
        documents = text_splitter.split_documents([temp_doc])

        return documents

    @classmethod
    def _convert_pdf_to_images(cls, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Image objects.

        Uses PyMuPDF to render each page as an image at 2x resolution.

        :param pdf_path: Path to the PDF file.
        :return: List of PIL Image objects, one per page.
        """
        images = []

        # Open PDF
        pdf_document = fitz.open(pdf_path)

        # Zoom matrix for better quality
        zoom_x = 2.0
        zoom_y = 2.0
        mat = fitz.Matrix(zoom_x, zoom_y)

        # Convert each page
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        # Close PDF
        pdf_document.close()

        return images

    @classmethod
    def _image_to_base64(cls, image: Image.Image) -> str:
        """Convert PIL Image to base64 string.

        :param image: PIL Image object.
        :return: Base64 encoded string of the image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    @classmethod
    def _process_page_with_llm(cls, image: Image.Image, page_num: int, chat_model) -> str:
        """Process a single page image with Azure OpenAI vision model.

        :param image: PIL Image object of the page.
        :param page_num: Page number (1-indexed for display).
        :param chat_model: Initialized chat model with vision capabilities.
        :return: Extracted markdown content for the page.
        """
        try:
            # Convert image to base64
            image_base64 = cls._image_to_base64(image)

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cls.EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]

            # Invoke the model
            response = chat_model.invoke(messages)

            # Extract content
            markdown_content = response.content

            # return f"\n\n## Page {page_num}\n\n{markdown_content}\n"
            return "\n" + markdown_content + "\n"

        except Exception as e:
            return f"\n\n## Page {page_num}\n\n[ERROR: Page {page_num} processing failed - {str(e)}]\n"

    @classmethod
    def _process_all_pages(cls, images: List[Image.Image]) -> str:
        """Process all pages in parallel using ThreadPoolExecutor.

        Processes up to 10 pages concurrently to speed up extraction.

        :param images: List of PIL Image objects.
        :return: Combined markdown content from all pages.
        """
        # Initialize Azure OpenAI or OpenAI model using init_chat_model
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            chat_model = init_chat_model(
                model="gpt-4.1",
                model_provider="azure_openai",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                max_tokens=16384,
            )
        else:
            chat_model = init_chat_model(
                model="gpt-4.1",
                model_provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        # Process pages in parallel with max 10 workers
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(cls._process_page_with_llm, image, idx + 1, chat_model): idx
                for idx, image in enumerate(images)
            }

            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    result = future.result()
                    results[page_idx] = result
                except Exception as e:
                    error_msg = f"\n\n## Page {page_idx + 1}\n\n[ERROR: Page {page_idx + 1} failed - {str(e)}]\n"
                    results[page_idx] = error_msg

        # Combine results in original page order
        combined_markdown = "".join(results[i] for i in sorted(results.keys()))
        return combined_markdown


@dataclass(eq=False)
class CsvSplitter(FileSplitter):
    """Splits CSV files into documents.

    This class provides functionality to load and split CSV files
    into Document objects using the appropriate CSV loader.
    """

    file_pattern_re = r".+\.(csv)"
    priority: int = 1

    @classmethod
    def analyze_csv_first_line(cls, file_path) -> Dict:
        """Analyze the first line of a CSV file to determine its structure.

        This method reads the first line of a CSV file to deduce the delimiter,
        quote character, and field names using the `csv.Sniffer` class. If the
        dialect cannot be detected, it defaults to using a comma as the delimiter
        and a double quote as the quote character.

        :param file_path: The path to the CSV file to analyze.
        :return: A dictionary containing the delimiter, field names, and optionally
                 the quote character.
        :raises FileNotFoundError: If the file at the specified path does not exist.
        :raises IOError: If there is an error opening or reading the file.
        """
        with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
            # Read the first line of the file
            first_line = csvfile.readline().strip()

            # Use the Sniffer class to deduce the delimiter and quote character
            sniffer = csv.Sniffer()
            try:
                # Deduce the dialect of the CSV
                dialect = sniffer.sniff(first_line)
            except csv.Error:
                # Default to comma if the sniffing fails
                dialect = csv.Dialect()
                dialect.delimiter = ","
                dialect.quotechar = '"'

            # Determine column names using the detected dialect
            csv_reader = csv.reader([first_line], dialect)
            column_names = next(csv_reader)

        # Return the extracted information
        d = {"delimiter": dialect.delimiter, "fieldnames": column_names}
        if dialect.quotechar:
            d.update(quotechar=dialect.quotechar)
        return d

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a CSV file into a list of Document objects.

        This method analyzes the first line of a CSV file to determine its structure
        and then loads the file into Document objects using the determined structure.

        :param file_path: The path to the CSV file to be processed.
        :return: A list of Document objects derived from the CSV file.
        """
        d = cls.analyze_csv_first_line(file_path)
        loader = CSVLoader(file_path, csv_args=d)
        return loader.load()
