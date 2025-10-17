"""PDF file splitter with text extraction.

This module provides functionality to load and split PDF files into smaller
document chunks using a text splitter.
"""

import logging
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dagmar.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PdfSplitter(BaseSplitter):
    """Splits PDF files into documents.

    This class provides functionality to load and split PDF files into smaller
    document chunks using a character-based text splitter.
    """

    file_pattern_re = r".+\.pdf"
    priority: int = 1
    use_cache: bool = True

    @classmethod
    def split(cls, file_path: str) -> list[Document]:
        """Split a PDF file into documents.

        Loads a PDF file and splits it into smaller chunks using a character-based
        text splitter for further processing. Uses caching to avoid reprocessing unchanged files.

        :param file_path: Path to the PDF file to be split.
        :return: A list of Document objects resulting from the split.
        """
        logger.info(f"Processing PDF file: {file_path}")
        try:
            # Check cache first if caching is enabled
            extracted_text = None
            if cls.use_cache:
                extracted_text = cls._load_from_cache(file_path)
                if extracted_text is not None:
                    logger.info(f"Using cached content for {file_path}")
                else:
                    logger.debug(f"No cache found for {file_path}")

            # If no cache hit, extract text from PDF
            if extracted_text is None:
                loader = PyPDFLoader(file_path, extraction_mode="plain", extract_images=False)
                documents = loader.load()
                # Combine all page content into a single text
                extracted_text = "\n\n".join([doc.page_content for doc in documents])

                # Save to cache if caching is enabled
                if cls.use_cache:
                    cls._save_to_cache(file_path, extracted_text)

            # Split the extracted text using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

            # Create a temporary document to split
            temp_doc = Document(page_content=extracted_text, metadata={"source": file_path})
            documents = text_splitter.split_documents([temp_doc])
            logger.debug(f"PDF split into {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}")
            raise
