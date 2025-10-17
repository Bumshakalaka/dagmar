"""Text and log file splitter.

This module provides functionality to load and split text-based files
into smaller document chunks using a text splitter.
"""

import logging
from dataclasses import dataclass
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dagmar.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class TxtSplitter(BaseSplitter):
    """Splits text and log files into documents.

    This class provides functionality to load and split text-based files
    into smaller document chunks using a character-based text splitter.
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
        logger.info(f"Processing text file: {file_path}")
        try:
            loader = TextLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)
            documents = loader.load_and_split(text_splitter=text_splitter)
            logger.debug(f"Text file split into {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            raise
