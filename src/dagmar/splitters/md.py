"""Markdown file splitter.

This module provides functionality to load and split markdown files
into smaller document chunks using a markdown-aware text splitter.
"""

import logging
from dataclasses import dataclass
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from dagmar.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class MdSplitter(BaseSplitter):
    """Splits markdown files into documents.

    This class provides functionality to load and split markdown files
    into smaller document chunks using a markdown-aware text splitter.
    """

    file_pattern_re = r".+\.md"
    priority: int = 1

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a markdown file into documents.

        Loads a markdown file and splits it into smaller chunks using a markdown-aware
        text splitter for further processing.

        :param file_path: Path to the markdown file to be split.
        :return: A list of Document objects resulting from the split.
        """
        logger.info(f"Processing markdown file: {file_path}")
        try:
            loader = TextLoader(file_path)
            text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
            documents = loader.load_and_split(text_splitter=text_splitter)
            logger.debug(f"Markdown file split into {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Failed to process markdown file {file_path}: {e}")
            raise
