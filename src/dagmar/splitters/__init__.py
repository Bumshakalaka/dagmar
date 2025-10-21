"""Vector store file splitter utilities.

This module provides functionality to split various file types into documents
for vector store indexing. It includes automatic file type detection and
specialized splitters for different document formats.

All splitters are automatically registered and can be retrieved using the get_splitter() function.
"""

import logging
import re
from typing import Type

from dagmar.splitters.base import FILE_SPLITTERS, BaseSplitter
from dagmar.splitters.csv import CsvSplitter
from dagmar.splitters.md import MdSplitter
from dagmar.splitters.pdf import PdfSplitter
from dagmar.splitters.pdf_llm import PdfLlmSplitter
from dagmar.splitters.pptx import PptxLlmSplitter
from dagmar.splitters.txt import TxtSplitter

logger = logging.getLogger(__name__)

# Export all splitters and the FILE_SPLITTERS registry for public API
__all__ = [
    "BaseSplitter",
    "PdfSplitter",
    "PdfLlmSplitter",
    "PptxLlmSplitter",
    "TxtSplitter",
    "MdSplitter",
    "CsvSplitter",
    "FILE_SPLITTERS",
    "get_splitter",
]


def get_splitter(file_path: str) -> Type[BaseSplitter]:
    """Retrieve the appropriate FileSplitter for a given file path.

    This function matches the file path against registered file patterns
    in FILE_SPLITTERS and returns the FileSplitter with the highest priority.

    :param file_path: Path to the file for which a splitter is needed.
    :return: The FileSplitter class with the highest priority that matches the file path.
    :raises ValueError: If no matching splitter is found for the given file path.
    """
    logger.debug(f"Finding splitter for file: {file_path}")
    ret = []
    for _, obj in FILE_SPLITTERS.items():
        if re.match(obj.file_pattern_re, file_path):
            ret.append([obj.priority, obj])
    if not ret:
        logger.error(f"No splitter found for file: '{file_path}'. Supported splitters: {list(FILE_SPLITTERS.keys())}")
        raise AttributeError(
            f"No splitter found for file: '{file_path}'. Supported splitters: {list(FILE_SPLITTERS.keys())}"
        )

    selected_splitter = sorted(ret, key=lambda x: x[0])[-1][1]
    logger.debug(f"Selected splitter: {selected_splitter.__name__} for file: {file_path}")
    return selected_splitter
