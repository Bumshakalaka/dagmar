"""CSV file splitter.

This module provides functionality to load and split CSV files
into Document objects using the appropriate CSV loader with automatic dialect detection.
"""

import csv
import logging
from dataclasses import dataclass
from typing import Dict, List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from dagmar.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class CsvSplitter(BaseSplitter):
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
        logger.debug(f"Analyzing CSV structure for file: {file_path}")
        with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
            # Read the first line of the file
            first_line = csvfile.readline().strip()

            # Use the Sniffer class to deduce the delimiter and quote character
            sniffer = csv.Sniffer()
            try:
                # Deduce the dialect of the CSV
                dialect = sniffer.sniff(first_line)
                logger.debug(f"Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
            except csv.Error:
                # Default to comma if the sniffing fails
                logger.debug("CSV dialect detection failed, using defaults")
                dialect = csv.Dialect()
                dialect.delimiter = ","
                dialect.quotechar = '"'

            # Determine column names using the detected dialect
            csv_reader = csv.reader([first_line], dialect)
            column_names = next(csv_reader)
            logger.debug(f"Detected {len(column_names)} columns: {column_names}")

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
        logger.info(f"Processing CSV file: {file_path}")
        try:
            d = cls.analyze_csv_first_line(file_path)
            loader = CSVLoader(file_path, csv_args=d)
            documents = loader.load()
            logger.debug(f"CSV file split into {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
            raise
