"""PDF splitter using LLM vision model for structured markdown extraction.

This module provides functionality to convert PDF pages to images and use
Azure OpenAI vision model to extract structured markdown content.
"""

import io
import logging
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from dagmar.splitters.llm_base import _LlmBaseSplitter

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PdfLlmSplitter(_LlmBaseSplitter):
    """Splits PDF files using LLM vision model for structured markdown extraction.

    This class converts PDF pages to images and uses Azure OpenAI vision model
    to extract structured markdown content, which is then split into documents.
    """

    file_pattern_re = r".+\.pdf"
    priority: int = 10  # Higher priority than regular PdfSplitter
    use_cache: bool = True

    @classmethod
    def _convert_to_images(cls, file_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Image objects.

        Uses PyMuPDF to render each page as an image at 2x resolution.

        :param file_path: Path to the PDF file.
        :return: List of PIL Image objects, one per page.
        """
        logger.debug(f"Converting PDF to images: {file_path}")
        images = []

        # Open PDF
        pdf_document = fitz.open(file_path)

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
