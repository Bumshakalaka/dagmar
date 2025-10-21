"""LLM-based file splitter with generic image-to-markdown conversion.

This module provides an abstract base class for LLM-based file splitters that
can convert various file formats to images and extract structured markdown content
using vision models.
"""

import logging
import os
import re
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from PIL import Image

from dagmar.md_fixer import MarkdownFixer
from dagmar.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())


@dataclass(eq=False)
class _LlmBaseSplitter(BaseSplitter):
    """Abstract base class for LLM-based file splitters.

    This class provides a generic framework for converting files to images
    and extracting structured markdown content using LLM vision models.
    Subclasses implement file-specific image conversion logic.

    Processes up to 10 pages/images in parallel for efficiency.
    """

    # LLM prompt for structured markdown extraction
    EXTRACTION_PROMPT = (Path(__file__).parent / "image_to_md_prompt.md").read_text()

    @classmethod
    @abstractmethod
    def _convert_to_images(cls, file_path: str) -> List[Image.Image]:
        """Convert file to PIL Image objects.

        This method must be implemented by subclasses to handle file-specific
        conversion logic (e.g., PDF to images, DOCX to images, etc.).

        :param file_path: Path to the file to convert.
        :return: List of PIL Image objects, one per page/slide.
        """
        raise NotImplementedError

    @classmethod
    def _resize_image(cls, image: Image.Image, max_dimension: int = 1024) -> Image.Image:
        """Resize image to fit within max_dimension while preserving aspect ratio.

        Only downsamples images; does not upscale.

        :param image: PIL Image object to resize.
        :param max_dimension: Maximum allowed dimension (width or height). Default 1024px.
        :return: Resized PIL Image object.
        """
        width, height = image.size

        # Check if resizing is needed (only downsize)
        if width <= max_dimension and height <= max_dimension:
            return image

        # Calculate scaling factor to fit within max_dimension
        scale = min(max_dimension / width, max_dimension / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @classmethod
    def split(cls, file_path: str) -> List[Document]:
        """Split a file into documents using LLM-based vision extraction.

        Converts file to images, processes them with LLM vision model
        to extract structured markdown, then splits the result into documents.
        Uses caching to avoid reprocessing unchanged files.

        :param file_path: Path to the file to be split.
        :return: A list of Document objects resulting from the split.
        """
        logger.info(f"Processing file with LLM vision: {file_path}")
        try:
            # Check cache first if caching is enabled
            combined_markdown = None
            if cls.use_cache:
                combined_markdown = cls._load_from_cache(file_path)
                if combined_markdown is not None:
                    logger.info(f"Using cached content for {file_path}")
                else:
                    logger.debug(f"No cache found for {file_path}")

            # If no cache hit, process with LLM
            if combined_markdown is None:
                # Convert file to images
                images = cls._convert_to_images(file_path)
                logger.info(f"Converted to {len(images)} images")

                # Process all images with LLM
                combined_markdown = cls._process_all_pages(images)
                logger.info(f"Processed {len(images)} images with LLM")

                # Save to cache if caching is enabled
                if cls.use_cache:
                    cls._save_to_cache(file_path, combined_markdown)

            fixer = MarkdownFixer()
            combined_markdown = fixer.process_content(combined_markdown, Path(file_path).stem)
            logger.info(f"Fixed markdown for {file_path}")

            # Keep chunking by H2 to preserve semantic sections
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "H2")], strip_headers=False)
            documents = text_splitter.split_text(combined_markdown)

            # Extract page markers per chunk, add metadata, and strip markers from content
            prev_page_ids = set()
            for doc in documents:
                doc.metadata["source"] = file_path

                page_ids = set()
                for m in re.findall(r"<!--\s*PAGE:(\d+)\s*-->", doc.page_content):
                    page_ids.add(int(m))

                if page_ids:
                    pages_sorted = sorted(page_ids)
                    prev_page_ids = pages_sorted
                    doc.metadata["page_nums"] = pages_sorted
                elif prev_page_ids:
                    doc.metadata["page_nums"] = prev_page_ids

                # Remove markers so they don't affect embeddings
                doc.page_content = re.sub(r"<!--\s*PAGE:\d+\s*-->\s*", "", doc.page_content)

            logger.info(f"LLM-processed file {file_path} split into {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise

    @classmethod
    def _process_page_with_llm(cls, image: Image.Image, page_num: int, chat_model) -> str:
        """Process a single image with LLM vision model.

        :param image: PIL Image object of the page.
        :param page_num: Page number (1-indexed for display).
        :param chat_model: Initialized chat model with vision capabilities.
        :return: Extracted markdown content for the page.
        """
        try:
            # Convert image to base64
            import base64
            import io

            buffered = io.BytesIO()
            resized_image = cls._resize_image(image)
            resized_image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()

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

            markdown_content = response.content
            logger.info(f"Successfully processed page {page_num}")

            return f"{markdown_content}\n<!-- PAGE:{page_num} -->\n"

        except Exception as e:
            logger.error(f"Failed to process page {page_num}: {e}")
            return f"[ERROR: Page {page_num} processing failed - {str(e)}]\n<!-- PAGE:{page_num} -->\n"

    @classmethod
    def _process_all_pages(cls, images: List[Image.Image]) -> str:
        """Process all images in parallel using ThreadPoolExecutor.

        Processes up to 10 images concurrently to speed up extraction.

        :param images: List of PIL Image objects.
        :return: Combined markdown content from all images.
        """
        logger.info(f"Starting parallel processing of {len(images)} images")

        # Initialize Azure OpenAI or OpenAI model using init_chat_model
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            logger.debug("Using Azure OpenAI model")
            chat_model = init_chat_model(
                model="gpt-4.1",
                model_provider="azure_openai",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                max_tokens=16384,
            )
        else:
            logger.debug("Using OpenAI model")
            chat_model = init_chat_model(
                model="gpt-4.1",
                model_provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        # Process images in parallel with max 10 workers
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
                    logger.error(f"Page {page_idx + 1} processing failed: {e}")
                    error_msg = f"[ERROR: Page {page_idx + 1} failed - {str(e)}]\n<!-- PAGE:{page_idx + 1} -->\n"
                    results[page_idx] = error_msg

        # Combine results in original page order
        combined_markdown = "".join(results[i] for i in sorted(results.keys()))
        logger.info(f"Completed parallel processing of {len(images)} images")
        return combined_markdown
