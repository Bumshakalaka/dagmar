"""Base file splitter class with automatic registration and caching.

This module provides the base functionality for file splitters including
automatic subclass registration, file pattern matching, and document caching.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

FILE_SPLITTERS: Dict[str, Type["BaseSplitter"]] = {}


@dataclass
class BaseSplitter:
    """Base class for file splitters.

    This class serves as a base for specific file splitters. Subclasses are automatically
    registered in the `FILE_SPLITTERS` dictionary unless their class name starts with '_'.

    :param file_pattern_re: Regular expression pattern for matching file types.
    :param priority: Priority of the file splitter (higher = preferred for file type).
    :param use_cache: Whether to use caching for processed content.
    """

    file_pattern_re: str
    priority: int
    use_cache: bool = False

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in `FILE_SPLITTERS`.

        Subclasses with names not starting with '_' are added to the global `FILE_SPLITTERS`
        dictionary, allowing for easy access and management of different file splitters.
        """
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("_"):
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

    @classmethod
    def _get_cache_key(cls, file_path: str) -> str:
        """Generate cache key from filename and modification time.

        :param file_path: Path to the file.
        :return: Cache key in format {filename_without_ext}_{timestamp}.
        """
        file_path_obj = Path(file_path)
        mtime = int(file_path_obj.stat().st_mtime)
        return f"{file_path_obj.stem}_{mtime}"

    @classmethod
    def _get_cache_path(cls, file_path: str) -> Path:
        """Get the full path to the cached file.

        :param file_path: Path to the original file.
        :return: Path to the cached markdown file.
        """
        cache_key = cls._get_cache_key(file_path)
        cache_dir = Path("./processed_files")
        return cache_dir / f"{cache_key}.md"

    @classmethod
    def _load_from_cache(cls, file_path: str) -> Optional[str]:
        """Load processed content from cache if it exists.

        :param file_path: Path to the original file.
        :return: Cached markdown content if exists, None otherwise.
        """
        cache_path = cls._get_cache_path(file_path)
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_path}: {e}")
                return None
        return None

    @classmethod
    def _save_to_cache(cls, file_path: str, content: str) -> None:
        """Save processed content to cache.

        :param file_path: Path to the original file.
        :param content: Markdown content to cache.
        """
        cache_path = cls._get_cache_path(file_path)
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content, encoding="utf-8")
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_path}: {e}")
