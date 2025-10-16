"""Markdown linter and fixer.

Detects and corrects:
  • missing top-level H1 heading
  • multiple H1 headers (keeps first, converts others to H2)
  • H1 not at first line (converts to H2, adds new H1)
  • heading-level jumps (H2 → H4, etc.)
  • incorrect header spacing (ensures exactly one space after #)
  • numbered headers with wrong levels (e.g., ## 3.1.2.1 → #### 3.1.2.1)
  • missing blank line before a heading
  • trailing whitespace
  • list item formatting (markers, spacing, indentation)
  • multiple consecutive blank lines (reduces 3+ to 2)

Optional cleaners (--cleaner flag):
  • toc: Remove table of contents sections

Usage:
    python md_fixer.py --input file.md [--output output.md] [--in-place]
    python md_fixer.py --input file.md --cleaner toc
    python md_fixer.py --input file.md --cleaner toc,future_cleaner --in-place
"""

import argparse
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from dagmar.logging_config import setup_logging

logger = logging.getLogger(__name__)


class MarkdownCleaner:
    """Base class for markdown cleaners."""

    def clean(self, lines: List[str], issues: List[str]) -> List[str]:
        """Clean markdown lines.

        Args:
            lines: List of markdown lines to clean
            issues: List to append issue messages to

        Returns:
            Cleaned list of lines

        """
        raise NotImplementedError("Subclasses must implement clean()")


class TocRemoverCleaner(MarkdownCleaner):
    """Removes table of contents sections from markdown."""

    def clean(self, lines: List[str], issues: List[str]) -> List[str]:
        """Remove TOC sections from markdown lines.

        Args:
            lines: List of markdown lines
            issues: List to append issue messages to

        Returns:
            List of lines with TOC sections removed

        """
        if not lines:
            return lines

        result = []
        i = 0
        toc_removed_count = 0

        while i < len(lines):
            line = lines[i]

            # Check for HTML TOC comments: <!-- toc --> or <!-- TOC -->
            if re.match(r"^\s*<!--\s*/?toc\s*-->", line, re.IGNORECASE):
                # Find the closing tag
                if re.match(r"^\s*<!--\s*toc\s*-->", line, re.IGNORECASE):
                    # Opening tag, skip until closing
                    toc_removed_count += 1
                    issues.append(f"Line {i + 1}: Removed HTML TOC section")
                    i += 1
                    while i < len(lines):
                        if re.match(r"^\s*<!--\s*/toc\s*-->", lines[i], re.IGNORECASE):
                            i += 1
                            break
                        i += 1
                    continue
                else:
                    # Standalone closing tag, skip it
                    i += 1
                    continue

            # Check for markdown TOC markers
            if re.match(r"^\s*\[?\[?TOC\]?\]?|\{:toc\}", line, re.IGNORECASE):
                issues.append(f"Line {i + 1}: Removed markdown TOC marker")
                toc_removed_count += 1
                i += 1
                continue

            # Check for standalone TOC-style lines (numbered with dots and page number at end)
            # Examples: "1.1 Scope ........ 20" or "  1.1.1 Application ... 20"
            if self._is_toc_entry_line(line):
                toc_removed_count += 1
                i += 1
                continue

            # Check for TOC header sections (H1 or H2)
            header_match = re.match(r"^(#{1,2})\s+(.+)", line)
            if header_match:
                hashes, title = header_match.groups()
                header_level = len(hashes)
                title_lower = title.lower().strip()

                # Check if it's a TOC-like title (but not overview)
                is_toc_title = False
                if "overview" not in title_lower:
                    if title_lower in ["contents", "table of contents", "toc", "index"]:
                        is_toc_title = True
                    elif title_lower == "content":
                        is_toc_title = True
                    # Check if header ends with a page number (likely a TOC entry formatted as header)
                    elif re.search(r"\s+\d+\s*$", title):
                        is_toc_title = True

                if is_toc_title:
                    # Scan ahead and collect first 20 non-empty lines (or until next header of same/higher level)
                    sample_lines = []
                    scan_idx = i + 1
                    while scan_idx < len(lines) and len(sample_lines) < 20:
                        scan_line = lines[scan_idx]

                        # Stop if we hit a header of same or higher level
                        next_header_match = re.match(r"^(#{1,6})\s", scan_line)
                        if next_header_match:
                            next_level = len(next_header_match.group(1))
                            if next_level <= header_level:
                                break

                        # Collect non-empty lines for analysis
                        if scan_line.strip():
                            sample_lines.append(scan_line)

                        scan_idx += 1

                    # Check if 70%+ of sample lines are TOC entries
                    if sample_lines:
                        toc_entry_count = sum(1 for line in sample_lines if self._is_toc_entry_line(line))
                        toc_percentage = toc_entry_count / len(sample_lines)

                        if toc_percentage >= 0.5:
                            # Find the actual end of the TOC section (next header of same/higher level)
                            toc_end = i + 1
                            while toc_end < len(lines):
                                end_header_match = re.match(r"^(#{1,6})\s", lines[toc_end])
                                if end_header_match:
                                    end_level = len(end_header_match.group(1))
                                    if end_level <= header_level:
                                        break
                                toc_end += 1

                            # Remove the entire TOC section
                            lines_removed = toc_end - i
                            issues.append(
                                f"Line {i + 1}: Removed TOC section '{title}' "
                                f"({lines_removed} lines, {toc_percentage:.0%} TOC entries)"
                            )
                            toc_removed_count += 1
                            i = toc_end

                            # Remove trailing blank lines after TOC
                            while i < len(lines) and lines[i] == "":
                                i += 1
                            continue

                    # If not enough evidence it's a TOC, but header is empty, still remove it
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        next_is_header = re.match(r"^#{1,6}\s", next_line)
                        if next_is_header or not next_line.strip():
                            issues.append(f"Line {i + 1}: Removed empty TOC header '{title}'")
                            toc_removed_count += 1
                            i += 1
                            # Remove trailing blank lines
                            while i < len(lines) and lines[i] == "":
                                i += 1
                            continue

            result.append(line)
            i += 1

        return result

    def _is_toc_entry_line(self, line: str) -> bool:
        """Check if a line is a TOC entry with page numbering or hierarchical numbering.

        Examples:
            1.1 Scope ........ 20
              1.1.1 Application ... 20
            2.1 Overview ........................... 50
            3.1 BLE/EP/CONN/REG/BV/01: An unregistered Endpoint... 26
            3. 8.3 Test Procedure ......................................................... 36

        Args:
            line: Line to check

        Returns:
            True if line appears to be a TOC entry

        """
        # Pattern 1: Lines with dots and page numbers at the end
        # "1.1 Scope ........ 20"
        if re.search(r"\.{2,}\s+\d+\s*$", line):
            # Check if it starts with numbering pattern (with optional leading whitespace)
            if re.match(r"^\s*\d+(\.\d+)*\s", line):
                return True

        # Pattern 2: Lines with hierarchical numbering (at least 2 levels)
        # and ending with a number (page number without dots)
        # "  1.1.1 Application 20" or "1.2 General Test Requirements 20"
        if re.match(r"^\s*\d+\.\d+(\.\d+)*\s+.+\s+\d+\s*$", line):
            return True

        # Pattern 3: Lines with hierarchical numbering (2+ levels) that contain colons
        # Often TOC entries for test cases like "3.1 BLE/EP/CONN/REG/BV/01: Description"
        if re.match(r"^\s*\d+\.\d+(\.\d+)*\s+[^:]+:.+", line):
            return True

        # Additional pattern (per prompt and file_context_0):
        # E.g.: 3. 8.3 Test Procedure ......................................................... 36
        # Allow optional space-dot at start, space, then a digit+dot, digit sequence, then maybe more, then some text,
        # then dots and a number
        if re.match(r"^\s*\d+\.\s*\d+(\.\d+)*\s+.+\.{2,}\s+\d+\s*$", line):
            return True

        return False


class MarkdownFixer:
    """Markdown linter and formatter."""

    def __init__(self, cleaners: Optional[List[MarkdownCleaner]] = None):
        """Initialize the MarkdownFixer.

        Args:
            cleaners: List of MarkdownCleaner instances to apply after fixing

        """
        self.issues: List[str] = []
        self.fixed_lines: List[str] = []
        self.prev_header_level: int = 0
        self.cleaners: List[MarkdownCleaner] = cleaners or []

    def process_content(self, content: str, file_name: Optional[str] = None) -> str:
        """Process markdown content and fix issues.

        Args:
            content: Markdown content to process
            file_name: Name of the file to use for the temporary file

        Returns:
            Processed markdown content

        """
        if file_name:
            prefix = file_name
        else:
            prefix = "dagmar_"
        with (
            tempfile.NamedTemporaryFile(
                delete=True, suffix=".md", mode="w+", encoding="utf-8", prefix=prefix
            ) as temp_file,
            tempfile.NamedTemporaryFile(
                delete=True, suffix=".md", mode="w+", encoding="utf-8", prefix="output_"
            ) as temp_file_output,
        ):
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
            self.process_file(Path(temp_file_path), Path(temp_file_output.name))
            return temp_file_output.read()

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Process a markdown file and fix issues.

        Args:
            input_path: Path to input markdown file
            output_path: Path to output markdown file

        Returns:
            True if issues were found and fixed, False otherwise

        """
        logger.info(f"Processing markdown file: {input_path}")
        # Read input file
        try:
            content = input_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)

        lines = content.splitlines(keepends=False)
        logger.debug(f"Read {len(lines)} lines from file")

        # Process lines
        self._fix_lines(lines)

        # Check and add missing H1 (after processing lines)
        self._add_missing_h1(input_path.stem)

        # Remove excess blank lines (3+ consecutive → 2)
        self._remove_excess_blank_lines()

        # Apply cleaners
        for cleaner in self.cleaners:
            logger.debug(f"Applying cleaner: {cleaner.__class__.__name__}")
            self.fixed_lines = cleaner.clean(self.fixed_lines, self.issues)
            # Clean up any excessive blank lines that might have been created
            self._remove_excess_blank_lines()

        # Report and write results
        output_path.write_text("\n".join(self.fixed_lines) + "\n", encoding="utf-8")
        if self.issues:
            logger.info(f"Found {len(self.issues)} issues to fix")
            self._report_issues()
            return True
        else:
            logger.info("No issues found in markdown file")
            return False

    def _fix_lines(self, lines: List[str]) -> None:
        """Process each line and apply fixes.

        Args:
            lines: List of lines from the input file

        """
        for i, raw_line in enumerate(lines, 1):
            # Check for trailing whitespace
            line = raw_line.rstrip()
            line = self._clean_line(line)
            if line != raw_line:
                self.issues.append(f"Line {i}: trailing whitespace removed")
            if self._skip_line(line):
                self.issues.append(f"Line {i}: NO DATA AVAILABLE detected")
                continue
            # Check and fix headers
            header_result = self._check_header(line, i)
            if header_result:
                line = header_result

            # Check and fix list items
            list_result = self._check_list_item(line, i)
            if list_result:
                line = list_result

            self.fixed_lines.append(line)

    def _clean_line(self, line: str) -> str:
        """Clean a line of trailing whitespace and other issues.

        Args:
            line: The line to clean

        Returns:
            Cleaned line

        """
        replace_strings = [
            ("&nbsp;", " "),
            ("&quot;", '"'),
            ("&apos;", "'"),
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
        ]
        for old, new in replace_strings:
            line = line.replace(old, new)
        return line

    def _skip_line(self, line: str) -> bool:
        """Check if a line should be skipped.

        Args:
            line: The line to check

        Returns:
            True if the line should be skipped, False otherwise

        """
        if line.strip() == "NO DATA AVAILABLE":
            return True
        return False

    def _check_header(self, line: str, line_num: int) -> Optional[str]:
        """Check and fix header formatting.

        Args:
            line: The line to check
            line_num: Line number for reporting

        Returns:
            Fixed line if it's a header, None otherwise

        """
        # Match headers with any spacing (or no spacing)
        header_match = re.match(r"^(#{1,6})(\s*)(.*)", line)
        if not header_match:
            return None

        hashes, spacing, content = header_match.groups()
        level = len(hashes)

        # Check if header content starts with numbering (e.g., "3.1.2.1")
        # The number of dots + 1 indicates the minimum expected header level
        numbering_match = re.match(r"^(\d+(?:\.\d+)*)(?:\s|$)", content)
        if numbering_match:
            numbering = numbering_match.group(1)
            # Count dots to determine minimum expected level
            dot_count = numbering.count(".")
            min_expected_level = dot_count + 1

            # Only adjust if:
            # 1. The header is under-nested (level < min_expected_level)
            # 2. The numbering has at least 1 dot (hierarchical numbering like "3.1", not just "3")
            # This prevents incorrectly forcing "## 1 Introduction" to become "# 1 Introduction"
            if dot_count >= 1 and level < min_expected_level and min_expected_level <= 6:
                self.issues.append(
                    f"Line {line_num}: header level adjusted from H{level} to H{min_expected_level} "
                    f'based on numbering "{numbering}"'
                )
                level = min_expected_level
                hashes = "#" * level

        # Check for heading level jumps
        if self.prev_header_level and level > self.prev_header_level + 1:
            self.issues.append(f"Line {line_num}: heading jumps H{self.prev_header_level} → H{level}")
        self.prev_header_level = level

        # Check for blank line before header
        if self.fixed_lines and self.fixed_lines[-1] != "":
            self.issues.append(f"Line {line_num}: missing blank line before heading")
            self.fixed_lines.append("")

        # Normalize header spacing (exactly one space)
        if not content:
            # Header with no content
            return line

        if spacing != " ":
            self.issues.append(f"Line {line_num}: header spacing normalized")
            return f"{hashes} {content}"

        return f"{hashes} {content}"

    def _check_list_item(self, line: str, line_num: int) -> Optional[str]:
        """Check and fix list item formatting.

        Args:
            line: The line to check
            line_num: Line number for reporting

        Returns:
            Fixed line if it's a list item, None otherwise

        """
        # Match bullet lists with leading spaces (indentation)
        bullet_match = re.match(r"^(\s*)([-*+])(\s*)(.*)", line)
        if bullet_match:
            indent, marker, spacing, content = bullet_match.groups()

            # Calculate proper indentation (should be multiple of 2)
            indent_len = len(indent)
            proper_indent_len = (indent_len // 2) * 2

            needs_fix = False
            if indent_len != proper_indent_len:
                needs_fix = True
            if marker != "-":
                needs_fix = True
            if spacing != " " and content:
                needs_fix = True

            if needs_fix:
                self.issues.append(f"Line {line_num}: list item formatting fixed")
                proper_indent = " " * proper_indent_len
                if content:
                    return f"{proper_indent}- {content}"
                else:
                    return f"{proper_indent}-"

            return line

        # Match numbered lists with leading spaces
        numbered_match = re.match(r"^(\s*)(\d+)\.\s*(.*)", line)
        if numbered_match:
            indent, number, content = numbered_match.groups()

            # Calculate proper indentation (should be multiple of 2)
            indent_len = len(indent)
            proper_indent_len = (indent_len // 2) * 2

            if indent_len != proper_indent_len:
                self.issues.append(f"Line {line_num}: list item indentation fixed")
                proper_indent = " " * proper_indent_len
                return f"{proper_indent}{number}. {content}"

            return line

        return None

    def _add_missing_h1(self, filename_stem: str) -> None:
        """Add H1 heading if missing or ensure H1 is at the first line.

        If H1 exists but is not at the first line, converts all H1s to H2s
        and adds a new H1 at the beginning.

        Args:
            filename_stem: Filename stem to use for generating H1

        """
        # Find all H1 headers in fixed_lines and their positions
        h1_positions = []
        for i, line in enumerate(self.fixed_lines):
            if re.match(r"^#\s", line):
                h1_positions.append(i)

        # Case 1: No H1 exists - add one at the beginning
        if not h1_positions:
            self.issues.append("Missing H1 heading - added automatically")
            title = filename_stem.replace("_", " ").replace("-", " ").title()
            self.fixed_lines.insert(0, f"# {title}")
            self.fixed_lines.insert(1, "")
            return

        # Case 2: H1 exists but not at first line (or after initial blank lines)
        # Find the first non-empty line position
        first_content_pos = 0
        for i, line in enumerate(self.fixed_lines):
            if line.strip():
                first_content_pos = i
                break

        # Check if the first content line is an H1
        if first_content_pos not in h1_positions:
            # H1 exists but not at the first line - convert all H1s to H2s
            self.issues.append("H1 not at first line - converting all H1s to H2s and adding new H1")
            for i in h1_positions:
                # Convert "# Title" to "## Title"
                self.fixed_lines[i] = "#" + self.fixed_lines[i]

            # Add new H1 at the beginning
            title = filename_stem.replace("_", " ").replace("-", " ").title()
            self.fixed_lines.insert(0, f"# {title}")
            self.fixed_lines.insert(1, "")
        # Case 3: H1 is at the first line - check if there are multiple H1s
        elif len(h1_positions) > 1:
            # Multiple H1s exist - convert all except the first to H2
            self.issues.append(f"Multiple H1 headers found - converting {len(h1_positions) - 1} to H2")
            for i in h1_positions[1:]:
                # Convert "# Title" to "## Title"
                self.fixed_lines[i] = "#" + self.fixed_lines[i]

    def _remove_excess_blank_lines(self) -> None:
        """Reduce 3+ consecutive blank lines to maximum of 2."""
        result = []
        blank_count = 0

        for line in self.fixed_lines:
            if line == "":
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
                elif blank_count == 3:
                    self.issues.append(f"Reduced {blank_count}+ consecutive blank lines to 2")
            else:
                blank_count = 0
                result.append(line)

        self.fixed_lines = result

    def _report_issues(self) -> None:
        """Report found issues to stdout."""
        for msg in self.issues:
            logger.debug(f"  • {msg}")


def main() -> None:
    """Run the markdown linter and fixer from CLI."""
    parser = argparse.ArgumentParser(
        description="Markdown linter and fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input file.md
  %(prog)s --input file.md --output fixed.md
  %(prog)s --input file.md --in-place
  %(prog)s --input file.md --cleaner toc
  %(prog)s --input file.md --cleaner toc,future_cleaner --in-place
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input markdown file to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: <input>_fixed.md)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify input file directly (ignores --output)",
    )
    parser.add_argument(
        "--cleaner",
        type=str,
        help="Comma-separated list of cleaners to apply (available: toc)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.log_level)

    # Validate input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Determine output path
    if args.in_place:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        output_path = args.input.with_stem(args.input.stem + "_fixed")

    # Parse and instantiate cleaners
    cleaners = []
    if args.cleaner:
        logger.debug(f"Parsing cleaners: {args.cleaner}")
        cleaner_names = [name.strip().lower() for name in args.cleaner.split(",")]
        for name in cleaner_names:
            if name == "toc":
                cleaners.append(TocRemoverCleaner())
                logger.debug("Added TocRemoverCleaner")
            else:
                logger.warning(f"Unknown cleaner '{name}', skipping...")

    # Process file
    logger.info(f"Starting markdown processing with {len(cleaners)} cleaners")
    fixer = MarkdownFixer(cleaners=cleaners)
    fixer.process_file(args.input, output_path)


if __name__ == "__main__":
    main()
