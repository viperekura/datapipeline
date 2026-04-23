"""File system scanning utilities."""

import os
from pathlib import Path
from typing import Callable, List, Optional


class FileScanner:
    """Utility for scanning and filtering files in a directory tree.

    Example::

        scanner = FileScanner()
        files = scanner.scan("./data", suffix=".jsonl")
        folders = scanner.scan_folders("./data", filter_func=is_valid_dir)
    """

    @staticmethod
    def scan(
        directory: str,
        suffix: Optional[str] = None,
        recursive: bool = True,
        relative: bool = False,
    ) -> List[str]:
        """Scan directory for files matching criteria.

        Args:
            directory: Root directory to scan.
            suffix: Filter files by extension (e.g., ".jsonl").
            recursive: Whether to search subdirectories.
            relative: Return relative paths if True.

        Returns:
            Sorted list of file paths.
        """
        root_path = Path(directory)
        if not root_path.exists():
            return []

        if recursive:
            pattern = "**/*" + suffix if suffix else "**/*"
            files = [str(p) for p in root_path.glob(pattern) if p.is_file()]
        else:
            pattern = "*" + suffix if suffix else "*"
            files = [str(p) for p in root_path.glob(pattern) if p.is_file()]

        if relative:
            files = [str(Path(f).relative_to(root_path)) for f in files]

        return sorted(files)

    @staticmethod
    def scan_folders(
        directory: str,
        filter_func: Optional[Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> List[str]:
        """Scan directory for subdirectories.

        Args:
            directory: Root directory to scan.
            filter_func: Optional predicate to filter directories.
            recursive: Whether to search subdirectories.

        Returns:
            Sorted list of directory paths.
        """
        root_path = Path(directory)
        if not root_path.exists():
            return []

        if recursive:
            folders = [str(p) for p in root_path.rglob("*") if p.is_dir()]
        else:
            folders = [str(p) for p in root_path.glob("*") if p.is_dir()]

        if filter_func:
            folders = [f for f in folders if filter_func(f)]

        return sorted(folders)

    @staticmethod
    def group_by_extension(files: List[str]) -> dict[str, List[str]]:
        """Group files by their extension.

        Args:
            files: List of file paths.

        Returns:
            Dictionary mapping extension to list of files.
        """
        groups: dict[str, List[str]] = {}
        for f in files:
            ext = Path(f).suffix
            if ext not in groups:
                groups[ext] = []
            groups[ext].append(f)
        return groups
