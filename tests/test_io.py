"""Tests for pipeline.io module."""

import os
import tempfile
import pytest
import torch
import h5py
from pathlib import Path

from pipeline.io import IOHandler


class TestIOHandler:

    def test_fetch_files_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            Path(subdir, "file3.txt").touch()

            files = IOHandler.fetch_files(tmpdir)
            assert len(files) == 3

    def test_fetch_files_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert IOHandler.fetch_files(tmpdir) == []

    def test_fetch_folders_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "folder1"))
            os.makedirs(os.path.join(tmpdir, "folder2"))
            os.makedirs(os.path.join(tmpdir, "folder1", "nested"))

            folders = IOHandler.fetch_folders(tmpdir)
            assert len(folders) == 3

    def test_fetch_folders_with_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "folder1"))
            os.makedirs(os.path.join(tmpdir, "folder2"))

            folders = IOHandler.fetch_folders(tmpdir, filter_func=lambda x: "folder1" in x)
            assert len(folders) == 1

    def test_save_and_load_h5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "sequence": [torch.tensor([1, 2, 3], dtype=torch.int32)],
                "labels": [torch.tensor([4, 5], dtype=torch.int32)],
            }
            IOHandler.save_h5(tmpdir, "test", tensor_group)

            assert os.path.exists(os.path.join(tmpdir, "test.h5"))
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            assert "sequence" in loaded
            assert "labels" in loaded
            assert torch.equal(loaded["sequence"][0], torch.tensor([1, 2, 3], dtype=torch.int32))
            assert torch.equal(loaded["labels"][0], torch.tensor([4, 5], dtype=torch.int32))

    def test_save_h5_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")
            IOHandler.save_h5(output_dir, "test", {"data": [torch.tensor([1, 2, 3])]})
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "test.h5"))

    def test_load_h5_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, data in enumerate([[1, 2, 3], [4, 5, 6]]):
                h5_path = os.path.join(tmpdir, f"file{i}.h5")
                with h5py.File(h5_path, 'w') as f:
                    grp = f.create_group("data")
                    grp.create_dataset('data_0', data=data)

            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            assert len(loaded["data"]) == 2

    def test_load_h5_with_rglob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            h5_path = os.path.join(subdir, "nested.h5")
            with h5py.File(h5_path, 'w') as f:
                grp = f.create_group("test")
                grp.create_dataset('data_0', data=[1, 2])

            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            assert "test" in loaded
            assert len(loaded["test"]) == 1

    def test_save_h5_multiple_tensors_per_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "batch": [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])],
            }
            IOHandler.save_h5(tmpdir, "multi", tensor_group)
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            assert len(loaded["batch"]) == 3
