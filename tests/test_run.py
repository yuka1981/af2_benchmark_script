import pytest
import os
import argparse
from src import run
from unittest.mock import patch, MagicMock

@pytest.fixture
def valid_args():
    return argparse.Namespace(
        input_fasta="test.fasta",
        description="test_description",
        mode="cpu",
        preset="monomer",
        platform="nvidia",
        msa="msa"
    )

@pytest.fixture
def mock_isfile():
    with patch("os.path.isfile", return_value=True):
        yield

# Fixture 用於模擬 subprocess.run
@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock_run:
        # 您可以在這裡設定 mock 的返回值或行為
        mock_run.return_value = MagicMock(returncode=0)  # 模擬成功執行外部命令
        yield mock_run  # 傳遞 mock_run 供測試使用, 


def test_validate_arguments(valid_args, mock_isfile):
    # Should not raise exception for valid arguments
    run.validate_arguments(valid_args)


@pytest.mark.parametrize("preset", ["invalid", "other"])
def test_invalid_preset(valid_args, preset, mock_isfile):
    valid_args.preset = preset
    with pytest.raises(ValueError):
        run.validate_arguments(valid_args)


def test_file_not_found(valid_args):
    valid_args.input_fasta = "missing.fasta"
    with pytest.raises(FileNotFoundError):
        run.validate_arguments(valid_args)


def test_setup_environment_missing_env(valid_args, monkeypatch, mock_isfile):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with pytest.raises(EnvironmentError):
        run.setup_environment(valid_args)

@pytest.mark.parametrize("mode,expected", [
    ("cpu", "-1"),
    ("1gpu0", "0"),
    ("1gpu1", "1"),
    ("2gpus", "0,1"),
])
def test_nvidia_cuda_visible_devices(valid_args, mode, expected, monkeypatch, mock_isfile, mock_subprocess_run):
    valid_args.platform = "nvidia"
    valid_args.mode = mode
    monkeypatch.setenv("CONDA_PREFIX", "/path/to/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "test_env")
    run.setup_environment(valid_args)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
    mock_subprocess_run.assert_called()

@pytest.mark.parametrize("mode,expected", [
    ("cpu", "-1"),
    ("1gpu0", "0"),
    ("1gpu1", "1"),
    ("2gpus", "0,1"),
])
def test_amd_hip_visible_devices(valid_args, mode, expected, monkeypatch, mock_isfile, mock_subprocess_run):
    valid_args.platform = "amd"
    valid_args.mode = mode
    monkeypatch.setenv("CONDA_PREFIX", "/path/to/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "test_env")
    run.setup_environment(valid_args)
    assert os.environ["HIP_VISIBLE_DEVICES"] == expected
    mock_subprocess_run.assert_called()

