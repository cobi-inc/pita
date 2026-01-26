import subprocess
import platform
import shutil
import re
from typing import Optional, Union

def get_total_vram() -> Union[int, str]:
    """
    Get the total VRAM (in MiB) of the primary GPU on the system.

    This function attempts to detect VRAM across different platforms and GPU types:
    - NVIDIA GPUs: Uses nvidia-smi (Windows & Linux)
    - AMD GPUs: Uses ROCm-smi (Linux)
    - Windows Generic: Uses PowerShell WMI queries

    Returns:
        Total VRAM in MiB (int) if successfully detected, or an error message (str)
        if detection fails or drivers are not installed.
    """
    os_name = platform.system()

    # --- NVIDIA (Windows & Linux) ---
    if shutil.which("nvidia-smi"):
        try:
            # Change 'memory.used' to 'memory.total'
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            return int(result.strip())
        except Exception as e:
            return f"Error reading Nvidia SMI: {e}"

    # --- WINDOWS (AMD, Intel, & Generic) ---
    if os_name == "Windows":
        try:
            # We use WMI (Win32_VideoController) to get the 'AdapterRAM'
            # We sort by size to find the largest GPU (ignoring small iGPUs if a dGPU exists)
            cmd = 'Get-CimInstance Win32_VideoController | Sort-Object -Property AdapterRAM -Descending | Select-Object -First 1 -ExpandProperty AdapterRAM'

            result = subprocess.check_output(
                ["powershell", "-Command", cmd],
                encoding="utf-8"
            )

            # Windows returns Bytes. Convert to MiB.
            vram_bytes = float(result.strip())
            vram_mib = int(vram_bytes / 1024 / 1024)
            return vram_mib
        except Exception:
            pass

    # --- LINUX (AMD ROCm) ---
    if shutil.which("rocm-smi"):
        try:
            # Use --showmeminfo vram and parse for 'Total'
            result = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram"], encoding="utf-8"
            )
            # Use regex to find "VRAM Total Memory (B): <numbers>"
            match = re.search(r"VRAM Total Memory \(B\):\s+(\d+)", result)
            if match:
                vram_bytes = int(match.group(1))
                vram_mib = vram_bytes // (1024 * 1024)
                return vram_mib
        except Exception as e:
            return f"Error reading ROCm SMI: {e}"

    return "Could not detect Total VRAM. Ensure drivers are installed."

def get_gpu_vram_usage_mb() -> Optional[int]:
    """
    Get the current VRAM usage (in MiB) across all NVIDIA GPUs.

    This function uses nvidia-smi to query current GPU memory usage and returns
    the sum across all GPUs if multiple are present.

    Returns:
        Total current VRAM usage in MiB across all GPUs, or None if nvidia-smi
        is not available or an error occurs.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Returns usage for all GPUs, sum them or take the first one
        vram_values = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        return sum(vram_values)  # Total across all GPUs, or use vram_values[0] for first GPU
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get VRAM usage from nvidia-smi: {e}")
        return None

def detect_model_type(model: str) -> Optional[str]:
    """
    Detect the model file type from a Hugging Face model repository.

    This function attempts to determine whether a Hugging Face model repository
    contains GGUF or safetensors files by querying the repository file list.

    Args:
        model: Model identifier, typically in the format "owner/repo" for Hugging Face repos.

    Returns:
        A string indicating the detected model type ("gguf" or "safetensors"), or None
        if the type could not be determined or if the huggingface_hub library is not available.
    """
    # Determine the model type (GGUF, safetensors, etc.) for Hugging Face repos
    detected_dtype = None

    # Try to detect model file type using the Hugging Face hub when the model looks
    # like a repo id (owner/repo) and the huggingface_hub package is available.
    if '/' in str(model) and not model.startswith("model/"):
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            try:
                files = api.list_repo_files(repo_id=model)
            except Exception:
                files = []

            # Check for common filetypes in the repo
            files_lower = [f.lower() for f in files]
            if any(f.endswith('.gguf') for f in files_lower):
                detected_dtype = 'gguf'
            elif any(f.endswith('.safetensors') for f in files_lower):
                detected_dtype = 'safetensors'
            else:
                print("Warning: Could not determine model type.")

        except Exception:
            # huggingface_hub not installed or network error — leave dtype as passed
            print("Warning: Could not import huggingface_hub or access model repo to detect model type.")

    return detected_dtype
