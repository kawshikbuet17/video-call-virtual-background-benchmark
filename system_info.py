"""Print this machine's hardware details for the results table in README.md.

Uses only the Python standard library, so it runs without any virtual environment:

    python system_info.py

Works on Windows, Linux and macOS. Anything it cannot detect is shown as "unknown".
"""
import os
import platform
import shutil
import subprocess
import sys


def run(cmd):
    """Run a command and return its output, or "" if it fails or is missing."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def cpu_name():
    system = platform.system()
    if system == "Windows":
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    return platform.processor() or "unknown"


def ram_gb():
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1024**3
        if system == "Linux":
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])  # first line is "MemTotal: N kB"
            return kb / 1024**2
        if system == "Darwin":
            return int(run(["sysctl", "-n", "hw.memsize"])) / 1024**3
    except (OSError, ValueError):
        pass
    return None


def gpu_names():
    # NVIDIA first: nvidia-smi also gives the memory size.
    if shutil.which("nvidia-smi"):
        out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if out:
            return [line.strip() for line in out.splitlines()]
    system = platform.system()
    if system == "Windows":
        out = run(["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_VideoController).Name"])
    elif system == "Linux":
        out = "\n".join(l for l in run(["lspci"]).splitlines() if "VGA" in l or "3D" in l)
    elif system == "Darwin":
        out = "\n".join(l.split(":", 1)[1].strip()
                        for l in run(["system_profiler", "SPDisplaysDataType"]).splitlines()
                        if "Chipset Model" in l)
    else:
        out = ""
    return [line.strip() for line in out.splitlines() if line.strip()]


def main():
    ram = ram_gb()
    ram_text = f"{ram:.0f} GB" if ram else "unknown"
    gpus = gpu_names()
    docker = run(["docker", "--version"]) if shutil.which("docker") else ""

    print("System information")
    print("------------------")
    print(f"OS:        {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"CPU:       {cpu_name()} ({os.cpu_count()} logical cores)")
    print(f"RAM:       {ram_text}")
    print(f"GPU(s):    {'; '.join(gpus) if gpus else 'none detected'}")
    print(f"Docker:    {docker or 'not found'}")
    print()
    print("Copy these into the Results table in README.md:")
    print(f"  Device: CPU = {cpu_name()} | GPU = {gpus[0] if gpus else 'none'}")
    print(f"  RAM:    {ram_text}")


if __name__ == "__main__":
    main()
