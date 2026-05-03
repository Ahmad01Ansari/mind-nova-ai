"""
MP4 to MP3 Batch Converter
--------------------------
Converts all .mp4 files in a specified folder to .mp3 audio files.
Requires: ffmpeg installed on the system
Install dependency: pip install tqdm
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg is not installed or not in PATH.")
        print("Install it via:")
        print("  Windows : https://ffmpeg.org/download.html")
        print("  macOS   : brew install ffmpeg")
        print("  Linux   : sudo apt install ffmpeg")
        sys.exit(1)


def convert_mp4_to_mp3(input_path: Path, output_path: Path, bitrate: str = "192k") -> bool:
    """
    Convert a single MP4 file to MP3.

    Args:
        input_path  : Path to the source .mp4 file
        output_path : Path for the output .mp3 file
        bitrate     : Audio bitrate (default: 192k)

    Returns:
        True on success, False on failure
    """
    cmd = [
        "ffmpeg",
        "-i", str(input_path),   # input file
        "-vn",                    # disable video stream
        "-acodec", "libmp3lame", # MP3 encoder
        "-ab", bitrate,          # audio bitrate
        "-ar", "44100",          # sample rate
        "-y",                    # overwrite output without asking
        str(output_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode(errors="replace").strip().splitlines()
        print(f"  [FAILED] {input_path.name}")
        # Print last 3 lines of ffmpeg error for quick diagnosis
        for line in error_msg[-3:]:
            print(f"    {line}")
        return False

    return True


def batch_convert(folder: str, bitrate: str = "192k", output_subfolder: str = None):
    """
    Scan a folder for .mp4 files and convert each to .mp3.

    Args:
        folder           : Path to the folder containing MP4 files
        bitrate          : Audio bitrate for output MP3s
        output_subfolder : If set, save MP3s in this subfolder; otherwise alongside source
    """
    source_dir = Path(folder)

    if not source_dir.exists():
        print(f"ERROR: Folder not found: {source_dir}")
        sys.exit(1)

    if not source_dir.is_dir():
        print(f"ERROR: Not a directory: {source_dir}")
        sys.exit(1)

    # Collect all MP4 files (case-insensitive)
    mp4_files = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    ])

    if not mp4_files:
        print(f"No .mp4 files found in: {source_dir}")
        return

    # Determine output directory
    if output_subfolder:
        out_dir = source_dir / output_subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = source_dir

    print(f"\n{'='*55}")
    print(f"  MP4 → MP3 Batch Converter")
    print(f"{'='*55}")
    print(f"  Source folder : {source_dir}")
    print(f"  Output folder : {out_dir}")
    print(f"  Bitrate       : {bitrate}")
    print(f"  Files found   : {len(mp4_files)}")
    print(f"{'='*55}\n")

    success_list = []
    failed_list  = []

    iterator = tqdm(mp4_files, unit="file") if HAS_TQDM else mp4_files

    for mp4_file in iterator:
        mp3_file = out_dir / (mp4_file.stem + ".mp3")

        if not HAS_TQDM:
            print(f"Converting: {mp4_file.name} ...", end=" ", flush=True)

        ok = convert_mp4_to_mp3(mp4_file, mp3_file, bitrate)

        if ok:
            success_list.append(mp4_file.name)
            if not HAS_TQDM:
                print("OK")
        else:
            failed_list.append(mp4_file.name)
            if not HAS_TQDM:
                print("FAILED")

    # Summary
    print(f"\n{'='*55}")
    print(f"  Done! {len(success_list)} converted, {len(failed_list)} failed.")
    if failed_list:
        print(f"\n  Failed files:")
        for name in failed_list:
            print(f"    - {name}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert all MP4 files in a folder to MP3 audio."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="/home/ahmad10raza/Desktop/piano",
        help="Path to the folder containing MP4 files (default: current directory)"
    )
    parser.add_argument(
        "--bitrate",
        default="128k",
        help="MP3 audio bitrate, e.g. 128k, 192k, 320k (default: 192k)"
    )
    parser.add_argument(
        "--output-subfolder",
        default="/home/ahmad10raza/Desktop/piano",
        metavar="SUBFOLDER",
        help="Save MP3s in this subfolder inside the source folder (optional)"
    )

    args = parser.parse_args()

    check_ffmpeg()
    batch_convert(args.folder, args.bitrate, args.output_subfolder)