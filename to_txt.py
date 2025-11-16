import sys
import os
import subprocess
import shutil
from pathlib import Path


def clone_repo(repo_url: str, clone_dir: str = "repo_clone"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    subprocess.run(["git", "clone", "--depth", "1",
                   repo_url, clone_dir], check=True)
    return clone_dir


def convert_md_to_txt(src_dir: str = "repo_clone", dest_dir: str = "md_as_txt"):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    for md_file in src_path.rglob("*.md"):
        txt_file = dest_path / (md_file.stem + ".txt")
        counter = 1
        while txt_file.exists():
            txt_file = dest_path / f"{md_file.stem}_{counter}.txt"
            counter += 1

        with (
            open(md_file, "r", encoding="utf-8") as f_in,
            open(txt_file, "w", encoding="utf-8") as f_out,
        ):
            f_out.write(f_in.read())

        print(f"Saved: {txt_file}")


def _check_ocrmypdf():
    if shutil.which("ocrmypdf") is None:
        print("ERROR: 'ocrmypdf' is not installed.")
        "Please install it to use this script.\n"
        "\tOn macOS: brew install ocrmypdf\n"
        "\tOn Ubuntu/Debian: sudo apt install ocrmypdf\n"
        "\tUsing Python's pip: pip install ocrmypdf\n"
        sys.exit(1)


def process_pdf(input_file):
    input_path = Path(input_file)
    output_file = input_path.with_suffix('.txt')
    _check_ocrmypdf()
    # Check if input file exists
    if not input_path.is_file():
        print(f"⚠️  WARNING: File not found, skipping: {input_file}")
        return False

    print("---")
    print(f"Processing '{input_file}' for English text...")
    print(f"Output will be saved as '{output_file}'")
    try:
        result = subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",
                "-l", "ben+eng",
                "--sidecar", str(output_file),
                str(input_path),
                os.devnull
            ],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            if output_file.is_file():
                print(f"✅ Success! Text saved to '{output_file}'")
                return True
            else:
                print(
                    f"❌ ERROR: OCR process ran, but the text file was not created for '{input_file}'.")
                return False
        else:
            print(f"❌ ERROR: OCR process failed for '{input_file}'.")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return False

    except Exception as e:
        print(
            f"❌ ERROR: Exception occurred while processing '{input_file}': {e}")
        return False

