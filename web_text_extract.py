import os
import re
from bs4 import BeautifulSoup
from pathlib import Path


def _clean_text(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[#*>`~_\-\[\]\{\}\(\)]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_from_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _extract_from_html(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    except Exception:
        return ""


def extract_texts(root_dir, output_dir="out"):
    # Convert root_dir to Path object and resolve to absolute path
    root_path = Path(root_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Walking directory tree: {root_path}\n")

    # Supported text-based file extensions
    text_extensions = [
        "md",
        "mdx",
        "html",
        "js",
        "jsx",
        "ts",
        "tsx",
        "txt",
        "css",
        "scss",
        "sass",
        "less",
        "json",
        "yaml",
        "yml",
        "xml",
        "svg",
        "sh",
        "bash",
    ]

    for root, dirs, files in os.walk(root_path, topdown=True):
        current_root = Path(root)
        try:
            rel_path = current_root.relative_to(root_path)
        except ValueError:
            rel_path = Path(".")

        # Create corresponding subdirectory in output
        output_subdir = output_path / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Print directory with proper indentation
        level = len(rel_path.parts) if rel_path != Path(".") else 0
        indent = " " * 4 * level
        print(f"{indent}📁 {current_root.name}/")
        subindent = " " * 4 * (level + 1)

        for file in sorted(files):  # Sort files for consistent output
            file_path = current_root / file
            ext = file_path.suffix.lstrip(".").lower()
            text = ""

            if ext in text_extensions:
                print(f"{subindent}📄 {file}")
                try:
                    if ext in ["html", "xml", "svg"]:
                        text = _extract_from_html(file_path)
                    else:
                        text = _extract_from_text(file_path)

                    if text:
                        text = _clean_text(text)

                        # Create output file path
                        output_file = output_subdir / f"{file_path.stem}.txt"

                        # Write to output file
                        try:
                            with open(output_file, "w", encoding="utf-8") as f:
                                f.write(text)
                        except Exception as e:
                            print(f"{subindent}❌ Error writing {output_file}: {e}")
                except Exception as e:
                    print(f"{subindent}❌ Error processing {file}: {e}")

    print(f"\n✅ All extracted and cleaned texts saved to: {output_path}")
    return output_path

