import to_txt
from docx import Document
import unicodedata
import os
import re
import shutil
import subprocess
from pathlib import Path



def _md_to_txt(src_dir: str, dest_dir: str):
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


def _normalize_unicode(text: str):
    return unicodedata.normalize("NFKC", text)


def _remove_urls_and_emails(text: str):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"ftp://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    text = re.sub(r"\S+\.(com|org|net|edu|gov|io|co|uk)\S*", "", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "", text)
    return text


def remove_noise(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = _remove_urls_and_emails(text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _filter_meaningful_words(text):
    words = text.split()

    for word in words:
        word_lower = word.lower()
        if re.match(r"^\d+$", word):  # Pure numbers
            continue
        if re.match(r"^\w*\d\w*$", word) and len(word) <= 4:  
            continue

        if re.search(r"(.)\1{2,}", word):
            continue

    return " ".join(
        [word for word in words if len(word) > 1]
    )


def _remove_repeated_phrases(text: str):
    sentences = re.split(r"[.!?]+", text)
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        if (
            clean_sentence
            and len(clean_sentence) > 10 
            and clean_sentence not in seen_sentences
        ):
            seen_sentences.add(clean_sentence)
            unique_sentences.append(sentence.strip())
    result = ". ".join(unique_sentences).strip()
    if result and not result[-1] in ".!?":
        result += "."
    return result


def _get_word_count(text: str):
    words = text.split()
    return len(words)


def _clean_text(text: str):
    text = _normalize_unicode(text)
    text = _filter_meaningful_words(text)
    text = _remove_repeated_phrases(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_file(file_path: Path):
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="UTF-32") as f:
                text = f.read()
    except UnicodeDecodeError:
        raise ValueError(f"Cannot decode {file_path.name} as UTF-8")

    if len(text.strip()) < 50:
        print(f"Skipping {file_path.name}: original content too short")
        return

def _token_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                yield word
            yield '\n'


def _split_txt_by_tokens_generator(token_limit=5000, input_dir="dataset"):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory '{input_dir}' not found!")
        return
    os.chdir(input_path)
    for file in Path('.').glob('*.txt'):
        print(f"Processing {file}...")
        base_name = file.stem
        token_count = 0
        part_num = 1
        output_path = Path(f"{base_name}_{part_num:03d}.txt")
        out_f = open(output_path, 'w', encoding='utf-8')
        for token in _token_generator(file):
            if token == '\n':
                out_f.write('\n')
            else:
                out_f.write(token + ' ')
                token_count += 1

                if token_count >= token_limit:
                    out_f.write('\n')
                    out_f.close()
                    part_num += 1
                    token_count = 0
                    output_path = Path(f"{base_name}_{part_num:03d}.txt")
                    out_f = open(output_path, 'w', encoding='utf-8')
        out_f.close()
        print(f"✅ Split completed for {file} -> {part_num} parts")


def _document_to_text(docx_path, txt_output_path):
    doc = Document(docx_path)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for para in doc.paragraphs:
            f.write(para.text + "\n")

def _pdf_to_text(pdf_path, txt_output_path):
    to_txt.process_pdf(pdf_path)
    ocr_txt_path = Path(pdf_path).with_suffix('.txt')
    if ocr_txt_path.exists():
        ocr_txt_path.rename(txt_output_path)
    else:
        raise FileNotFoundError(f"OCR text file not found for {pdf_path}")


def _datanowmalization(directory):
    output_dir = Path(directory)
    if not output_dir.exists():
        print(f"Directory '{output_dir}' not found.")
        return
    files_processed = 0
    print("Starting file cleaning and filtering...")
    for file_path in output_dir.iterdir():
        if file_path.is_file() and file_path.suffix in {".txt", ".json", ".md", ".csv", ".docx", ".odt", ".doc", ".html"}:
            try:
                if file_path.suffix in {".docx", ".odt", ".doc"}:
                    txt_output_path = file_path.with_suffix(".txt")
                    document_to_text(str(file_path), str(txt_output_path))
                    clean_file(txt_output_path)
                else:
                    clean_file(file_path)
                files_processed += 1
            except Exception as e:
                print(f"Failed on {file_path.name}: {e}")
    print("\nProcessing complete:")
    print(f"- {files_processed} files cleaned")


def datarationization(directory):
    output_dir = Path(directory)
    if not output_dir.exists():
        print(f"Directory '{output_dir}' not found.")
        return
    _datanowmalization("dataset")
    _split_txt_by_tokens_generator(token_limit=5000, input_dir="dataset")