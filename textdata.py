import os
import re
import unicodedata
from pathlib import Path
from docx import Document
import to_txt


def normalize_unicode(text: str):
    return unicodedata.normalize("NFKC", text)


def remove_urls_and_emails(text: str):
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
    text = remove_urls_and_emails(text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def filter_meaningful_words(text: str, min_word_length: int = 2):
    words = text.split()
    noise_words = {
        "http",
        "https",
        "www",
        "com",
        "org",
        "contact",
        "follow",
    }

    for word in words:
        word_lower = word.lower()
        # Remove noise words
        if word_lower in noise_words:
            continue

        if re.match(r"^\d+$", word):  # Pure numbers
            continue
        if re.match(r"^\w*\d\w*$", word) and len(word) <= 4:  # Mixed short alphanumeric
            continue

        if re.search(r"(.)\1{2,}", word):
            continue

    return " ".join(
        [word for word in words if len(word) >= min_word_length]
    )


def remove_repeated_phrases(text: str):
    sentences = re.split(r"[.!?]+", text)
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        if (
            clean_sentence
            and len(clean_sentence) > 10  # Only check substantial sentences
            and clean_sentence not in seen_sentences
        ):
            seen_sentences.add(clean_sentence)
            unique_sentences.append(sentence.strip())
    result = ". ".join(unique_sentences).strip()
    if result and not result[-1] in ".!?":
        result += "."
    return result


def get_word_count(text: str):
    words = text.split()
    return len(words)


def clean_text(text: str):
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = filter_meaningful_words(text)
    text = remove_repeated_phrases(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_file(file_path: Path):
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

    original_text = text
    cleaned_text = clean_text(text)

    if len(cleaned_text.strip()) > 20:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        final_word_count = get_word_count(cleaned_text)
        final_size_kb = file_path.stat().st_size / 1024
        original_size = len(original_text.encode("utf-8")) / 1024
        reduction = (
            ((original_size - final_size_kb) / original_size * 100)
            if original_size > 0
            else 0
        )
        if final_word_count < 10:
            print(
                f"Warning: {file_path.name} became very small after cleaning (words: {final_word_count}, size: {final_size_kb:.2f}KB)"
            )
        else:
            print(
                f"Cleaned: {file_path.name} (words: {final_word_count}, size: {final_size_kb:.2f}KB, reduced by {reduction:.1f}%)"
            )
    else:
        print(
            f"Warning: {file_path.name} has no meaningful content after cleaning")


def token_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                yield word
            yield '\n'


def split_txt_by_tokens_generator(token_limit=5000, input_dir='llmdataset'):
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
        for token in token_generator(file):
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


def document_to_text(docx_path: str, txt_output_path: str):
    doc = Document(docx_path)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for para in doc.paragraphs:
            f.write(para.text + "\n")

def pdf_to_text(pdf_path: str, txt_output_path: str):
    to_txt.process_pdf(pdf_path)
    ocr_txt_path = Path(pdf_path).with_suffix('.txt')
    if ocr_txt_path.exists():
        ocr_txt_path.rename(txt_output_path)
    else:
        raise FileNotFoundError(f"OCR text file not found for {pdf_path}")

def datanowmalization(directory: str = "."):
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


def datarationization(directory: str = "."):
    output_dir = Path(directory)
    if not output_dir.exists():
        print(f"Directory '{output_dir}' not found.")
        return
    datanowmalization("llmdataset")
    split_txt_by_tokens_generator(token_limit=5000, input_dir="llmdataset")
