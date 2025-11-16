import os
import re
import shutil
import subprocess
from pathlib import Path



def convert_md_to_txt(src_dir: str, dest_dir: str = "md_as_txt"):
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


