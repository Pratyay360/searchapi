import requests
import os
from urllib.parse import unquote
from pathlib import Path
import web_fetch

def download_doc(url, download_dir):
    try:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        response = requests.get(
            url, 
            timeout=15,
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()
        
        # Handle content-disposition filename
        filename = None
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"\'')
        
        # Fallback to URL filename
        if not filename:
            filename = os.path.basename(unquote(url.split('?')[0]))
            if not filename or '.' not in filename:
                filename = 'downloaded_file'
        
        filepath = os.path.join(download_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Downloaded: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed: {url}")
        return False


def download_pdf(url, download_dir):
    try:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        response = requests.get(
            url, 
            timeout=15,
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()
        
        # Handle content-disposition filename
        filename = None
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"\'')
        
        # Fallback to URL filename
        if not filename:
            filename = os.path.basename(unquote(url.split('?')[0]))
            if not filename or '.' not in filename:
                filename = 'downloaded_file'
        
        filepath = os.path.join(download_dir, filename)
        
        # Download file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Downloaded: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed: {url}")
        return False

async def fetch_and_download_pdfs(param: str, download_dir: str):
    pdf_urls = await web_fetch.search_pdf(param)
    for url in pdf_urls:
        download_pdf(url, download_dir) 
    print(f"Total PDFs downloaded: {len(pdf_urls)}")