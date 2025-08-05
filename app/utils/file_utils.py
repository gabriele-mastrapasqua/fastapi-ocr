import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import paddleocr
import pytesseract
import pdfplumber
import tempfile
import io
import base64
import os
import traceback
import time
import requests

import app.utils.ocr_utils as ocr_utils


# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Imposta solo il livello


def image_to_base64(attachment_bytes):
    """Convert bytes array to base64 used for images attachments in openai format"""
    return base64.b64encode(attachment_bytes).decode("utf-8")

def image_url_to_base64(attachment_url):
    """Fetch an image public downloadable URL and convert to base64 used for images attachments in openai format"""
    attachment_bytes = requests.get(attachment_url).content
    if not attachment_bytes:
        raise ValueError(f"Failed to fetch image from URL: {attachment_url}")
    
    # Convert bytes to base64
    logger.info(f"*** Image URL to base64: {attachment_url}")
    logger.debug(f"*** Image URL to base64: {attachment_bytes[:100]}...")
    return base64.b64encode(attachment_bytes).decode("utf-8")

def pdf_url_to_base64(attachment_url, max_pages = 1):
    """Fetch an image public downloadable URL and convert to base64 used for images attachments in openai format"""
    attachment_bytes = requests.get(attachment_url).content
    if not attachment_bytes:
        raise ValueError(f"Failed to fetch image from URL: {attachment_url}")
    
    # Convert bytes to base64
    logger.info(f"*** Image URL to base64: {attachment_url}")
    logger.debug(f"*** Image URL to base64: {attachment_bytes[:100]}...")
    images, num_pages = pdf_to_images(attachment_bytes, base_64=True, dpi_quality=300, resize_max_dim=1000, resize_mp=1.5, max_pages = max_pages)
    if not images:
        raise ValueError(f"Failed to convert PDF to images: {attachment_url}")
    
    print(f"*** PDF URL to base64: {attachment_url} - converted {len(images)} images", flush=True)
    return images


def pdf_to_images(contents, base_64 = False, dpi_quality = 300, resize_max_dim = 1000, resize_mp=1.5, max_pages = 1):
    import tempfile, os, io
    """
    Convert a PDF file bytes content to a list of images.
    
    @contents: PDF file bytes content.
    @base_64: If True, return images as base64 strings, otherwise as PIL
    @max_pages: extract only max pages, if 0 will extract all pages from pdf.

    Returns: list of images (PIL Image or base64 strings). + num pages in PDF
    """
    def _convert_image_list(image, images):
        
        # added: smart resize for good OCR performance and quality
        image = ocr_utils.smart_resize_for_ocr(image, max_dimension=resize_max_dim,  min_mp = resize_mp )

        # rotate the image if needed, some are -90/90 from scans
        image, needs_rotation = ocr_utils.rotate_image_if_needed(image)
        

        if base_64:
            # Convert PIL Image to base64 string
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images.append(img_str)
        else:
            # Append PIL Image directly
            images.append(image)


    try:
        tmp_file_path = None
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
            
        pages_processed = 0
        images = []
        with pdfplumber.open(tmp_file_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"🔄 *** pdf to images: Extracting {num_pages} pages, DPI: {dpi_quality}, resize max dim: {resize_max_dim}, min mp: {resize_mp}...", flush=True)

            for page in pdf.pages:
                # limit max number of pages
                pages_processed += 1
                if max_pages != 0 and pages_processed > max_pages:
                    print(f"*** limit max pages to convert from PDF to images {max_pages}", flush=True)
                    break

                # Convert page to image
                image = page.to_image(resolution=dpi_quality).original  # Era 300
                _convert_image_list(image, images)
                print(f"* pdfplumber images to return {len(images)}", flush=True)


            # 2 - fallback to another lib if this pdf fails to load
            if num_pages == 0:
                import fitz
                doc = fitz.open(tmp_file_path)
                
                num_pages = doc.page_count
                print(f"*** fitz pdf pages count {num_pages}", flush=True)
                
                for page_num in range(len(doc)):
                    # limit max number of pages
                    pages_processed += 1
                    if max_pages != 0 and pages_processed > max_pages:
                        print(f"*** limit max pages to convert from PDF to images {max_pages}", flush=True)
                        break


                    page = doc[page_num]
                  
                    # Convert page to image
                    zoom = 300 / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))

                    _convert_image_list(image, images)
                    print(f"* fitz pdf lib images to return {len(images)}", flush=True)
            
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)    
    
    return images, num_pages

