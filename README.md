# Fastapi + ocr
PaddleOCR and tesseract as a FastAPI service

## Features

* **Switchable OCR Engines**: Choose between **Tesseract** (faster, lower quality) or **PaddleOCR** (higher quality, slower on CPU).

* **OCR Bypass for Faster Extraction**: Automatically detects unusual fonts (e.g., CID fonts) that prevent direct text extraction from PDFs. If the text can be extracted directly, the OCR step is skipped entirely — resulting in up to **100x faster** processing.

* **Auto-Rotate Documents**: Detect and correct page rotation using Tesseract’s angle detection (adds \~0.5s–1.3s per page). Scanned documents are often rotated ±90°. By rotating pages before OCR, recognition accuracy improves significantly. PaddleOCR can handle rotated pages, but results are better if rotation is corrected in advance.

* **High-Quality PDF to Image Conversion**: Converts PDFs to **PNG** or **JPG** with the best possible quality.

* **Optional Image Scaling**: Resize input images to improve OCR speed and performance.

* **Basic Image Preprocessing**: Enhance Tesseract recognition through grayscale conversion, color correction, and sharpening filters.

* **LLM-Friendly Output**: Returns clean, structured text optimized for use with large language models — ideal for adding RAG (Retrieval-Augmented Generation) context to prompts.

* **PDF to Base64 Image**: Converts PDF pages to base64-encoded images, useful for passing as file attachments in OpenAI's chat completion format.

* **Custom Page Limits**: Process only a selected range of pages (e.g., from page 1 to N), which is useful for large PDFs where relevant content is at the beginning.

## TODOs
- [ ] docker version of PaddleOCR for gpu
- [ ] improve API params for custom image pre-processing before conversion
- [ ] handle other kind of file formats, like pptx, docx, excel, ...

## swagger api
Swagger docs Are published under http://localhost:9292/docs

## development

run with 
```sh
make up
```

stop with 
```sh
make down
```

force rebuild with 
```sh
make build
```

## usage in other projects
Add this service in your docker compose file like this:

```yml
version: '3.8'

services:
  # FastAPI ocr service
  fastapi-ocr-api:
    platform: linux/amd64
    image: gabrielem0/fastapi-ocr:latest
    container_name: fastapi-ocr-api
    ports:
      - "9292:9292"
    volumes:
      - paddle_models_cpu:/root/.paddle
      - paddleocr_models_cpu:/root/.paddleocr
    environment:
      - DEVICE=cpu
    restart: unless-stopped

volumes:
  paddle_models_cpu:
  paddleocr_models_cpu:
```

## Calling this service for OCR
You can call this service using curl or for ex. py:

```py
def call_ocr_service(file_path: str, engine: str = "auto", force_angle_rotation = 0):
    """Call the OCR service to extract text from a PDF file

    Args:engine (str): The OCR engine to use, default is "auto", or "tesseract" or "paddleocr"
    force_angle_rotation (int): Force rotation angle for the OCR, default is 0 (no rotation)
    Returns:
        dict: The OCR response containing extracted text and metadata
    """
    import requests

    url = 'http://localhost:9292/ocr?force_angle_rotation=0'

    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        files = {
            'file': (file_path, f, 'application/pdf'),
        }
        data = {
            'engine': engine
        }
        headers = {
            'accept': 'application/json'
        }

        response = requests.post(url, headers=headers, files=files, data=data)

    # Output response
    res = response.json()
    return res
```

You can choose the OCR engine to use, default is "auto", or "tesseract" or "paddleocr".

With auto this service will use tesseract for bigger documents with more pages, otherwise paddleocr for more quality but slower speed.

