# paddleocr-fastapi
PaddleOCR and tesseract as a FastAPI service

## features
- Able to switch to tesseract (faster, less quality) or PaddleOCR (better quality, on cpu is slower);
- Bypass OCR for extraction speed when possible: auto detect if there are strange fonts (Cid) from PDFs that prevent us to extract text directly from the PDF. If possible, skip the OCR process entirely and extract text from the pdf directly. This speedup the process of text extraction of 100x;
- Detect and auto rotate documents if necessary: It will use tesseract angle detection for PDFs (usually this will add +0.5s - 1.3s time per page to check page rotation). Usually documents scans are -90 / +90  more or less. With this approximation we can easly send a redacted image of the document's page so the recognition performance will improve. PaddleOCR will automatically try to work even on rotated pages, but the quality of the OCR will improve fixing the rotation before sending the page to the OCR process;
- Pdf to image conversion in PNG / JPG using best quality;
- Optionally scale and resize the converted input image to improve OCR speed performace;
- Basic image pre-processing to improve tesseract recognition: gray scale, color corrections, sharpness filters;
- LLM friendly, will return clean text useful to be processed to an LLM, for example for adding some RAG context to your prompt;
- Pdf to image in base64: useful to be used for llm attachments (openai sdk chat completition format);
- custom limit how many pdf pages to process from page 1-n, useful for large PDFs where the interestin content is at the start of the file.

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

