# paddleocr-fastapi
PaddleOCR and tesseract as a FastAPI service

## features
- auto rotation of rotated documents
- pdf to image to ocr
- pdf to image to base64 list of images, useful for llm attachments


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

