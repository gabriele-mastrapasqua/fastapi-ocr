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
    image: gabrielem0/fastapi-ocr:latest
    container_name: fastapi-ocr-api
    ports:
      - "9292:9292"
    volumes:
      - .:/app
      - paddle_models_cpu:/root/.paddle
      - paddleocr_models_cpu:/root/.paddleocr
    environment:
      - DEVICE=cpu
    restart: unless-stopped


volumes:
  paddle_models_cpu:
  paddleocr_models_cpu:
```