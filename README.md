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