from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import paddleocr
import cv2
import numpy as np
from PIL import Image
import io
import os
import logging
import app.utils.ocr_utils as utils
import traceback
from enum import Enum
import pdfplumber
import tempfile

import app.services.paddleocr as paddleocr

# Setup logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Imposta solo il livello

app = FastAPI(title="PaddleOCR API", version="1.0.0")

paddleOcrEngine = paddleocr.PaddleOCREngine(lang="it")


@app.get("/")
async def root():
    return {"message": "PaddleOCR API Server", "device": os.getenv('DEVICE', 'cpu')}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": os.getenv('DEVICE', 'cpu')}


class Engine(str, Enum):
    auto = "auto"
    paddleocr = "paddleocr"
    tesseract = "tesseract"

@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(..., description="File to perform OCR on. Can be PDF or Image."),
    engine: Engine = Form(Engine.auto, description="OCR Engine to use: auto for auto select between tesseract for large files, or paddleocr."),  # default "auto"
    force_angle_rotation: int = 0,  # angolo di rotazione forzato (0 per nessuna rotazione, -90, 90, 180, ...)
    
):
    """
    Perform OCR on uploaded file
    """

    content_type = file.content_type
    file_type = None
    if content_type == "application/pdf":
        file_type = "PDF"
    elif content_type.startswith("image/"):
        file_type = "Image"
    else:
        file_type = "Unsupported"
    logger.info(f"Received file type: {file_type}, content type: {content_type}")


    if engine == Engine.auto:
        # TODO - check page count, if > 2 use tesseract, else use paddleOCR
        ocr = paddleOcrEngine
    elif engine == Engine.paddleocr:
        ocr = paddleOcrEngine
    elif engine == Engine.tesseract:
        # TODO - implement tesseract OCR
        ocr = paddleOcrEngine

    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not available")
    
    try:
        # read bytes from the file upload
        contents = await file.read()

        if file_type == "PDF":
            images, num_pages_in_pdf =utils.pdf_to_images(contents, base_64=False)
            logger.info(f"PDF converted to {num_pages_in_pdf} images")
            if not images:
                raise HTTPException(status_code=400, detail="No pages found in PDF")
            results = []
            for image in images:
                logger.info(f"Performing OCR on image {image} with force_angle_rotation={force_angle_rotation}")
                response = ocr.execute_ocr(image, force_angle_rotation=force_angle_rotation)
                results.append(response)
            return JSONResponse({"results": results, "num_pages": num_pages_in_pdf})
            
        elif file_type == "Image":
            # Read image
            image = Image.open(io.BytesIO(contents))
            response = ocr.execute_ocr(image, force_angle_rotation=force_angle_rotation)
            return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9292)