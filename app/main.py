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

import app.services.paddleocr as paddleocr
import app.services.tesseract as tesseract

# Setup logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Imposta solo il livello

app = FastAPI(title="FastAPI OCR API", version="1.0.0")

paddleOcrEngine = paddleocr.PaddleOCREngine(lang="it")

tesseractOCREngine = tesseract.TesseractOCREngine(lang="ita", ocr_config="--psm 11")


@app.get("/")
async def root():
    return {"message": "FastAPI OCR API Server", "device": os.getenv('DEVICE', 'cpu')}

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
    force_angle_rotation: int = Form(0, description="If set to 0 will not be activated, if set to an angle will force page rotations. Useful to force rotation for some pdf scans. The system will try to automatically rotate pages if needed when this is set to 0 (default)"), 
    use_tesseract_after_min_pages: int = Form(2, description="Let's use tesseract after at least (def 2 pages) as it's more faster."),
    bypass_ocr_use_pdf_text_if_available: bool = Form(True, description="use pdf text if it's safe to use (no strange cid fonts, etc) so bypass OCR completely. Useful for pc exports pdf for faster responses. Set to False will always use ocr."), 
    
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
        ocr = tesseractOCREngine

    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not available")
    
    try:
        # read bytes from the file upload
        contents = await file.read()
        results = {}

        if file_type == "PDF":
            logger.info(f"Lets try first to extract txt from pdf if is safe to do so... bypass_ocr_use_pdf_text_if_available params is set to {bypass_ocr_use_pdf_text_if_available}")
            texts, num_pages_in_pdf = utils.pdf_to_text(contents)
            logger.info(f"PDF converted to {num_pages_in_pdf} images")

            if bypass_ocr_use_pdf_text_if_available and len(texts) > 0:
                extracted_text = "\n".join(texts)
                logger.info(f"bypass_ocr_use_pdf_text_if_available is {bypass_ocr_use_pdf_text_if_available} and PDF has some text extrable without OCR and safe to use (without CID fonts): first 100 chars: {repr(extracted_text[:100])} ")
                # so if pdf can export safely txt, let's bypass OCR and return readable txt from pdf
                ocr = tesseractOCREngine
                results = {
                    "success": True,
                    "results": extracted_text,
                    "total_text": extracted_text,
                    "device": os.getenv('DEVICE', 'cpu')
                }
            else: 
                logger.info(f"Let's do OCR")
                images, num_pages_in_pdf =utils.pdf_to_images(contents, base_64=False)
                logger.info(f"PDF converted to {num_pages_in_pdf} images")

                if not images:
                    raise HTTPException(status_code=400, detail="No pages found in PDF")
                
                # def use_tesseract_after_min_pages = 2 min pages to run using tesseract instead of paddle (more faster, less quality)
                if engine == Engine.tesseract or (engine == Engine.auto and num_pages_in_pdf > use_tesseract_after_min_pages):
                    # check page count, if > 2 use tesseract, else use paddleOCR
                    ocr = tesseractOCREngine
                    logger.info(f"Using Tesseract OCR for {num_pages_in_pdf} pages")
                    
                    # tesseract can run multiple pages in parallel to speed up the process.
                    response = ocr.execute_ocr(images, force_angle_rotation=force_angle_rotation)
                    #response = ocr.execute_ocr_performance_test(images, force_angle_rotation=force_angle_rotation)

                    results = response
                else:
                    # Use paddleOCR for small PDFs for high quality results
                    # paddle is much slower
                    logger.info(f"Using PaddleOCR for {num_pages_in_pdf} pages")
                    
                    for image in images:
                        logger.info(f"Performing OCR on image {image} with force_angle_rotation={force_angle_rotation}")
                        response = ocr.execute_ocr(image, force_angle_rotation=force_angle_rotation)
                        results = response

            # finally return a response    
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

@app.post("/pdf-to-images")
async def pdf_to_images(
    file: UploadFile = File(..., description="File to perform OCR on. Can be PDF or Image."),
    force_angle_rotation: int = 0,  # angolo di rotazione forzato (0 per nessuna rotazione, -90, 90, 180, ...)    
    to_base64: bool = True,  # se base64, ritorna una lista di immagini in base64
    dpi_quality: int = 300,  # 300 super good quality, 200 faster but base quality
    resize_max_dim: int = 1000, # max dimension for resize, if image is bigger than this it will be resized
    resize_mp:int  = 1.5,  # megapixel for resize, if image is bigger than this it will be resized
):
    """
    Perform PDF to images conversion
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



    if file_type != "PDF":
        raise HTTPException(status_code=503, detail=f"Cannot convert {file_type} format to images! only pdf is supported!")
    
    try:
        # read bytes from the file upload
        contents = await file.read()

        if file_type == "PDF":
            images, num_pages_in_pdf = utils.pdf_to_images(contents, base_64=to_base64, dpi_quality=dpi_quality)
            return JSONResponse({"images": images, "to_base64": to_base64, "num_pages": num_pages_in_pdf})
    except Exception as e:
        logger.error(f"PDF Conversion processing failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF conversion processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9292)