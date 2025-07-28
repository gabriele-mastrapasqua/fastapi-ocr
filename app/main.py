from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import paddleocr
import cv2
import numpy as np
from PIL import Image
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR API", version="1.0.0")

# Initialize PaddleOCR
try:
    device = os.getenv('DEVICE', 'cpu')
    use_gpu = device.lower() == 'gpu'
    
    logger.info(f"Initializing PaddleOCR with device: {device}")
    ocr = paddleocr.PaddleOCR(
        use_angle_cls=True,
        lang='it',  # Puoi cambiare la lingua qui
        use_gpu=use_gpu,
        show_log=True,
    )
    logger.info("PaddleOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {e}")
    ocr = None

@app.get("/")
async def root():
    return {"message": "PaddleOCR API Server", "device": os.getenv('DEVICE', 'cpu')}

@app.get("/health")
async def health_check():
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not available")
    return {"status": "healthy", "device": os.getenv('DEVICE', 'cpu')}

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on uploaded image
    """
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not available")
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Perform OCR
        logger.info("Starting OCR processing")
        result = ocr.ocr(img_array, cls=True)
        
        # Process results
        extracted_text = []
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                continue
            for line in res:
                text = line[1][0]
                confidence = line[1][1]
                #bbox = line[0]
                extracted_text.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    #"bbox": bbox
                })
        
        logger.info(f"OCR completed, found {len(extracted_text)} text elements")
        
        return JSONResponse({
            "success": True,
            "results": extracted_text,
            "total_text_len": len(extracted_text),
            "total_text": "\n".join([item['text'] for item in extracted_text]),
            "device": os.getenv('DEVICE', 'cpu')
        })
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr/simple")
async def simple_ocr(file: UploadFile = File(...)):
    """
    Perform simple OCR and return only concatenated text
    """
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not available")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        result = ocr.ocr(img_array, cls=True)
        
        # Extract just the text
        all_text = []
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                continue
            for line in res:
                all_text.append(line[1][0])
        
        combined_text = '\n'.join(all_text)
        
        return JSONResponse({
            "success": True,
            "text": combined_text,
            "device": os.getenv('DEVICE', 'cpu')
        })
        
    except Exception as e:
        logger.error(f"Simple OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9292)