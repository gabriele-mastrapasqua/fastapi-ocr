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


def sort_text_blocks(results):
    """
    Ordina i blocchi di testo da sinistra a destra, dall'alto al basso (stile italiano)
    """
    if not results or results[0] is None:
        return results
    
    # Estrai tutti i blocchi con le loro coordinate
    blocks = []
    for line in results[0]:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]
        
        # Calcola il centro del bounding box
        center_x = sum([point[0] for point in bbox]) / 4
        center_y = sum([point[1] for point in bbox]) / 4
        
        blocks.append({
            'bbox': bbox,
            'text': text,
            'confidence': confidence,
            'center_x': center_x,
            'center_y': center_y
        })
    
    # Ordina prima per Y (dall'alto al basso), poi per X (da sinistra a destra)
    # Usa una tolleranza per Y per gestire testi sulla stessa riga
    y_tolerance = 20  # pixel di tolleranza per considerare testi sulla stessa riga
    
    blocks.sort(key=lambda block: (block['center_y'] // y_tolerance, block['center_x']))
    
    # Ricostruisci il formato originale
    sorted_results = []
    for block in blocks:
        sorted_results.append([
            block['bbox'],
            [block['text'], block['confidence']]
        ])
    
    return [sorted_results]

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
        det_model_dir=None,  # Usa modelli default
        rec_model_dir=None,
        cls_model_dir=None
    )
    logger.info("PaddleOCR initialized successfully")

    # Log informazioni sui modelli (se disponibili)
    try:
        logger.info(f"Detection model: {getattr(ocr, 'det_predictor', 'Default')}")
        logger.info(f"Recognition model: {getattr(ocr, 'rec_predictor', 'Default')}")
        logger.info(f"Classification model: {getattr(ocr, 'cls_predictor', 'Default')}")
    except:
        logger.info("Using default PaddleOCR models")

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
async def perform_ocr(
    file: UploadFile = File(...),
    det_limit_side_len: int = 960,  # Risoluzione per detection
    det_limit_type: str = 'max',    # 'max' o 'min'
    rec_batch_num: int = 6,         # Batch size per recognition
    max_text_length: int = 25       # Lunghezza massima testo riconosciuto):
):
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
        
        # OCR con parametri personalizzati
        logger.info(f"Advanced OCR with det_limit_side_len: {det_limit_side_len}, rec_batch_num: {rec_batch_num}")
        
        # Applica i parametri direttamente all'OCR
        ocr.det_limit_side_len = det_limit_side_len
        ocr.det_limit_type = det_limit_type
        ocr.rec_batch_num = rec_batch_num
        ocr.max_text_length = max_text_length

        # Perform OCR
        logger.info("Starting OCR processing")
        result = ocr.ocr(img_array, cls=True)

        # Ordina i risultati
        sorted_result = sort_text_blocks(result)
        
        
        # Process results
        # Process results
        extracted_text = []
        confidences = []
        for idx in range(len(sorted_result)):
            res = sorted_result[idx]
            if res is None:
                continue

            for line in res:
                text = line[1][0]
                confidence = line[1][1]
                bbox = line[0]
                confidences.append(confidence)
                extracted_text.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    # "bbox": bbox,
                    # "bbox_center": {
                    #     "x": round(sum([point[0] for point in bbox]) / 4, 2),
                    #     "y": round(sum([point[1] for point in bbox]) / 4, 2)
                    # }
                })
        
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        
        logger.info(f"Advanced OCR completed, found {len(extracted_text)} elements, avg confidence: {avg_confidence}")
        
        return JSONResponse({
            "success": True,
            "results": extracted_text,
            "total_text": "\n".join([item['text'] for item in extracted_text]),
            "statistics": {
                "total_blocks": len(extracted_text),
                "average_confidence": avg_confidence,
                "min_confidence": round(min(confidences), 4) if confidences else 0.0,
                "max_confidence": round(max(confidences), 4) if confidences else 0.0
            },
            "parameters_used": {
                "det_limit_side_len": det_limit_side_len,
                "det_limit_type": det_limit_type,
                "rec_batch_num": rec_batch_num,
                "max_text_length": max_text_length
            },
            "device": os.getenv('DEVICE', 'cpu')
        })
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9292)