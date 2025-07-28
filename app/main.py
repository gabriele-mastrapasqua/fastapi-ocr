from fastapi import FastAPI, File, UploadFile, HTTPException
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

        original_img = img_array.copy()
        rotation_applied = 0
        corrected_img = img_array
        
        # Step 1: Rileva la rotazione usando solo il full OCR 
        # (evita il problema con rec=False, cls=False)
        logger.info("Detecting document rotation...")
        
        try:
            # Usa OCR completo per rilevare rotazione
            initial_result = ocr.ocr(img_array, cls=True)
            
            if initial_result and initial_result[0] and len(initial_result[0]) > 0:
                # Calcola l'angolo di rotazione dai bounding box
                angles = []
                for line in initial_result[0]:
                    bbox = line[0]
                    p1, p2 = bbox[0], bbox[1]
                    angle_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                    angle_deg = np.degrees(angle_rad)
                    
                    # Normalizza l'angolo tra -90 e 90
                    if angle_deg > 90:
                        angle_deg -= 180
                    elif angle_deg < -90:
                        angle_deg += 180
                        
                    angles.append(angle_deg)
                
                if angles:
                    document_rotation = np.median(angles)  # Usa mediana per robustezza
                    
                    # Applica correzione se necessario (soglia di 5 gradi)
                    if abs(document_rotation) > 5:
                        rotation_applied = -document_rotation
                        corrected_img = utils.rotate_image(img_array, rotation_applied)
                        logger.info(f"Applied rotation correction: {rotation_applied:.2f}°")
                        
                        # Re-applica OCR sull'immagine corretta
                        logger.info("Performing OCR on rotation-corrected image...")
                        result = ocr.ocr(corrected_img, cls=True)
                    else:
                        # Usa il risultato iniziale se non serve correzione
                        result = initial_result
                        logger.info("No rotation correction needed")
                else:
                    result = initial_result
                    logger.info("Could not determine rotation angle, using original image")
            else:
                logger.info("No text detected for rotation analysis, using original image")
                result = initial_result
                
        except Exception as rotation_error:
            logger.warning(f"Rotation detection failed: {rotation_error}, proceeding with original image")
            result = ocr.ocr(img_array, cls=True)
            rotation_applied = 0

        # Step 2: Ordina i risultati
        sorted_result = utils.sort_text_blocks(result)
        
        # Step 3: Process results
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
                })
        
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        
        logger.info(f"OCR completed: {len(extracted_text)} elements, avg confidence: {avg_confidence}, rotation: {rotation_applied:.2f}°")
        
        return JSONResponse({
            "success": True,
            "results": extracted_text,
            "total_text": "\n".join([item['text'] for item in extracted_text]),
            "rotation_correction": {
                "original_rotation_detected": round(-rotation_applied, 2) if rotation_applied != 0 else 0,
                "correction_applied": round(rotation_applied, 2),
                "was_corrected": rotation_applied != 0
            },
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9292)