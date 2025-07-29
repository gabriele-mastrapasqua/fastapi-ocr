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
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Imposta solo il livello


class PaddleOCREngine:
    def __init__(self, lang = "it"):
        # Initialize PaddleOCR
        try:
            device = os.getenv('DEVICE', 'cpu')
            use_gpu = device.lower() == 'gpu'
            
            logger.info(f"Initializing PaddleOCR with device: {device}")
            self.ocr = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang=lang,  # Puoi cambiare la lingua qui
                use_gpu=use_gpu,
                show_log=True,
                det_model_dir=None,  # Usa modelli default
                rec_model_dir=None,
                cls_model_dir=None,
                #cls_model_dir="/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/",
                ocr_version='PP-OCRv3',
                cpu_threads=os.cpu_count(), 
                
            )
            logger.info("PaddleOCR initialized successfully")

            # Log informazioni sui modelli (se disponibili)
            try:
                logger.info(f"Detection model: {getattr(self.ocr, 'det_predictor', 'Default')}")
                logger.info(f"Recognition model: {getattr(self.ocr, 'rec_predictor', 'Default')}")
                logger.info(f"Classification model: {getattr(self.ocr, 'cls_predictor', 'Default')}")
            except:
                logger.info("Using default PaddleOCR models")

        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None


    def execute_ocr(self, 
            image, 
            force_angle_rotation=0, 
            det_limit_side_len: int = 960,  # Risoluzione per detection
            det_limit_type: str = 'max',    # 'max' o 'min'
            rec_batch_num: int = 6,         # Batch size per recognition
            max_text_length: int = 25       # Lunghezza massima testo riconosciuto):) -> dict:
    ):
        """
        Perform OCR on the given image array.
        :param img_array: The image array to perform OCR on.
        :param force_angle_rotation: If not 0, forces the rotation of the image before OCR.
        :return: The OCR results.
        """
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # OCR con parametri personalizzati
        logger.info(f"Advanced OCR with det_limit_side_len: {det_limit_side_len}, rec_batch_num: {rec_batch_num}")
        

        # Applica i parametri direttamente all'OCR
        self.ocr.det_limit_side_len = det_limit_side_len
        self.ocr.det_limit_type = det_limit_type
        self.ocr.rec_batch_num = rec_batch_num
        self.ocr.max_text_length = max_text_length

        # TODO detect rotation of the image and rotate to improve OCR results
        angle_rotation_detected, image_needs_rotation = utils.detect_angle_rotation_tesseract(preproc_img=image)
        logger.info(f"Detected rotation angle: {angle_rotation_detected} degrees")

        # TEST 
        #rotation_applied = -90
        rotation_applied = 0
        if image_needs_rotation or force_angle_rotation != 0:
            if force_angle_rotation != 0:
                rotation_applied = force_angle_rotation
            else:
                rotation_applied = angle_rotation_detected
        img_array = utils.rotate_image_numpy(img_array, angle=rotation_applied)  # Forza rotazione a 0 per test

        # Usa OCR con paddleOCR per migliori risultati
        initial_result = self.ocr.ocr(img_array, cls=True)
        result = initial_result

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
        
        return {
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
        }
    
