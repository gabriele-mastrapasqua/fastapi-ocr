from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pytesseract import image_to_data, Output
import pytesseract
import traceback

import app.utils.ocr_utils as utils
from typing import List, Optional, Dict, Any, Tuple
import io
from PIL import Image

# Setup logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Imposta solo il livello


def ocr_single_page_worker(page_data: Tuple[int, int, bytes, str, str, int]) -> Tuple[int, str, float]:
    """
    Worker function for parallel OCR processing.
    Must be defined at module level for pickle serialization.
    """
    page_num, resolution, image_bytes, ocr_lang, ocr_config, force_angle_rotation = page_data
    
    import traceback
    try:
        import PIL
        from PIL import Image
        import io
        import app.utils.ocr_utils as utils
        
        print(f"→ Processing page {page_num + 1} with resolution {resolution}...")
        # Convert bytes to PIL Image
        img_page = Image.open(io.BytesIO(image_bytes))

        # Preprocess image for better OCR
        print(f"→ Preprocessing image for OCR...")
        preproc_img = utils.preprocess_image_for_ocr_tesseract(img_page)

        # OSD = Orientation and Script Detection
        print(f"→ Detecting orientation and rotate if necessary...")
        angle_rotation_detected, image_needs_rotation = utils.detect_angle_rotation_tesseract(preproc_img)
        print(f"Detected rotation angle: {angle_rotation_detected} degrees")

        # TEST 
        #rotation_applied = -90
        rotation_applied = 0
        if image_needs_rotation or force_angle_rotation != 0:
            if force_angle_rotation != 0:
                rotation_applied = force_angle_rotation
            else:
                rotation_applied = angle_rotation_detected
            # then rotate the image
            print(f"→ Rotating image by {rotation_applied} degrees...")
            preproc_img = utils.rotate_image_pil(preproc_img, angle=rotation_applied)

        
        # run tesseract OCR
        """
            --psm 4  # assume layout in colonne
            --psm 6  # assume layout uniforme, testo in blocchi
            --psm 11 # sparse text
        """
        print(f"→ Running OCR with config: {ocr_config} and lang: {ocr_lang}...")
        ocr_text = pytesseract.image_to_string(preproc_img, config=ocr_config, lang=ocr_lang)
        
        
        # Clean text
        ocr_text = utils.clean_text(ocr_text)
        
        # calculate OCR quality score
        print(f"→ Page {page_num + 1} OCR completed, calculating quality score...")
        ocr_score = utils.calculate_ocr_score_tesseract(preproc_img)
        print(f"✅ Page {page_num + 1} OCR completed with score {ocr_score}")
        
        return (page_num, ocr_text, ocr_score, rotation_applied)

    except Exception as e:
        logger.error(f"❌ OCR failed for page {page_num + 1}: {str(e)}")
        traceback.print_exc()
        return (page_num, "")


class TesseractOCREngine:
    def __init__(self, lang = "ita", ocr_config = "--psm 11"):
        self.lang = lang
        self.ocr_config = ocr_config
        self.max_workers = mp.cpu_count()  # Use all available CPU cores

    def execute_ocr(self, 
            images,                         # list of images (PIL Image or base64 strings)
            force_angle_rotation=0, 
    ):
        """
        Perform OCR on the given images PIL.
        :param images: PIL Image or list of PIL Images.
        :param force_angle_rotation: If not 0, forces the rotation of the image before OCR.
        :return: The OCR results.

        """

        resolution = 300  # DPI resolution for OCR
        tasks=[]
        all_text = []
        for i, image in enumerate(images):
            # convert PIL Image to bytes to be passed to another process 
            buf = io.BytesIO()
            format = "PNG" if image.mode == "RGB" else "JPEG"
            image.save(buf, format=format)
            img_bytes = buf.getvalue()
            tasks.append((i, resolution, img_bytes, self.lang, self.ocr_config, force_angle_rotation))
            all_text.append(None) # Initialize with None for each page

        # schedule parallel OCR processing
        ocr_scores = []
        ocr_rotations = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all OCR tasks
            future_to_page = {
                executor.submit(ocr_single_page_worker, task): task[0] 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                try:
                    page_num, ocr_text, ocr_score, rotation_applied = future.result()
                    all_text[page_num] = ocr_text
                    ocr_scores.append(ocr_score)
                    ocr_rotations.append(rotation_applied)
                    logger.info(f"✅ OCR completed for page {page_num + 1}")
                except Exception as e:
                    page_num = future_to_page[future]
                    logger.error(f"❌ OCR failed for page {page_num + 1}: {str(e)}")
                    all_text[page_num] = ""
                    ocr_scores.append(0)  # Default score for failed OCR
                    ocr_rotations.append(0)  # Default score for failed OCR
        
        extracted_text = "\n".join(all_text)
        logger.info(f"🎉 Parallel OCR processing completed for {len(tasks)} pages, with tesseract ocr score {ocr_score} and text all len {len(extracted_text)}")

        avg_confidence = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0.0
        avg_rotation = sum(ocr_rotations) / len(ocr_rotations) if ocr_rotations else 0.0

        return {
            "success": True,
            "results": extracted_text,
            "total_text": extracted_text,
            "rotation_correction": {
                "average_correction_applied": round(avg_rotation, 2),
                "correction_applied": round(rotation_applied, 2),
                "was_corrected": rotation_applied != 0,
                "rotations": ocr_rotations,
            },
            "statistics": {
                "total_blocks": len(extracted_text),
                "average_confidence": avg_confidence,
                "min_confidence": round(min(ocr_scores), 4) if ocr_scores else 0.0,
                "max_confidence": round(max(ocr_scores), 4) if ocr_scores else 0.0,
                "ocr_scores": ocr_scores,
            },
            "parameters_used": {
                "ocr_config": self.ocr_config,
                "ocr_lang": self.lang,
            },
            "device": 'cpu'
        }
    
