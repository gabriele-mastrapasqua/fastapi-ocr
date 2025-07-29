import time
import sys
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
logger.setLevel(logging.INFO)

def profile_print(msg, start_time=None):
    """Print con timestamp per profiling - force flush per Docker"""
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{current_time:.3f}] {msg} (+{elapsed:.3f}s)", flush=True)
    else:
        print(f"[{current_time:.3f}] {msg}", flush=True)
    return current_time

def ocr_single_page_worker(page_data: Tuple[int, int, bytes, str, str, int]) -> Tuple[int, str, float]:
    """
    Worker function for parallel OCR processing - OTTIMIZZATO.
    Must be defined at module level for pickle serialization.
    """
    page_num, resolution, image_bytes, ocr_lang, ocr_config, force_angle_rotation = page_data
    
    worker_start = time.time()
    profile_print(f"🚀 WORKER START - Page {page_num + 1}")
    
    try:
        # STEP 1: Import (dovrebbe essere veloce se già cached)
        import_start = profile_print(f"→ Page {page_num + 1}: Starting imports...")
        import PIL
        from PIL import Image
        import io
        import app.utils.ocr_utils as utils
        profile_print(f"→ Page {page_num + 1}: Imports completed", import_start)

        # STEP 2: Image loading da bytes
        load_start = profile_print(f"→ Page {page_num + 1}: Loading image from bytes ({len(image_bytes)} bytes)...")
        img_page = Image.open(io.BytesIO(image_bytes))
        profile_print(f"→ Page {page_num + 1}: Image loaded - Size: {img_page.size}, Mode: {img_page.mode}", load_start)

        # STEP 3: Preprocessing (OTTIMIZZATO)
        preproc_start = profile_print(f"→ Page {page_num + 1}: Starting preprocessing...")
        
        # OTTIMIZZAZIONE: Usa preprocessing veloce quando possibile
        if utils.should_use_fast_preprocessing(img_page):
            profile_print(f"→ Page {page_num + 1}: Using FAST preprocessing...")
            preproc_img = utils.preprocess_image_for_ocr_tesseract_fast(img_page)
        else:
            profile_print(f"→ Page {page_num + 1}: Using STANDARD preprocessing...")
            preproc_img = utils.preprocess_image_for_ocr_tesseract(img_page)
        profile_print(f"→ Page {page_num + 1}: Preprocessing completed", preproc_start)

        # STEP 4: Rotation detection (OTTIMIZZATO)
        rotation_start = profile_print(f"→ Page {page_num + 1}: Detecting orientation...")
        
        # OTTIMIZZAZIONE 1: Skip detection se force_angle_rotation è impostato
        if force_angle_rotation != 0:
            profile_print(f"→ Page {page_num + 1}: Skipping rotation detection (forced rotation: {force_angle_rotation})")
            angle_rotation_detected = force_angle_rotation
            image_needs_rotation = True
        # OTTIMIZZAZIONE 2: Skip detection per immagini che probabilmente non ne hanno bisogno
        elif utils.should_skip_rotation_detection(img_page):
            profile_print(f"→ Page {page_num + 1}: Skipping rotation detection (heuristic)")
            angle_rotation_detected = 0
            image_needs_rotation = False
        else:
            # OTTIMIZZAZIONE 3: Usa detection veloce
            profile_print(f"→ Page {page_num + 1}: Using FAST rotation detection...")
            angle_rotation_detected, image_needs_rotation = utils.detect_angle_rotation_tesseract_fast(preproc_img)
            profile_print(f"→ Page {page_num + 1}: Rotation detection completed - Angle: {angle_rotation_detected}°", rotation_start)

        # STEP 5: Image rotation se necessaria
        rotation_applied = 0
        if image_needs_rotation or force_angle_rotation != 0:
            rotate_start = profile_print(f"→ Page {page_num + 1}: Applying rotation...")
            if force_angle_rotation != 0:
                rotation_applied = force_angle_rotation
            else:
                rotation_applied = angle_rotation_detected
            preproc_img = utils.rotate_image_pil(preproc_img, angle=rotation_applied)
            profile_print(f"→ Page {page_num + 1}: Rotation applied ({rotation_applied}°)", rotate_start)

        # STEP 6: OCR con Tesseract (PROBABILMENTE IL COLLO DI BOTTIGLIA PRINCIPALE)
        ocr_start = profile_print(f"→ Page {page_num + 1}: Starting Tesseract OCR...")
        profile_print(f"→ Page {page_num + 1}: OCR Config: '{ocr_config}', Lang: '{ocr_lang}'")
        
        # OTTIMIZZAZIONE: Usa image_to_string direttamente invece di image_to_data se non serve struttura
        ocr_text = pytesseract.image_to_string(preproc_img, config=ocr_config, lang=ocr_lang)
        profile_print(f"→ Page {page_num + 1}: Tesseract OCR completed - Text length: {len(ocr_text)}", ocr_start)
        
        # STEP 7: Text cleaning
        clean_start = profile_print(f"→ Page {page_num + 1}: Cleaning text...")
        ocr_text = utils.clean_text(ocr_text)
        profile_print(f"→ Page {page_num + 1}: Text cleaning completed", clean_start)
        
        # STEP 8: Quality score calculation (OTTIMIZZATO)
        score_start = profile_print(f"→ Page {page_num + 1}: Calculating OCR quality score...")
        
        # OTTIMIZZAZIONE: Usa calcolo veloce invece di seconda chiamata Tesseract
        profile_print(f"→ Page {page_num + 1}: Using FAST score calculation...")
        ocr_score = utils.calculate_ocr_score_fast(ocr_text)
        
        # OPZIONE: Se vuoi il calcolo preciso (ma lento), decomment questa linea:
        # ocr_score = utils.calculate_ocr_score_tesseract(preproc_img)
        
        profile_print(f"→ Page {page_num + 1}: Quality score calculated: {ocr_score}", score_start)
        
        total_time = time.time() - worker_start
        profile_print(f"✅ Page {page_num + 1}: WORKER COMPLETED in {total_time:.3f}s")
        
        return (page_num, ocr_text, ocr_score, rotation_applied)

    except Exception as e:
        error_time = time.time() - worker_start
        profile_print(f"❌ Page {page_num + 1}: WORKER FAILED after {error_time:.3f}s - Error: {str(e)}")
        logger.error(f"❌ OCR failed for page {page_num + 1}: {str(e)}")
        traceback.print_exc()
        return (page_num, "")


class TesseractOCREngine:
    def __init__(self, lang="ita", ocr_config="--psm 11"):
        self.lang = lang
        self.ocr_config = ocr_config
        self.max_workers = min(4, mp.cpu_count())
        
        # OTTIMIZZAZIONI TESSERACT
        # Aggiungi opzioni per velocizzare Tesseract
        base_config = ocr_config
        speed_optimizations = [
            "-c tessedit_do_invert=0",  # Skip invert
            "-c load_system_dawg=0",    # Skip system dictionary  
            "-c load_freq_dawg=0",      # Skip frequency dictionary
            "-c load_punc_dawg=0",      # Skip punctuation dictionary
            "-c load_number_dawg=0",    # Skip number dictionary
            "-c load_unambig_dawg=0",   # Skip unambiguous dictionary
            "-c load_bigram_dawg=0",    # Skip bigram dictionary
            "-c load_fixed_length_dawgs=0",  # Skip fixed length dictionaries
        ]
        self.ocr_config = f"{base_config} {' '.join(speed_optimizations)}"
        
        profile_print(f"🔧 TesseractOCREngine initialized:")
        profile_print(f"   - Language: {self.lang}")
        profile_print(f"   - Config: {self.ocr_config}")
        profile_print(f"   - Max workers: {self.max_workers}")

    def execute_ocr(self, 
            images,                         
            force_angle_rotation=0, 
    ):
        """
        Perform OCR on the given images PIL - VERSIONE OTTIMIZZATA.
        """
        total_start = time.time()
        profile_print(f"🚀 STARTING OCR ENGINE - {len(images)} images, force_rotation={force_angle_rotation}")

        resolution = 300  # OTTIMIZZAZIONE: Riduci risoluzione se la qualità è accettabile
        tasks = []
        all_text = []
        
        # STEP 1: Preparazione tasks
        prep_start = profile_print(f"→ Preparing tasks for {len(images)} images...")
        for i, image in enumerate(images):
            task_start = time.time()
            
            # OTTIMIZZAZIONE: Usa formato più efficiente per il salvataggio
            buf = io.BytesIO()
            # JPEG è più veloce da salvare/caricare rispetto a PNG per foto/scansioni
            format_to_use = "JPEG" if image.mode in ["RGB", "L"] else "PNG"
            quality = 95 if format_to_use == "JPEG" else None
            
            if quality:
                image.save(buf, format=format_to_use, quality=quality, optimize=True)
            else:
                image.save(buf, format=format_to_use, optimize=True)
                
            img_bytes = buf.getvalue()
            tasks.append((i, resolution, img_bytes, self.lang, self.ocr_config, force_angle_rotation))
            all_text.append(None)
            
            task_time = time.time() - task_start
            if task_time > 0.1:  # Log solo se prende più di 100ms
                profile_print(f"   - Image {i+1}: {len(img_bytes)} bytes, took {task_time:.3f}s")
        
        profile_print(f"→ Tasks preparation completed", prep_start)

        # STEP 2: Parallel processing
        parallel_start = profile_print(f"→ Starting parallel OCR with {self.max_workers} workers...")
        
        ocr_scores = []
        ocr_rotations = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(ocr_single_page_worker, task): task[0] 
                for task in tasks
            }
            
            completed_count = 0
            for future in as_completed(future_to_page):
                try:
                    page_num, ocr_text, ocr_score, rotation_applied = future.result()
                    all_text[page_num] = ocr_text
                    ocr_scores.append(ocr_score)
                    ocr_rotations.append(rotation_applied)
                    completed_count += 1
                    profile_print(f"✅ Page {page_num + 1} completed ({completed_count}/{len(tasks)})")
                except Exception as e:
                    page_num = future_to_page[future]
                    profile_print(f"❌ Page {page_num + 1} failed: {str(e)}")
                    all_text[page_num] = ""
                    ocr_scores.append(0)
                    ocr_rotations.append(0)
        
        profile_print(f"→ Parallel OCR completed", parallel_start)
        
        # STEP 3: Results aggregation
        final_start = profile_print(f"→ Aggregating results...")
        extracted_text = "\n".join(all_text)
        avg_confidence = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0.0
        avg_rotation = sum(ocr_rotations) / len(ocr_rotations) if ocr_rotations else 0.0
        
        total_time = time.time() - total_start
        profile_print(f"🎉 OCR ENGINE COMPLETED in {total_time:.3f}s")
        profile_print(f"   - Total text length: {len(extracted_text)}")
        profile_print(f"   - Average confidence: {avg_confidence:.3f}")
        profile_print(f"   - Average time per page: {total_time/len(images):.3f}s")

        return {
            "success": True,
            "results": extracted_text,
            "total_text": extracted_text,
            "rotation_correction": {
                "average_correction_applied": round(avg_rotation, 2),
                "correction_applied": round(ocr_rotations[-1] if ocr_rotations else 0, 2),
                "was_corrected": any(r != 0 for r in ocr_rotations),
                "rotations": ocr_rotations,
            },
            "statistics": {
                "total_blocks": len(extracted_text),
                "average_confidence": avg_confidence,
                "min_confidence": round(min(ocr_scores), 4) if ocr_scores else 0.0,
                "max_confidence": round(max(ocr_scores), 4) if ocr_scores else 0.0,
                "ocr_scores": ocr_scores,
                "total_processing_time": round(total_time, 3),
                "average_time_per_page": round(total_time/len(images), 3),
            },
            "parameters_used": {
                "ocr_config": self.ocr_config,
                "ocr_lang": self.lang,
                "resolution": resolution,
            },
            "device": 'cpu'
        }