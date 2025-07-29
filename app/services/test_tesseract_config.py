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

# DIAGNOSI E FIX PERFORMANCE TESSERACT

def profile_print(msg, start_time=None):
    """Print con timestamp per profiling"""
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{current_time:.3f}] {msg} (+{elapsed:.3f}s)", flush=True)
    else:
        print(f"[{current_time:.3f}] {msg}", flush=True)
    return current_time

def analyze_image_for_ocr(img: Image.Image) -> dict:
    """
    Analizza un'immagine per identificare cosa rallenta Tesseract.
    """
    analysis = {
        'size_pixels': img.size[0] * img.size[1],
        'dimensions': img.size,
        'mode': img.mode,
        'megapixels': round((img.size[0] * img.size[1]) / 1000000, 2)
    }
    
    # Calcola complexity
    if analysis['megapixels'] > 5:
        analysis['complexity'] = 'VERY_HIGH'
    elif analysis['megapixels'] > 2:
        analysis['complexity'] = 'HIGH'  
    elif analysis['megapixels'] > 1:
        analysis['complexity'] = 'MEDIUM'
    else:
        analysis['complexity'] = 'LOW'
    
    return analysis

def resize_image_for_fast_ocr(img: Image.Image, target_mp: float = 1.0) -> Image.Image:
    """
    Ridimensiona intelligentemente l'immagine per OCR veloce.
    target_mp: megapixel target (default 1.0 = 1 milione di pixel)
    """
    current_mp = (img.size[0] * img.size[1]) / 1000000
    
    if current_mp <= target_mp:
        return img
    
    # Calcola il fattore di scala
    scale_factor = (target_mp / current_mp) ** 0.5
    new_width = int(img.size[0] * scale_factor)
    new_height = int(img.size[1] * scale_factor)
    
    print(f"   → Resizing from {img.size} ({current_mp:.2f}MP) to ({new_width}x{new_height}) ({target_mp:.2f}MP)", flush=True)
    
    # Usa LANCZOS per qualità migliore durante il downscaling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def get_optimal_tesseract_config(image_analysis: dict, fast_mode: bool = True) -> str:
    """
    Genera config Tesseract ottimale basato sull'analisi dell'immagine.
    """
    base_configs = {
        'ULTRA_FAST': [
            "--psm 6",  # Assume uniform block of text
            "-c tessedit_pageseg_mode=6",
            "-c preserve_interword_spaces=1",
        ],
        'FAST': [
            "--psm 6",
            "-c tessedit_pageseg_mode=6", 
            "-c preserve_interword_spaces=1",
            "-c load_system_dawg=0",
            "-c load_freq_dawg=0",
        ],
        'BALANCED': [
            "--psm 6",
            "-c preserve_interword_spaces=1",
            "-c load_system_dawg=0",
            "-c load_freq_dawg=0",
            "-c load_punc_dawg=0",
        ]
    }
    
    # Scegli config basato su complexity
    if image_analysis['complexity'] in ['VERY_HIGH', 'HIGH'] or fast_mode:
        config_parts = base_configs['ULTRA_FAST']
    elif image_analysis['complexity'] == 'MEDIUM':
        config_parts = base_configs['FAST'] 
    else:
        config_parts = base_configs['BALANCED']
    
    # Aggiungi ottimizzazioni comuni
    common_optimizations = [
        "-c tessedit_do_invert=0",
        "-c tessedit_make_box_file=0", 
        "-c tessedit_write_images=0",
        "-c classify_bln_numeric_mode=0",
    ]
    
    return " ".join(config_parts + common_optimizations)

def test_tesseract_performance(img: Image.Image, lang: str = "ita") -> dict:
    """
    Test performance Tesseract con diverse configurazioni.
    """
    results = {}
    
    # Test 1: Immagine originale con config minimale
    print(f"🧪 Testing original image size: {img.size}", flush=True)
    
    minimal_config = "--psm 6 -c tessedit_do_invert=0"
    start_time = time.time()
    try:
        text1 = pytesseract.image_to_string(img, config=minimal_config, lang=lang)
        time1 = time.time() - start_time
        results['original_minimal'] = {'time': time1, 'text_len': len(text1)}
        print(f"   → Original + minimal config: {time1:.3f}s ({len(text1)} chars)", flush=True)
    except Exception as e:
        results['original_minimal'] = {'time': 999, 'error': str(e)}
        print(f"   → Original + minimal config FAILED: {str(e)}", flush=True)
    
    # Test 2: Immagine ridimensionata
    small_img = resize_image_for_fast_ocr(img, target_mp=0.5)  # 0.5 megapixel
    print(f"🧪 Testing resized image: {small_img.size}", flush=True)
    
    start_time = time.time()
    try:
        text2 = pytesseract.image_to_string(small_img, config=minimal_config, lang=lang)
        time2 = time.time() - start_time
        results['resized_minimal'] = {'time': time2, 'text_len': len(text2)}
        print(f"   → Resized + minimal config: {time2:.3f}s ({len(text2)} chars)", flush=True)
    except Exception as e:
        results['resized_minimal'] = {'time': 999, 'error': str(e)}
        print(f"   → Resized + minimal config FAILED: {str(e)}", flush=True)
    
    # Test 3: Lingua inglese invece di italiana
    print(f"🧪 Testing with English language", flush=True)
    start_time = time.time()
    try:
        text3 = pytesseract.image_to_string(small_img, config=minimal_config, lang="eng")
        time3 = time.time() - start_time
        results['resized_english'] = {'time': time3, 'text_len': len(text3)}
        print(f"   → Resized + English: {time3:.3f}s ({len(text3)} chars)", flush=True)
    except Exception as e:
        results['resized_english'] = {'time': 999, 'error': str(e)}
        print(f"   → Resized + English FAILED: {str(e)}", flush=True)
    
    return results

def ocr_single_page_worker_performance_test(page_data: Tuple[int, int, bytes, str, str, int]) -> Tuple[int, str, float]:
    """
    Worker per test performance con diverse ottimizzazioni.
    """
    page_num, resolution, image_bytes, ocr_lang, ocr_config, force_angle_rotation = page_data
    
    worker_start = time.time()
    print(f"🔬 PERFORMANCE TEST WORKER - Page {page_num + 1}", flush=True)
    
    try:
        # Caricamento
        img_page = Image.open(io.BytesIO(image_bytes))
        print(f"→ Page {page_num + 1}: Loaded image {img_page.size}, mode: {img_page.mode}", flush=True)
        
        # Analisi immagine
        analysis = analyze_image_for_ocr(img_page)
        print(f"→ Page {page_num + 1}: Analysis: {analysis}", flush=True)
        
        # Preprocessing minimo
        if img_page.mode != 'L':
            preproc_img = img_page.convert('L')
        else:
            preproc_img = img_page
            
        # SKIP ROTATION per focus su Tesseract
        print(f"→ Page {page_num + 1}: SKIPPING rotation detection for pure OCR test", flush=True)
        
        # **TEST CRITICO**: Diverse ottimizzazioni Tesseract
        print(f"→ Page {page_num + 1}: Running Tesseract performance tests...", flush=True)
        perf_results = test_tesseract_performance(preproc_img, ocr_lang)
        
        # Usa il risultato migliore
        best_result = min(perf_results.items(), key=lambda x: x[1].get('time', 999))
        best_method, best_data = best_result
        
        print(f"→ Page {page_num + 1}: BEST METHOD: {best_method} in {best_data['time']:.3f}s", flush=True)
        
        # Per il risultato finale, usa metodo ottimizzato
        if analysis['megapixels'] > 1.0:
            # Ridimensiona se troppo grande
            final_img = resize_image_for_fast_ocr(preproc_img, target_mp=0.8)
        else:
            final_img = preproc_img
            
        # Config ottimale
        optimal_config = get_optimal_tesseract_config(analysis, fast_mode=True)
        
        # OCR finale ottimizzato
        final_start = time.time()
        print(f"→ Page {page_num + 1}: Final OCR with optimal settings...", flush=True)
        print(f"→ Page {page_num + 1}: Final image size: {final_img.size}", flush=True)
        print(f"→ Page {page_num + 1}: Final config: {optimal_config}", flush=True)
        
        # PROVA PRIMA LINGUA INGLESE (spesso più veloce)
        try:
            ocr_text = pytesseract.image_to_string(final_img, config=optimal_config, lang="eng")
            final_time = time.time() - final_start
            print(f"→ Page {page_num + 1}: Final OCR (ENG) completed in {final_time:.3f}s", flush=True)
        except:
            # Fallback su italiano
            ocr_text = pytesseract.image_to_string(final_img, config=optimal_config, lang=ocr_lang)
            final_time = time.time() - final_start
            print(f"→ Page {page_num + 1}: Final OCR (ITA) completed in {final_time:.3f}s", flush=True)
        
        # Cleanup
        ocr_text = ocr_text.strip()
        ocr_score = len(ocr_text) / 100  # Score approssimativo
        
        total_time = time.time() - worker_start
        print(f"✅ Page {page_num + 1}: PERFORMANCE TEST COMPLETED in {total_time:.3f}s", flush=True)
        print(f"   📊 Final OCR time: {final_time:.3f}s, Text: {len(ocr_text)} chars", flush=True)
        
        return (page_num, ocr_text, ocr_score, 0)

    except Exception as e:
        error_time = time.time() - worker_start
        print(f"❌ Page {page_num + 1}: PERFORMANCE TEST FAILED after {error_time:.3f}s", flush=True)
        import traceback
        traceback.print_exc()
        return (page_num, "", 0.0, 0)
