import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import paddleocr
import pytesseract
import pdfplumber
import tempfile
import io
import base64
import os
import time

def profile_print(msg, start_time=None):
    """Print con timestamp per profiling - force flush per Docker"""
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{current_time:.3f}] {msg} (+{elapsed:.3f}s)", flush=True)
    else:
        print(f"[{current_time:.3f}] {msg}", flush=True)
    return current_time


def resize_image_for_fast_ocr(img: Image.Image, target_mp: float = 0.8) -> Image.Image:
    """
    Ridimensiona intelligentemente per OCR veloce.
    QUESTA È LA CHIAVE DELLE PERFORMANCE!
    """
    current_mp = (img.size[0] * img.size[1]) / 1000000
    
    if current_mp <= target_mp:
        return img
    
    # Calcola fattore di scala per raggiungere target megapixel
    scale_factor = (target_mp / current_mp) ** 0.5
    new_width = int(img.size[0] * scale_factor)
    new_height = int(img.size[1] * scale_factor)
    
    print(f"   → CRITICAL RESIZE: {img.size} ({current_mp:.2f}MP) → ({new_width}x{new_height}) ({target_mp:.2f}MP)", flush=True)
    
    # LANCZOS per qualità ottimale
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def smart_resize_for_ocr(img: Image.Image, max_dimension: int = 2000, min_mp: float = 2.0) -> Image.Image:
    """
    Resize intelligente che mantiene la qualità OCR.
    
    Args:
        img: Immagine PIL
        max_dimension: Dimensione massima per lato (default: 2000px)
        min_mp: Megapixel minimi da mantenere (default: 2.0MP)
    
    Returns:
        Immagine ridimensionata mantenendo qualità OCR
    """
    current_mp = (img.size[0] * img.size[1]) / 1000000
    width, height = img.size
    max_current_dimension = max(width, height)
    
    print(f"   → Original: {img.size} ({current_mp:.2f}MP)", flush=True)
    
    # Se l'immagine è già piccola, non toccarla
    if current_mp <= min_mp and max_current_dimension <= max_dimension:
        print(f"   → Image already optimal, no resize needed", flush=True)
        return img
    
    # Calcola nuovo resize basato su dimensione massima E megapixel minimi
    scale_by_dimension = max_dimension / max_current_dimension if max_current_dimension > max_dimension else 1.0
    scale_by_mp = (min_mp / current_mp) ** 0.5 if current_mp > min_mp else 1.0
    
    # Usa il fattore più conservativo (che riduce meno)
    scale_factor = max(scale_by_dimension, scale_by_mp)
    
    # Non ridimensionare se il fattore è molto vicino a 1
    if scale_factor > 0.9:
        print(f"   → Scale factor {scale_factor:.3f} too close to 1, keeping original", flush=True)
        return img
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_mp = (new_width * new_height) / 1000000
    
    print(f"   → QUALITY RESIZE: {img.size} ({current_mp:.2f}MP) → ({new_width}x{new_height}) ({new_mp:.2f}MP)", flush=True)
    print(f"   → Scale factor: {scale_factor:.3f}", flush=True)
    
    # Usa LANCZOS per qualità ottimale
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Migliora l'immagine per OCR mantenendo le dimensioni.
    Applica solo miglioramenti che non cambiano la risoluzione.
    """
    enhance_start = time.time()
    
    # Se già in grayscale, applica solo sharpening leggero
    if img.mode == 'L':
        # Sharpening leggero per migliorare il testo
        from PIL import ImageFilter, ImageEnhance
        
        # Applica un leggero sharpening
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Migliora contrasto leggermente
        enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = enhancer.enhance(1.1)
        
        enhance_time = time.time() - enhance_start
        print(f"   → Image enhancement completed in {enhance_time:.3f}s", flush=True)
        return enhanced
    
    # Se a colori, converti a grayscale con preprocessing ottimizzato
    import cv2
    import numpy as np
    
    # Converti PIL a numpy
    img_array = np.array(img)
    
    # Converti a BGR per OpenCV
    if img.mode == 'RGB':
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Converti a grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # per migliorare il contrasto locale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Applica threshold adattivo per migliorare il testo
    # Ma solo se aiuta (alcuni documenti sono già ben contrastati)
    mean_brightness = np.mean(enhanced)
    if mean_brightness < 200:  # Solo se l'immagine non è già molto chiara
        enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Riconverti a PIL
    result = Image.fromarray(enhanced, mode='L')
    
    enhance_time = time.time() - enhance_start
    print(f"   → Advanced image enhancement completed in {enhance_time:.3f}s", flush=True)
    return result

def pdf_to_images(contents, base_64 = False):
    """
    Convert a PDF file bytes content to a list of images.
    
    @contents: PDF file bytes content.
    @base_64: If True, return images as base64 strings, otherwise as PIL

    Returns: list of images (PIL Image or base64 strings). + num pages in PDF
    """
    try:
        tmp_file_path = None
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
            
        images = []
        with pdfplumber.open(tmp_file_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                # Convert page to image
                image = page.to_image(resolution=300).original  # Era 300
                
                # added: smart resize for good OCR performance and quality
                image = smart_resize_for_ocr(image, max_dimension=1000,  min_mp=1.5, )

                if base_64:
                    # Convert PIL Image to base64 string
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    images.append(img_str)
                else:
                    # Append PIL Image directly
                    images.append(image)
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)    
    
    return images, num_pages

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

def rotate_image_numpy(image, angle):
    """
    Ruota un'immagine dell'angolo specificato
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calcola la matrice di rotazione
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcola le nuove dimensioni dell'immagine
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Aggiusta la matrice di rotazione per tenere conto della traslazione
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Applica la rotazione
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def rotate_image_pil(image: Image.Image, angle: int) -> Image.Image:
    """
    Ruota un'immagine PIL dell'angolo specificato.
    OTTIMIZZAZIONE: Usa expand=True per evitare crop
    """
    rotation_start = profile_print(f"   → PIL rotation starting ({angle}°)...")
    result = image.rotate(angle, expand=True)
    profile_print(f"   → PIL rotation completed", rotation_start)
    return result

def detect_angle_rotation_tesseract(preproc_img: Image.Image) -> tuple:
    """
    Detect angle rotation using Tesseract OCR.
    ⚠️  QUESTA È LA FUNZIONE PIÙ LENTA! ⚠️ 
    Returns (angle, needs_rotation)
    """
    osd_start = profile_print(f"   → OSD detection starting...")
    
    try:
        # COLLO DI BOTTIGLIA PRINCIPALE: image_to_osd è MOLTO lento
        osd = pytesseract.image_to_osd(preproc_img)
        profile_print(f"   → OSD detection completed", osd_start)
        
        profile_print(f"   → OSD Output: {osd}")

        # Estrai angolo di rotazione
        rotation_line = [line for line in osd.split('\n') if "Orientation in degrees" in line]
        if rotation_line:
            angle = int(rotation_line[0].split(":")[1].strip())
            profile_print(f"   → Rotation detected: {angle}°")

            needs_rotation = angle != 0
            if angle != 0:
                corrected_angle = - ((360 - angle) % 360)
                return corrected_angle, needs_rotation
        
        profile_print(f"   → No rotation needed")
        return 0, False
        
    except Exception as e:
        profile_print(f"   → OSD detection FAILED: {str(e)}", osd_start)
        return 0, False

def detect_angle_rotation_tesseract_fast(preproc_img: Image.Image) -> tuple:
    """
    VERSIONE OTTIMIZZATA della detection di rotazione.
    Usa un'immagine ridimensionata per velocizzare OSD.
    """
    osd_start = profile_print(f"   → FAST OSD detection starting...")
    
    try:
        # OTTIMIZZAZIONE 1: Ridimensiona l'immagine per OSD
        original_size = preproc_img.size
        scale_factor = min(800 / max(original_size), 1.0)  # Max 800px sulla dimensione maggiore
        
        if scale_factor < 1.0:
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            small_img = preproc_img.resize(new_size, Image.Resampling.LANCZOS)
            profile_print(f"   → Image resized from {original_size} to {new_size} for OSD")
        else:
            small_img = preproc_img
            
        # OTTIMIZZAZIONE 2: Config OSD ottimizzato
        osd_config = "--psm 0 -c min_characters_to_try=10"
        
        osd = pytesseract.image_to_osd(small_img, config=osd_config)
        profile_print(f"   → FAST OSD detection completed", osd_start)
        
        # Estrai angolo di rotazione
        rotation_line = [line for line in osd.split('\n') if "Orientation in degrees" in line]
        if rotation_line:
            angle = int(rotation_line[0].split(":")[1].strip())
            profile_print(f"   → Rotation detected: {angle}°")

            needs_rotation = angle != 0
            if angle != 0:
                corrected_angle = - ((360 - angle) % 360)
                return corrected_angle, needs_rotation
        
        profile_print(f"   → No rotation needed")
        return 0, False
        
    except Exception as e:
        profile_print(f"   → FAST OSD detection FAILED: {str(e)}", osd_start)
        return 0, False

def preprocess_image_for_ocr_tesseract(img_page: Image.Image) -> Image.Image:
    """
    Preprocess the image for OCR by converting to BGR, applying grayscale and thresholding.
    VERSIONE OTTIMIZZATA con profiling.
    """
    preproc_start = profile_print(f"   → Image preprocessing starting...")
    
    # Conversione a numpy array
    conv_start = profile_print(f"   → Converting PIL to numpy...")
    img = np.array(img_page)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    profile_print(f"   → PIL to numpy conversion completed", conv_start)
    
    # OTTIMIZZAZIONE: Commenta la parte HSV che era già commentata
    # per evitare elaborazioni non necessarie
    
    # Conversione grayscale
    gray_start = profile_print(f"   → Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    profile_print(f"   → Grayscale conversion completed", gray_start)

    # Apply adaptive thresholding
    thresh_start = profile_print(f"   → Applying OTSU thresholding...")
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile_print(f"   → OTSU thresholding completed", thresh_start)
    
    # Conversione back to PIL
    pil_start = profile_print(f"   → Converting back to PIL...")
    preproc_img = Image.fromarray(thresh)
    profile_print(f"   → PIL conversion completed", pil_start)
    
    profile_print(f"   → Image preprocessing COMPLETED", preproc_start)
    return preproc_img

def preprocess_image_for_ocr_tesseract_fast(img_page: Image.Image) -> Image.Image:
    """
    VERSIONE ULTRA-VELOCE del preprocessing.
    Skip operazioni non essenziali.
    """
    preproc_start = profile_print(f"   → FAST preprocessing starting...")
    
    # OTTIMIZZAZIONE: Se l'immagine è già in grayscale, skip conversioni
    if img_page.mode == 'L':
        profile_print(f"   → Image already grayscale, skipping conversions")
        # Apply solo thresholding semplice
        img_array = np.array(img_page)
        _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        result = Image.fromarray(thresh)
        profile_print(f"   → FAST preprocessing COMPLETED", preproc_start)
        return result
    
    # Conversione diretta RGB -> Grayscale usando PIL (più veloce)
    gray_img = img_page.convert('L')
    
    # Thresholding semplice invece di OTSU (più veloce)
    img_array = np.array(gray_img)
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    result = Image.fromarray(thresh)
    profile_print(f"   → FAST preprocessing COMPLETED", preproc_start)
    return result

def calculate_ocr_score_tesseract(preproc_img: Image.Image) -> float:
    """
    ⚠️  QUESTO È UN ALTRO COLLO DI BOTTIGLIA! ⚠️ 
    Fa una seconda passata OCR solo per calcolare il punteggio.
    """
    from pytesseract import Output
    
    score_start = profile_print(f"   → OCR score calculation starting...")
    
    # QUESTO È LENTO: fa un'altra chiamata a Tesseract
    data = pytesseract.image_to_data(preproc_img, output_type=Output.DICT)
    profile_print(f"   → OCR data extraction completed", score_start)
    
    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    if not confidences:
        return 0
    avg_score = sum(confidences) / len(confidences)
    ocr_score = round(avg_score, 2)
    
    profile_print(f"   → OCR score calculated: {ocr_score}")
    return ocr_score

def calculate_ocr_score_fast(ocr_text: str) -> float:
    """
    VERSIONE VELOCE: calcola un punteggio approssimativo basato sul testo estratto
    invece di fare una seconda chiamata a Tesseract.
    """
    if not ocr_text.strip():
        return 0.0
    
    # Calcola un punteggio basato su:
    # - Presenza di caratteri alfanumerici
    # - Rapporto caratteri validi/totali
    # - Presenza di parole complete
    
    total_chars = len(ocr_text)
    if total_chars == 0:
        return 0.0
    
    alphanumeric_chars = sum(1 for c in ocr_text if c.isalnum())
    words = ocr_text.split()
    valid_words = [w for w in words if len(w) > 2 and any(c.isalpha() for c in w)]
    
    # Score basato su diverse metriche
    char_ratio = alphanumeric_chars / total_chars
    word_ratio = len(valid_words) / max(len(words), 1)
    
    # Combina i punteggi
    estimated_score = (char_ratio * 0.6 + word_ratio * 0.4) * 100
    
    return round(min(estimated_score, 100), 2)

def clean_text(text):
    """Ottimizzato: usa compile per regex performance"""
    import re
    # Compile regex una volta sola per performance
    if not hasattr(clean_text, '_regex'):
        clean_text._regex = re.compile(r"^[e®=]", flags=re.MULTILINE)
    
    return clean_text._regex.sub("- ", text)

# FUNZIONI DI UTILITÀ PER SKIP OPERAZIONI COSTOSE

def should_skip_rotation_detection(img_page: Image.Image) -> bool:
    """
    Euristica per determinare se saltare la detection di rotazione.
    """
    width, height = img_page.size
    
    # Se l'immagine è molto piccola, probabilmente non ha bisogno di rotation detection
    if width < 500 or height < 500:
        return True
    
    # Se l'aspect ratio è molto diverso da quello tipico di un documento, skip
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return True
    
    return False

def should_use_fast_preprocessing(img_page: Image.Image) -> bool:
    """
    Determina se usare preprocessing veloce basato sulle caratteristiche dell'immagine.
    """
    width, height = img_page.size
    
    # Per immagini piccole, usa preprocessing veloce
    if width * height < 500000:  # < 0.5 megapixel
        return True
    
    # Se l'immagine è già in modalità L (grayscale), usa fast
    if img_page.mode == 'L':
        return True
        
    return False