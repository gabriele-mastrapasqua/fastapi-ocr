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

def pdf_to_images(contents, base_64 = False):
    """
    Convert a PDF file bytes content to a list of images.
    
    @contents: PDF file bytes content.
    @base_64: If True, return images as base64 strings, otherwise as PIL

    Returns: list of images (PIL Image or base64 strings). + num pages in PDF
    """
    tmp_file_path = None
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name
        
    images = []
    with pdfplumber.open(tmp_file_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            # Convert page to image
            # Use resolution=300 for better quality
            # You can adjust the resolution as needed
            image = page.to_image(resolution=300).original
            if base_64:
                # Convert PIL Image to base64 string
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append(img_str)
            else:
                # Append PIL Image directly
                images.append(image)
            
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


def detect_angle_rotation_tesseract(preproc_img: Image.Image) -> int:
    """
    Detect angle retation using Tesseract OCR.
    Returns the float angle value normalized to 0-360 degrees. 0 if no rotation is detected.
    """
    # Use Tesseract to find orientation
    osd = pytesseract.image_to_osd(preproc_img)
    print(f"OSD Output: {osd}\n")

    # Estrai angolo di rotazione
    rotation_line = [line for line in osd.split('\n') if "Orientation in degrees" in line]
    if rotation_line:
        angle = int(rotation_line[0].split(":")[1].strip())
        print(f"→ Rotazione rilevata: {angle}°")

        """
        Orientation in degrees	Rotazione del testo	        Azione da fare sull’immagine
        0	                    Corretta	                Nessuna
        90	                    Ruotata a destra	        Ruotare -90° (a sinistra)
        180	                    Capovolta	                Ruotare -180°
        270	                    Ruotata a sinistra	        Ruotare -90° (a destra)
        """

        # Correggi l'immagine se necessario
        needs_rotation = angle != 0
        if angle != 0:
            corrected_angle = - ((360 - angle) % 360)
            return corrected_angle, needs_rotation
        
    else:
        print("→ Nessun angolo di rotazione rilevato, immagine non modificata.")
        
    return 0, needs_rotation

