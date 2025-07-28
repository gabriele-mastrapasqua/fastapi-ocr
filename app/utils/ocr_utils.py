import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import paddleocr


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
