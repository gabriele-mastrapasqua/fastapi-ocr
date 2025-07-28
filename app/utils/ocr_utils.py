
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
