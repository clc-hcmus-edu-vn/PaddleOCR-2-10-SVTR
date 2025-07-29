from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time
from sklearn.cluster import DBSCAN
from sortedcontainers import SortedDict

DET_MODEL_DIR = 'models/det'
REC_MODEL_DIR = 'models/rec_nom'
REC_CHAR_DICT_PATH = 'ppocr/utils/dict/new_nom_dict.txt'
REC_IMAGE_SHAPE = '3,48,48'
REC_ALGORITHM = 'SVTR'
CONVERT_DICT_PATH = 'ppocr/utils/dict/combined_unique_chars.txt'

def cluster_columns(boxes, eps_w_multiplier=0.6, is_vertical=True):
    """
    Group character boxes into columns/rows using 1D clustering.

    Parameters:
        boxes: array-like of shape (N, 4) with [x_min, y_min, x_max, y_max]
        eps_w_multiplier: multiple of median char-width/height for DBSCAN eps
        is_vertical: True for vertical text (cluster by x), False for horizontal (cluster by y)

    Returns:
        labels: integer cluster label per box
    """
    boxes = np.asarray(boxes)
    
    if is_vertical:
        return _cluster_vertical(boxes, eps_w_multiplier)
    else:
        return _cluster_horizontal(boxes, eps_w_multiplier)

def _cluster_vertical(boxes, eps_w_multiplier):
    """Cluster boxes vertically (group by x-coordinates for vertical text)."""
    x_min, _, x_max, _ = boxes.T
    
    # Compute char widths and centroids
    widths = x_max - x_min
    W = np.median(widths)
    x_centroids = (x_min + x_max) / 2
    
    # 1D clustering via DBSCAN
    eps = eps_w_multiplier * W
    clustering = DBSCAN(eps=eps, min_samples=3, n_jobs=-1)
    raw_labels = clustering.fit_predict(x_centroids.reshape(-1, 1))
    
    # Sort clusters by mean-X descending (right-most first)
    return _sort_and_remap_clusters(raw_labels, x_centroids, reverse=True)

def _cluster_horizontal(boxes, eps_w_multiplier):
    """Cluster boxes horizontally (group by y-coordinates for horizontal text)."""
    _, y_min, _, y_max = boxes.T
    
    # Compute char heights and centroids
    heights = y_max - y_min
    H = np.median(heights)
    y_centroids = (y_min + y_max) / 2
    
    # 1D clustering via DBSCAN
    eps = eps_w_multiplier * H
    clustering = DBSCAN(eps=eps, min_samples=2, n_jobs=-1)
    raw_labels = clustering.fit_predict(y_centroids.reshape(-1, 1))
    
    # Sort clusters by mean-Y ascending (top-most first)
    return _sort_and_remap_clusters(raw_labels, y_centroids, reverse=False)

def _sort_and_remap_clusters(raw_labels, centroids, reverse=False):
    """Sort clusters by their centroid positions and remap labels."""
    # Get unique cluster labels (excluding noise = -1)
    unique = [lbl for lbl in np.unique(raw_labels) if lbl >= 0]
    
    # Compute mean centroid of each cluster
    mean_centroids = {lbl: centroids[raw_labels == lbl].mean() for lbl in unique}
    
    # Sort clusters by mean centroid
    sorted_clusters = sorted(unique, key=lambda lbl: mean_centroids[lbl], reverse=reverse)
    
    # Build remapping: old_label → new_label
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    
    # Apply remapping (noise stays -1)
    new_labels = np.array([remap[lbl] if lbl >= 0 else -1 for lbl in raw_labels])
    
    return new_labels

def group_text_by_clusters(text_results, labels, is_vertical=True):
    """Group text results by cluster labels and sort appropriately."""
    clustered_result = SortedDict()
    
    # Group by cluster labels
    for label, line in zip(labels, text_results):
        if label not in clustered_result:
            clustered_result[label] = []
        clustered_result[label].append(line)
    
    # Sort within each cluster
    for label in clustered_result:
        if is_vertical:
            # Sort by y position for vertical text
            clustered_result[label].sort(key=lambda x: x[0][1])
        else:
            # Sort by x position for horizontal text
            clustered_result[label].sort(key=lambda x: x[0][0])
    
    return clustered_result

def convert_ocr_result_format(ocr_result):
    """Convert OCR result to simplified format with bounding boxes."""
    altered_result = []
    for line in ocr_result[0]:
        coords = (line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1])
        altered_result.append([coords, line[1]])
    return altered_result

def sort_ocr_results(ocr_result, is_vertical=True):
    """
    Sort OCR results in reading order using clustering.
    
    Parameters:
        ocr_result: PaddleOCR result format
        is_vertical: True for vertical text layout, False for horizontal
        
    Returns:
        list: Sorted OCR results in reading order
    """
    if not ocr_result or not ocr_result[0]:
        return ocr_result
    
    # Convert result format
    altered_result = convert_ocr_result_format(ocr_result)
    
    # Cluster text regions
    labels = cluster_columns(
        np.array([line[0] for line in altered_result]), 
        is_vertical=is_vertical
    )
    
    # Group results by clusters
    clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
    
    # Flatten results in reading order
    sorted_results = []
    for cluster_label in sorted(clustered_result.keys()):
        sorted_results.extend(clustered_result[cluster_label])
    
    # Convert back to PaddleOCR format
    final_result = []
    for coords, (text, confidence) in sorted_results:
        # Convert back to 4-corner format
        x_min, y_min, x_max, y_max = coords
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        final_result.append([box, (text, confidence)])
    
    return [final_result]

def sort_ocr_end_results(ocr_result, is_vertical=True):
    """
    Sort OCR end results in reading order using clustering.
    
    Parameters:
        ocr_result: PaddleOCR result format
        is_vertical: True for vertical text layout, False for horizontal
        
    Returns:
        list: Sorted OCR results in reading order, grouped by clusters
    """
    if not ocr_result or not ocr_result[0]:
        return ocr_result
    
    final_result = []
    for line in ocr_result:
        text = ''.join([(item[1][0] + ' ') for item in line])
        confidence = np.mean([item[1][1] for item in line])
        x_min = min([item[0][0][0] for item in line])
        y_min = min([item[0][0][1] for item in line])
        x_max = max([item[0][2][0] for item in line])
        y_max = max([item[0][2][1] for item in line])
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        final_result.append([box, (text, confidence)])
    return [final_result]

def getDetectionOcr():
    """Create a PaddleOCR instance for detection only."""
    return PaddleOCR(
        det_model_dir=DET_MODEL_DIR,
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        det=True,
        rec=False,  # Detection only
        cls=False
    )

def getRecognitionOcr():
    """Create a PaddleOCR instance for recognition only."""
    return PaddleOCR(
        rec_model_dir=REC_MODEL_DIR,
        rec_char_dict_path=REC_CHAR_DICT_PATH,
        rec_image_shape=REC_IMAGE_SHAPE,
        rec_algorithm=REC_ALGORITHM,
        use_angle_cls=False,
        use_space_char=True,
        use_gpu=False,
        max_text_length=1,
        drop_score=0,
        show_log=False,
        det=False,  # Recognition only
        rec=True,
        cls=False
    )

class FastOcrProcessor:
    """Fast OCR processor that parallelizes text region recognition within a single image."""
    
    def __init__(self, rec_pool_size=4):
        self.detector = getDetectionOcr()
        self.rec_pool = [getRecognitionOcr() for _ in range(rec_pool_size)]
        self.rec_lock = threading.Lock()
    
    @contextmanager
    def acquire_recognizer(self):
        """Context manager to acquire and release recognition OCR instances safely."""
        recognizer = None
        try:
            with self.rec_lock:
                if self.rec_pool:
                    recognizer = self.rec_pool.pop()
            
            if recognizer is None:
                # If no recognizer available, create a temporary one
                recognizer = getRecognitionOcr()
                yield recognizer
            else:
                yield recognizer
        finally:
            if recognizer is not None:
                with self.rec_lock:
                    # Only return to pool if it was from the original pool
                    if len(self.rec_pool) < 6:  # Prevent unlimited growth
                        self.rec_pool.append(recognizer)
    
    def recognize_text_region(self, img, box):
        """Recognize text in a specific region using the recognition pool."""
        try:
            # Extract the text region from the image
            box = np.array(box).astype(np.int32)
            
            # Handle different box formats
            if box.ndim == 1:
                # If box is 1D, it might be [x1, y1, x2, y2] format
                if len(box) == 4:
                    x_min, y_min, x_max, y_max = box
                else:
                    print(f"Unexpected 1D box format with length {len(box)}")
                    return "", 0.0
            elif box.ndim == 2:
                # If box is 2D, it's in [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                x_min = max(0, int(np.min(box[:, 0])))
                y_min = max(0, int(np.min(box[:, 1])))
                x_max = min(img.shape[1], int(np.max(box[:, 0])))
                y_max = min(img.shape[0], int(np.max(box[:, 1])))
            else:
                print(f"Unexpected box dimensions: {box.ndim}")
                return "", 0.0
            
            # Ensure coordinates are within image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img.shape[1], int(x_max))
            y_max = min(img.shape[0], int(y_max))
            
            # Extract region
            text_region = img[y_min:y_max, x_min:x_max]
            
            if text_region.size == 0:
                return "", 0.0
            
            # Use recognition OCR from pool
            with self.acquire_recognizer() as recognizer:
                if recognizer is None:
                    return "", 0.0
                
                # Perform recognition on the text region
                rec_result = recognizer.ocr(text_region, det=False, cls=False)
                
                if rec_result and rec_result[0] and len(rec_result[0]) > 0:
                    text, confidence = rec_result[0][0]
                    return text, confidence
                else:
                    return "", 0.0
                    
        except Exception as e:
            print(f"Error in text region recognition: {e}")
            return "", 0.0
    
    def fast_ocr(self, img, max_workers=4, is_vertical=True):
        """Perform fast OCR on an image by parallelizing text region recognition."""
        try:
            detection_start = time.time()
            det_result = self.detector.ocr(img, det=True, cls=False, rec=False)
            detection_time = time.time() - detection_start
            
            if not det_result or not det_result[0]:
                print("No text regions detected")
                return []
            
            # Extract detected boxes from detection result
            detected_boxes = det_result[0]
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit recognition tasks for all detected regions
                futures = [
                    executor.submit(self.recognize_text_region, img, box) 
                    for box in detected_boxes
                ]
                
                # Collect results with progress bar
                recognition_results = []
                for future in futures:
                    recognition_results.append(future.result())
            
            # Step 3: Combine detection and recognition results
            combined_results = []
            for i, (box, (text, confidence)) in enumerate(zip(detected_boxes, recognition_results)):
                if text:  # Only include if text was recognized
                    combined_results.append([box, (text, confidence)])
            
            # Sort results in reading order
            if combined_results:
                # Convert to PaddleOCR format for sorting
                ocr_format_result = [combined_results]
                sorted_result = sort_ocr_results(ocr_format_result, is_vertical=is_vertical)
                return sorted_result[0] if sorted_result else []
            else:
                return []
            
        except Exception as e:
            print(f"Error in fast OCR processing: {e}")
            return []

# Initialize global fast OCR processor
fast_ocr_processor = FastOcrProcessor(rec_pool_size=4)

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def load_vietnamese_font(font_size=20):
    try:
        font = ImageFont.truetype("arial", font_size)
        return font
    except Exception:
        print("Arial font not found, using default font")
        return ImageFont.load_default()

def visualize_results(image, result, output_path='visualized_output.jpg'):
    # Load the Vietnamese dictionary
    try:
        with open(CONVERT_DICT_PATH, 'r', encoding='utf-8') as f:
            nom_dict = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: {CONVERT_DICT_PATH} not found!")
        return
    
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Load Vietnamese-compatible font
    font = load_vietnamese_font(font_size=20)
    
    # Check if result is the new grouped structure
    if result is None:
        print("No OCR results to visualize")
        return
    
     # Draw detection boxes and recognition results
    for idx, line in enumerate(result):
        # Extract box coordinates
        boxes = line[0]
        box = np.array(boxes).astype(np.int32).reshape(-1, 2)
        
        # Draw polygon around text area (convert points to tuple for PIL)
        points = [(point[0], point[1]) for point in box]
        draw.line(points + [points[0]], fill=(0, 255, 0), width=2)
        
        # Get text and confidence (tuple format)
        text, _ = line[1]
        print(f"Line {idx}: {text}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the visualization
    pil_image.save(output_path)
    print(f"Visualization saved to {output_path}")

def process_image(image_path, output_path=None, max_workers=4, is_vertical=True):
    """Process a single image using fast parallel OCR.
    
    Args:
        image_path (str): Path to the image file
        output_path (str, optional): Path for visualization output
        max_workers (int): Number of parallel workers for recognition
        is_vertical (bool): True for vertical text layout, False for horizontal
    
    Returns:
        list: OCR results in PaddleOCR format [[[box, (text, confidence)]]]
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_name = os.path.basename(image_path)
        
        # Use fast OCR processor with sorting
        result = fast_ocr_processor.fast_ocr(img, max_workers=max_workers, is_vertical=is_vertical)        
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = 'output'
            output_filename = f"visualized_{os.path.splitext(image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
        
        # Load the Vietnamese dictionary for label conversion
        try:
            with open(CONVERT_DICT_PATH, 'r', encoding='utf-8') as f:
                nom_dict = f.read().splitlines()
        except FileNotFoundError:
            print(f"Warning: {CONVERT_DICT_PATH} not found! Returning raw results.")
            nom_dict = None
        
        # Convert characters to Vietnamese labels while maintaining PaddleOCR format
        if result and nom_dict:
            converted_result = []
            for item in result:
                box, (text, confidence) = item
                
                # Convert character to Vietnamese text
                try:
                    char_index = char2code(text)
                    if 0 <= char_index < len(nom_dict):
                        viet_text = nom_dict[char_index]
                    else:
                        viet_text = f"[Unknown char: {text}]"
                except Exception as e:
                    viet_text = f"[Error: {text}]"
                
                # Maintain original PaddleOCR format: [box, (text, confidence)]
                converted_item = [box, (text, confidence)]
                converted_result.append(converted_item)
            
            num_regions = len(converted_result)
            print(f"Successfully processed {num_regions} text regions with Vietnamese labels")
            
            # Return in PaddleOCR format: [[[results]]]
            return [converted_result]
        elif result:
            # Return raw result if dictionary not available
            num_regions = len(result)
            print(f"Successfully processed {num_regions} text regions (no label conversion)")
            # Return in PaddleOCR format: [[[results]]]
            return [result]
        else:
            print(f"No text found in {image_name}")
            # Return empty result in PaddleOCR format
            return [[]]
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_image_with_sentences(image_path, output_path=None, max_workers=4, is_vertical=True, min_words=2):
    """Process a single image using fast parallel OCR and extract grouped sentences.
    
    Args:
        image_path (str): Path to the image file
        output_path (str, optional): Path for visualization output
        max_workers (int): Number of parallel workers for recognition
        is_vertical (bool): True for vertical text layout, False for horizontal
        min_words (int): Minimum number of words required for a valid sentence (default: 2)
    
    Returns:
        dict: Dictionary containing OCR results, grouped sentences, and metadata
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_name = os.path.basename(image_path)
        
        # Use fast OCR processor with sorting
        result = fast_ocr_processor.fast_ocr(img, max_workers=max_workers, is_vertical=is_vertical)        
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = 'output'
            output_filename = f"visualized_{os.path.splitext(image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
        
        # Load the Vietnamese dictionary for label conversion
        try:
            with open(CONVERT_DICT_PATH, 'r', encoding='utf-8') as f:
                nom_dict = f.read().splitlines()
        except FileNotFoundError:
            print(f"Warning: {CONVERT_DICT_PATH} not found! Returning raw results.")
            nom_dict = None
        
        # Extract grouped sentences from OCR results
        grouped_sentences = []
        if result and nom_dict:
            grouped_sentences = extract_grouped_sentences_from_ocr_result([result], nom_dict, is_vertical, min_words=min_words)
        
        # Convert characters to Vietnamese labels and group by clusters
        if result and nom_dict:
            # Convert result format for clustering
            altered_result = convert_ocr_result_format([result])
            
            # Cluster text regions
            labels = cluster_columns(
                np.array([line[0] for line in altered_result]), 
                is_vertical=is_vertical
            )
            
            # Group results by clusters
            clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
            
            # Convert to grouped OCR results
            grouped_ocr_results = []
            for cluster_label in sorted(clustered_result.keys()):
                cluster_items = clustered_result[cluster_label]
                cluster_result = []
                
                for item in cluster_items:
                    coords, (text, confidence) = item
                    
                    # Convert character to Vietnamese text
                    try:
                        char_index = char2code(text)
                        if 0 <= char_index < len(nom_dict):
                            viet_text = nom_dict[char_index]
                        else:
                            viet_text = f"[Unknown char: {text}]"
                    except Exception as e:
                        viet_text = f"[Error: {text}]"
                    
                    # Convert back to 4-corner format
                    x_min, y_min, x_max, y_max = coords
                    box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    
                    # Add to cluster result
                    cluster_result.append([box, (viet_text, confidence)])
                
                grouped_ocr_results.append(cluster_result)
            
            num_regions = sum(len(cluster) for cluster in grouped_ocr_results)
            
            # Return dictionary with grouped OCR results and grouped sentences
            return grouped_ocr_results
        elif result:
            # Return raw result if dictionary not available
            num_regions = len(result)
            print(f"Successfully processed {num_regions} text regions (no label conversion)")
            
            # Group raw results by clusters
            altered_result = convert_ocr_result_format([result])
            labels = cluster_columns(
                np.array([line[0] for line in altered_result]), 
                is_vertical=is_vertical
            )
            clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
            
            # Convert to grouped OCR results
            grouped_ocr_results = []
            for cluster_label in sorted(clustered_result.keys()):
                cluster_items = clustered_result[cluster_label]
                cluster_result = []
                
                for item in cluster_items:
                    coords, (text, confidence) = item
                    # Convert back to 4-corner format
                    x_min, y_min, x_max, y_max = coords
                    box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    cluster_result.append([box, (text, confidence)])
                
                grouped_ocr_results.append(cluster_result)
            
            return grouped_ocr_results
        else:
            print(f"No text found in {image_name}")
            # Return empty result in PaddleOCR format
            return None
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_batch_images(image_paths, max_workers=4, rec_workers=4, is_vertical=True):
    """Process multiple images using parallel processing.
    
    Args:
        image_paths (list): List of image file paths
        max_workers (int): Maximum number of image processing threads
        rec_workers (int): Number of recognition workers per image
        is_vertical (bool): True for vertical text layout, False for horizontal
    
    Returns:
        list: List of processing results for each image
    """
    results = []
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_image, img_path, max_workers=rec_workers, is_vertical=is_vertical) 
            for img_path in image_paths
        ]
        
        # Collect results with progress bar
        for future in futures:
            results.append(future.result())
    
    # Print summary
    successful = sum(1 for r in results if r is not None)
    failed = len(results) - successful
    total_regions = sum(len(r[0]) for r in results if r is not None and len(r) > 0)
    
    print(f"\nBatch processing complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Total text regions found: {total_regions}")
    if successful > 0:
        print(f"Average regions per image: {total_regions/successful:.1f}")
    
    return results

def extract_sentences_from_clustered_results(clustered_result, nom_dict, min_words=2):
    """
    Extract sentences from clustered OCR results in reading order.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        nom_dict: Vietnamese dictionary for text conversion
        min_words: Minimum number of words required for a valid sentence (default: 2)
        
    Returns:
        list: List of sentences in reading order (filtered by minimum word count)
    """
    sentences = []
    
    # Process clusters in order (columns/rows)
    sorted_clusters = sorted(clustered_result.keys())
    
    for cluster_label in sorted_clusters:
        cluster_lines = clustered_result[cluster_label]
        
        # Extract words from this cluster/column
        cluster_words = []
        for line in cluster_lines:
            coordinates = line[0]
            text, confidence = line[1]
            
            # Convert to Vietnamese text
            vietnamese_text = get_vietnamese_text(text, nom_dict)
            cluster_words.append(vietnamese_text)
        
        # Join words in this cluster to form a sentence
        if cluster_words:
            sentence = ' '.join(cluster_words)
            # Only include sentences with minimum word count
            if len(cluster_words) >= min_words:
                sentences.append(sentence)
            else:
                print(f"Filtered out single-word sentence: '{sentence}' (cluster {cluster_label})")
    
    return sentences

def get_vietnamese_text(text, nom_dict):
    """Convert character to Vietnamese text using the dictionary."""
    try:
        return nom_dict[char2code(text)]
    except (IndexError, ValueError):
        return text  # Return original text if conversion fails

def extract_sentences_from_ocr_result(ocr_result, nom_dict, is_vertical=True, min_words=2):
    """
    Extract sentences from OCR results by clustering and grouping.
    
    Parameters:
        ocr_result: PaddleOCR result format
        nom_dict: Vietnamese dictionary for text conversion
        is_vertical: True for vertical text layout, False for horizontal
        min_words: Minimum number of words required for a valid sentence (default: 2)
        
    Returns:
        list: List of sentences in reading order (filtered by minimum word count)
    """
    if not ocr_result or not ocr_result[0]:
        return []
    
    # Convert result format
    altered_result = convert_ocr_result_format(ocr_result)
    
    # Cluster text regions
    labels = cluster_columns(
        np.array([line[0] for line in altered_result]), 
        is_vertical=is_vertical
    )
    
    # Group results by clusters
    clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
    
    # Extract sentences from clustered results
    sentences = extract_sentences_from_clustered_results(clustered_result, nom_dict, min_words=min_words)
    
    return sentences

def extract_grouped_sentences_from_ocr_result(ocr_result, nom_dict, is_vertical=True, min_words=2):
    """
    Extract grouped sentences from OCR results by clustering and grouping.
    Each group represents a column/row of text.
    
    Parameters:
        ocr_result: PaddleOCR result format
        nom_dict: Vietnamese dictionary for text conversion
        is_vertical: True for vertical text layout, False for horizontal
        min_words: Minimum number of words required for a valid sentence (default: 2)
        
    Returns:
        list: List of sentence groups, where each group is a list of sentences from the same cluster
    """
    if not ocr_result or not ocr_result[0]:
        return []
    
    # Convert result format
    altered_result = convert_ocr_result_format(ocr_result)
    
    # Cluster text regions
    labels = cluster_columns(
        np.array([line[0] for line in altered_result]), 
        is_vertical=is_vertical
    )
    
    # Group results by clusters
    clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
    
    # Extract grouped sentences from clustered results
    grouped_sentences = extract_grouped_sentences_from_clustered_results(clustered_result, nom_dict, min_words=min_words)
    
    return grouped_sentences

def extract_grouped_sentences_from_clustered_results(clustered_result, nom_dict, min_words=2):
    """
    Extract grouped sentences from clustered OCR results in reading order.
    Each group represents a column/row of text.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        nom_dict: Vietnamese dictionary for text conversion
        min_words: Minimum number of words required for a valid sentence (default: 2)
        
    Returns:
        list: List of sentence groups, where each group is a list of sentences from the same cluster
    """
    grouped_sentences = []
    
    # Process clusters in order (columns/rows)
    sorted_clusters = sorted(clustered_result.keys())
    
    for cluster_label in sorted_clusters:
        cluster_lines = clustered_result[cluster_label]
        
        # Extract words from this cluster/column
        cluster_words = []
        for line in cluster_lines:
            coordinates = line[0]
            text, confidence = line[1]
            
            # Convert to Vietnamese text
            vietnamese_text = get_vietnamese_text(text, nom_dict)
            cluster_words.append(vietnamese_text)
        
        # Only include sentences with minimum word count
        if len(cluster_words) >= min_words:
            # Add the cluster words as a group (list of words)
            grouped_sentences.append(cluster_words)
        else:
            print(f"Filtered out single-word sentence: '{' '.join(cluster_words)}' (cluster {cluster_label})")
    
    return grouped_sentences

if __name__ == "__main__":
    image_path = 'test_images/image.png'
    result = process_image_with_sentences(image_path, max_workers=4, is_vertical=True)
    img = cv2.imread(image_path)
    result = sort_ocr_end_results(result, is_vertical=True)
    visualize_results(img, result[0], output_path='test_images/nom_visualized.png')