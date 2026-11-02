#------------------------------------IMPORT-----------------------------------
import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from pathlib import Path
from collections import defaultdict

# Set seeds for reproducibility (Crucial for consistent data augmentation)
SEED = 1225
random.seed(SEED)
np.random.seed(SEED)

# Import PyTorch dependencies
import torch
torch.manual_seed(SEED)

# Import torchvision dependencies
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from IPython.display import display

pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img

    
def get_bottom_corner(mask, side="left", offset_x=0, tol=5):
    """
    Extracts the bottom corner point (left or right) of the mask, using the main contour 
    and looking for the points with almost maximum y (within tol).
    """
    # Extract contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Assume the largest contour is the one of interest
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2)
    
    # Determine the maximum y coordinate (bottom part)
    max_y = np.max(pts[:, 1])
    # Take points whose y coordinate is within a tolerance of max_y
    bottom_pts = pts[np.abs(pts[:, 1] - max_y) <= tol]
    if len(bottom_pts) == 0:
        bottom_pts = pts[pts[:, 1] == max_y]
    
    if side == "left":
        # For the left block, we choose the one with the smallest x
        pt = bottom_pts[np.argmin(bottom_pts[:, 0])]
    else:
        # For the right block, we choose the one with the largest x
        pt = bottom_pts[np.argmax(bottom_pts[:, 0])]
    
    return (pt[0] + offset_x, pt[1])


def get_top_corner(mask, side="left", offset_x=0, tol=5):
    """
    Extracts the top corner point (left or right) of the mask, using the main contour 
    and looking for the points with almost minimum y (within tol).
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2)
    
    # Determine the minimum y coordinate (top part)
    min_y = np.min(pts[:, 1])
    top_pts = pts[np.abs(pts[:, 1] - min_y) <= tol]
    if len(top_pts) == 0:
        top_pts = pts[pts[:, 1] == min_y]
    
    if side == "left":
        # For the left block, we choose the one with the smallest x (top left corner)
        pt = top_pts[np.argmin(top_pts[:, 0])]
    else:
        # For the right block, we choose the one with the largest x (top right corner)
        pt = top_pts[np.argmax(top_pts[:, 0])]
    
    return (pt[0] + offset_x, pt[1])

def classify_mask_from_array_improved3(img, top_percentage=0.3):
    """
    Classifies the implant as 'U Type' or 'Straight Type' based on the top part (top_percentage).
    Returns:
        tipo,
        drop1_orig, drop2_orig, peak1_orig, peak2_orig

    - If 'U Type', returns the 4 points.
    - If 'Straight Type', returns None for all points.
    """
        
    def find_top_row(section, col_index):
        """
        Returns the first row where there is a white pixel in the column col_index
        within the sub-image 'section'.
        If not found, uses the middle as a fallback.
        """
        rows = np.where(section[:, col_index] == 255)[0]
        if rows.size > 0:
            return rows[0]
        else:
            return section.shape[0] // 2


    def find_top_row_local(section, col_index, window=3):
        """
        Searches, in a horizontal neighborhood [col_index - window, col_index + window],
        the first white row of each column, and returns the LOWEST row among all.
        """
        h, w = section.shape
        cmin = max(0, col_index - window)
        cmax = min(w - 1, col_index + window)
        
        top_rows = []
        for c in range(cmin, cmax + 1):
            rows = np.where(section[:, c] == 255)[0]
            if rows.size > 0:
                top_rows.append(rows[0])
        
        if not top_rows:
            return h // 2
        
        return max(top_rows)


    # Ensure the image is in grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contour 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No se detectó un implante en la imagen."

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.reshape(-1, 2)  # (N,2)

    
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate height to crop (top part)
    top_height = int(h * top_percentage)
    top_cutoff = y + top_height  # Global Y coordinate that delimits the top part

    # Take only the contour points that are above top_cutoff
    top_contour_points = contour_points[contour_points[:, 1] < top_cutoff]

    if len(top_contour_points) == 0:
        return "La parte superior del implante no tiene contorno suficiente."

    # New bounding box ONLY for the top part
    x2, y2, w2, h2 = cv2.boundingRect(top_contour_points)

    # Crop the exact top sub-image
    top_section = binary[y2 : y2 + h2, x2 : x2 + w2]

    # Calculate vertical profile of that sub-image
    vertical_profile = np.sum(top_section == 255, axis=0)

    # --- Peak detection in the profile ---
    peaks, _ = find_peaks(vertical_profile, height=np.max(vertical_profile) * 0.5)
    if len(peaks) < 2:
        return ("No se detectaron suficientes picos para clasificar como tipo U.", None, None, None, None)

    # Sort so that peak1 < peak2
    peaks.sort()
    peak1, peak2 = peaks[0], peaks[1]

    # --- Calculate the derivative and find drops ---
    deriv = np.diff(vertical_profile)

    
    # Adjustment: offset_local scaled based on the distance (peak2-peak1)
    distance_peaks = peak2 - peak1
    # We want an offset that is <= 7, but no more than half the distance
    offset_local = min(7, max(1, distance_peaks // 2))

    # Calculate drop1
    start_index = peak1 + offset_local
    if start_index >= peak2:
        drop1 = peak1 + 1
    else:
        region_deriv = deriv[start_index:peak2]
        local_index = np.argmin(region_deriv)
        drop1 = start_index + local_index + 1

    # Calculate drop2
    right_half_start = len(vertical_profile) // 2
    
    # Adjustment: if peak2 is close to the center, reduce to 0.67
    # To avoid running out of range in very small images
    
    relative_factor = 0.67
    # If (peak2 < right_half_start), adjust
    if peak2 < right_half_start:
        # (rare case) => put offset_start close to peak2
        offset_start = max(peak1, peak2 - offset_local)
    else:
        offset_start = right_half_start + int(relative_factor * (peak2 - right_half_start))
        # Clamp to avoid going out of bounds
        offset_start = max(peak1, min(offset_start, peak2 - 1))

    drop2 = peak2
    for i in range(offset_start + 1, peak2 + 1):
        if vertical_profile[i] > vertical_profile[i - 1]:
            drop2 = i - 1
            break

    # Ensure that drop1 < drop2
    if drop1 > drop2:
        drop1, drop2 = drop2, drop1

    # --- Classify as 'U Type' or 'Straight Type' ---
    mid_index = len(vertical_profile) // 2
    left_extreme = np.mean(vertical_profile[: mid_index // 2])
    right_extreme = np.mean(vertical_profile[-(mid_index // 2) :])
    center_val = np.mean(vertical_profile[mid_index - 5 : mid_index + 5])
    if center_val < min(left_extreme, right_extreme) * 0.6:
        tipo = "Tipo U"
    else:
        tipo = "Tipo Recto"

    # If it is "Straight Type", we do not extract anything
    if tipo == "Tipo Recto":
        return (tipo, None, None, None, None)

   
    # If it is "U Type", we extract drops and peaks
    
    # Make the drops more or less aligned
    center_x = (peak1 + peak2) / 2.0
    dist1 = center_x - drop1
    dist2 = drop2 - center_x

    tolerance = 5  # pixels of tolerance
    if abs(abs(dist1) - abs(dist2)) > tolerance:
        avg_dist = (abs(dist1) + abs(dist2)) / 2.0
        drop1 = int(center_x - avg_dist)
        drop2 = int(center_x + avg_dist)

    # Ensure we don't go out of range
    drop1 = max(0, min(drop1, len(vertical_profile) - 1))
    drop2 = max(0, min(drop2, len(vertical_profile) - 1))

    # Locate the drops (lowest white row in a horizontal neighborhood)
    drop1_y = find_top_row_local(top_section, drop1, window=3)
    drop2_y = find_top_row_local(top_section, drop2, window=3)
    drop1_orig = (x2 + drop1, y2 + drop1_y)
    drop2_orig = (x2 + drop2, y2 + drop2_y)

    # Peaks (direct top row)
    peak1_orig = (x2 + peak1, y2 + find_top_row(top_section, peak1))
    peak2_orig = (x2 + peak2, y2 + find_top_row(top_section, peak2))

    return (tipo, drop1_orig, drop2_orig, peak1_orig, peak2_orig)


def enderezar_mascara_recto_regresion_lineal(binary_mask, shape_label, debug=False, ignore_top_rows=25):
    """
    Takes a binary mask of a Straight implant and:
      1) Extracts the 'top line' (first white pixel for each column).
      2) Fits y = m*x + b (linear regression) minimizing vertical distances.
      3) Calculates angle = -arctan(m) and rotates the mask to make that top line horizontal.

    Returns:
      - rotated_mask: the rotated mask
      - final_angle: the angle (in degrees) of the applied rotation
    """
    # Ensure it is binary (0/255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    h, w = binary_mask.shape
    top_points_x = []
    top_points_y = []

    # Extract top points
    # Extract the first white pixel (the top line) for each column,
    # but ignore those that are above 'ignore_top_rows'.
    for col in range(w):
        rows = np.where(binary_mask[:, col] == 255)[0]
        if rows.size > 0:
            y_top = rows[0]  # first white row in column 'col'
            if y_top <= ignore_top_rows:
                top_points_x.append(col)
                top_points_y.append(y_top)
    

    if len(top_points_x) < 2:
        # print("No hay suficientes puntos para la regresión lineal (menos de 2 columnas con parte superior).")
        return binary_mask, 0.0

    # Linear fit y = m*x + b using np.polyfit
    top_points_x = np.array(top_points_x, dtype=np.float32)
    top_points_y = np.array(top_points_y, dtype=np.float32)

    m, b = np.polyfit(top_points_x, top_points_y, 1)  # deg=1 => line

    # Angle with the horizontal = arctan(m)
    angle_radians = np.arctan(m)
    angle_degrees = np.degrees(angle_radians)
    final_angle = angle_degrees  # we rotate in the opposite direction

    
    if debug:
        # Calculate two extremes of the line based on x = 0 and x = w
        x_line = np.array([0, w])
        y_line = m * x_line + b

    # Rotate the image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    rotated_mask = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST)

    return rotated_mask, final_angle

def get_keypoints(block, is_left=True, offset_x=0):
    contornos, _ = cv2.findContours(block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contornos) == 0:
        return [], []
    c = max(contornos, key=cv2.contourArea)

    puntos = [tuple(pt[0]) for pt in c]
    puntos.sort(key=lambda p: p[1])

    y_dict = defaultdict(list)
    for (x, y) in puntos:
        y_dict[y].append(x)

    f = []
    for y in sorted(y_dict.keys()):
        x_extreme = min(y_dict[y]) if is_left else max(y_dict[y])
        f.append((y, x_extreme + offset_x))

    ys = np.array([p[0] for p in f])
    xs = np.array([p[1] for p in f])


    # Adjust distance based on image size
    img_width = block.shape[1]  

    if img_width < 40:
        peak_distance = 5
    else:
        peak_distance = 15

    # Detect peaks and valleys with adjusted distance
    peaks, _ = find_peaks(xs, distance=peak_distance)
    valleys, _ = find_peaks(-xs, distance=peak_distance)

    # Swap for the left side
    if is_left:
        peaks, valleys = valleys, peaks

    peaks_pts = [(xs[i], ys[i]) for i in peaks]
    valleys_pts = [(xs[i], ys[i]) for i in valleys]

    return peaks_pts, valleys_pts


# Function that gets the keypoints and calculates the regression only with the valleys
def get_keypoints_with_regression(peaks_pts, valleys_pts):
    
    def calculate_regression_params(xs, ys):
        xs_arr = np.array(xs).reshape(-1, 1)  # Format (n_samples, 1)
        ys_arr = np.array(ys)
        model = LinearRegression()
        model.fit(xs_arr, ys_arr)
        slope = model.coef_[0]
        intercept = model.intercept_
        angle = np.degrees(np.arctan(slope))
        return slope, intercept, angle

    # Extract coordinates of the valleys
    xs_valleys = [p[0] for p in valleys_pts]
    ys_valleys = [p[1] for p in valleys_pts]
    
    # Verify that the arrays are not empty
    if not xs_valleys or not ys_valleys:  
        # print("Advertencia: No hay datos suficientes para regresión. Saltando esta imagen.")
        return None  
    
    # Convert to NumPy arrays after verification
    xs_valleys = np.array(xs_valleys)
    ys_valleys = np.array(ys_valleys)

    # Sort the points by the x coordinate to ensure correct line tracing
    indices_orden = np.argsort(xs_valleys)
    xs_valleys = np.array(xs_valleys)[indices_orden]
    ys_valleys = np.array(ys_valleys)[indices_orden]
    
    # Calculate regression parameters
    slope, intercept, angle = calculate_regression_params(xs_valleys, ys_valleys)
    
    return peaks_pts, valleys_pts, slope, intercept, angle, xs_valleys, ys_valleys


def corregir_x_a_linea_regresion(pt, slope, intercept, mask):
    """
    Corrects the x coordinate of a point (x, y) using the regression line:
        y = slope * x + intercept
    The original row (y) is kept if the pixel at (x_corr, y) is white.
    If it is not, it searches in the x_corr column for the white pixel with the largest y.
    
    Args:
        pt: original (x, y) tuple.
        slope: slope of the line.
        intercept: y-intercept of the line.
        mask: binary mask (white pixels are > 0).
    
    Returns:
        (x_corr, y_corr): the corrected point.
    """
    x, y = pt
    # Avoid division by zero if the slope is very small
    if abs(slope) < 1e-8:
        return pt
    # Calculate the corrected column from the equation:
    # y = slope*x + intercept  =>  x_corr = (y - intercept) / slope
    x_corr = (y - intercept) / slope
    x_corr_int = int(round(x_corr))
    
    # Ensure that x_corr_int is within the mask boundaries
    x_corr_int = max(0, min(x_corr_int, mask.shape[1]-1))
    
    # If at position (x_corr_int, y) it is already in a white zone, we use that position.
    if mask[y, x_corr_int] > 0:
        return (x_corr_int, y)
    else:
        # In that column, we extract all rows where the mask is white.
        columna = mask[:, x_corr_int]
        indices_blancos = np.where(columna > 0)[0]
        if len(indices_blancos) > 0:
            # We choose the lowest row (largest y) of that column
            y_corr = int(np.max(indices_blancos))
            return (x_corr_int, y_corr)
        else:
            # If there is no white pixel in that column, the original point is returned
            return pt
                        

def clip_line(x1, y1, x2, y2, width, height):
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2

def transform_points_back(points, center, angle_degrees, final_angle, position):
    def rotate_points_180(points, center):
        # The 180° rotation is symmetric: (x, y) -> (2*center_x - x, 2*center_y - y)
        return [(int(2 * center[0] - x), int(2 * center[1] - y)) for (x, y) in points]

    # Add the angles
    total_angle = final_angle + angle_degrees
    
    # If the image is Top (Superior), rotate 180°.
    if position == "Superior":
        points = rotate_points_180(points, center)
    
    # Create the rotation matrix (with -total_angle)
    M = cv2.getRotationMatrix2D(center, -total_angle, 1.0)
    transformed = []
    for pt in points:
        # Verify that pt can be unpacked into (x, y)
        try:
            x, y = pt
        except Exception as e:
            # print("Advertencia: punto mal formado, se ignora:", pt)
            continue
        pt_homog = np.array([x, y, 1])
        new_pt = M.dot(pt_homog)
        transformed.append((int(new_pt[0]), int(new_pt[1])))
    return transformed

def rotate_image_around_point(img, angle, center, fillcolor=(0, 0, 0)):
    """
    Rotates an image around a specific point (center).
    """
    # Translate image so that the desired center is at the center of the image
    cx, cy = int(center[0]), int(center[1])
    translate_to_center = Image.new("RGB", img.size, fillcolor)
    translate_to_center.paste(img, (-cx + img.width // 2, -cy + img.height // 2))

    # Rotate around the center of the image
    rotated = translate_to_center.rotate(angle, expand=True, fillcolor=fillcolor)

    # Crop around the new center
    new_center = (rotated.width // 2, rotated.height // 2)
    return rotated, new_center

def process_implant(file_id, sample_img, annotation_df, labels, base_dir: Path):
    """
    Processes the image (identified by file_id) using the annotations in annotation_df,
    and returns a list of dictionaries with the extracted features.
    It is assumed that sample_img is a PIL instance of the original image.
    """
    #-----------------------CROP IMAGE WITH BOUNDING BOX-----------------------------------
    features_list = []
    
    # Get bboxes and segmentation of the selected image
    bboxes = annotation_df.loc[file_id]['bboxes']
    polygon_points = annotation_df.loc[file_id]['segmentation']
    diameter = annotation_df.loc[file_id]['diameter']

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # Only process if the label is "ACTIVE", MKIII, MKIV, etc.
        if label not in ["ACTIVE", "MKIII","MKIV", "NOBEL SPEEDY", "PARALLEL"]:
            continue

        current_sample_img = sample_img.copy()

        # bbox: [x, y, w, h]
        x_min, y_min, w, h = bbox
        x_max, y_max = x_min + w, y_min + h

        center_bbox = (x_min + w // 2, y_min + h // 2)

        # Add margin to the bbox
        margin = 3
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(current_sample_img.width, x_max + margin)
        y_max = min(current_sample_img.height, y_max + margin)

        # check if right isn't less than ledt
        if x_max <= x_min or y_max <= y_min:
            # print(f"Coordinates of the bbox are invalid for file_id {file_id}, skipping this bbox.")
            continue

        # Crop the image with PIL
        cropped_img = current_sample_img.crop((x_min, y_min, x_max, y_max))

        orig_width_bbox = cropped_img.width - 2 * margin
        orig_height_bbox = cropped_img.height - 2 * margin
        implant_bbox_ratio = orig_width_bbox / orig_height_bbox

        # Adjust the polygons to the cropped image
        adjusted_polygons = []
        for poly in polygon_points:
            if not poly or len(poly) == 0 or not poly[0]:
                continue
            # Extract the list of numbers and group into (x, y) pairs
            points = list(zip(poly[0][::2], poly[0][1::2]))
            # Adjust each point by subtracting (x_min, y_min)
            adjusted_points = [(x - x_min, y - y_min) for x, y in points]
            adjusted_polygons.append(adjusted_points)

        # Convert cropped image to tensor (format C x H x W)
        cropped_tensor = F.to_tensor(cropped_img)

        # Generate masks from the adjusted polygons
        mask_tensors = []
        for poly in adjusted_polygons:
            mask_img = create_polygon_mask(cropped_img.size, poly)
            mask_tensor = F.pil_to_tensor(mask_img).bool()  # Tensor with shape (1, H, W)
            mask_tensors.append(mask_tensor)
        
        if mask_tensors:
            # Group the masks into a tensor (N, 1, H, W)
            masks = torch.stack(mask_tensors)
            # draw_segmentation_masks expects (N, H, W); we remove the channel dimension
            masks = masks[:, 0]
            
            # Create a list of colors for each mask
            color_palette = ["red", "blue", "green", "yellow", "purple", "orange"]
            colors = [color_palette[j % len(color_palette)] for j in range(masks.shape[0])]
            
            # Draw all masks on the cropped image
            annotated_tensor = draw_segmentation_masks(cropped_tensor, masks, alpha=0.5, colors=colors)
            annotated_img = F.to_pil_image(annotated_tensor)

            # Visualize each mask individually
            for j, poly in enumerate(adjusted_polygons):
                mask_img = create_polygon_mask(cropped_img.size, poly)
                mask_img = mask_img.convert("L")  
                mask_np = np.array(mask_img)
                if mask_np.max() == 1:
                    mask_np = mask_np * 255
                # Apply threshold to binarize
                _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

                if np.count_nonzero(binary_mask) == 0:
                    continue

                #-------------------------------STRAIGHTEN THE MASK---------------------------

                # Generate SKELETON
                skeleton = cv2.ximgproc.thinning(binary_mask)

                # Find coordinates of pixels where skeleton has a value of 255
                ys, xs = np.where(skeleton == 255)
                # Combine into a coordinate matrix (each row is [x, y])
                points = np.column_stack((xs, ys))

                # Ensure there are enough points
                if len(points) > 0:
                    mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))

                    # First eigenvector == principal direction.
                    principal_axis = eigenvectors[0]  
                    
                    # Center
                    center = mean[0]  
                    
                    # To visualize the axis, define two points along the axis
                    scale = 100  
                    pt1 = (int(center[0] - scale * principal_axis[0]), int(center[1] - scale * principal_axis[1]))
                    pt2 = (int(center[0] + scale * principal_axis[0]), int(center[1] + scale * principal_axis[1]))
                else:
                    # print("No points found in the skeleton")
                    # If there is no skeleton, we exit or rotate 0
                    pt1, pt2 = (0,0), (0,0)

                # Draw the principal axis line on the skeleton
                skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                cv2.line(skeleton_color, pt1, pt2, (0, 0, 255), thickness=2)

                # Get coordinates of the white points of the skeleton
                y_skel, x_skel = np.where(skeleton > 0)

                # If there are enough points, apply PCA for angle
                if len(x_skel) > 1:
                    points_skel = np.column_stack((x_skel, y_skel))  # Convert to (x, y)

                    # Apply PCA
                    mean2, eigenvectors2 = cv2.PCACompute(points_skel.astype(np.float32), mean=None)

                    # Principal vector
                    principal_vector = eigenvectors2[0]

                    # Calculate angle in degrees with respect to the horizontal
                    angle = np.arctan2(principal_vector[1], principal_vector[0])
                    angle_degrees = np.degrees(angle)
                else:
                    angle_degrees = 0

                # Decide if the mask is U OR STRAIGHT (Tipo U or Tipo Recto)
                x_, y_, w_, h_ = cv2.boundingRect(binary_mask)
                aspect_ratio = w_ / float(h_)

                # Choose a threshold; if the shape is wider than tall => "U"
                if aspect_ratio >= 1.2:
                    shape_label = "U"       # We want it to be HORIZONTAL
                    # If the principal direction is very close to 90°, rotate it to make it horizontal
                    # Simple equation: we rotate by -angle_degrees => it will become horizontal
                    final_angle = angle_degrees
                else:
                    shape_label = "Recto"   # We want it to be VERTICAL
                    # To make the principal axis vertical, we rotate (90 - angle_degrees)
                    final_angle = 90 - angle_degrees

                # ROTATE THE IMAGE
                angle_rad = np.radians(angle_degrees)
                (h, w) = binary_mask.shape[:2]

                # Calculate new dimensions after rotating
                # - We use absolute values of cos and sin to cover the "worst" case in each dimension.
                cos_ = abs(np.cos(angle_rad))
                sin_ = abs(np.sin(angle_rad))

                new_w = int(w * cos_ + h * sin_)
                new_h = int(h * cos_ + w * sin_)

                center_img = (w // 2, h // 2)

                # Create the rotation matrix using -final_angle (so that the principal axis
                # aligns as desired)
                rotation_matrix = cv2.getRotationMatrix2D(center_img, -final_angle, 1.0)

                # Adjust the translation so the image appears centered in the new size
                rotation_matrix[0, 2] += (new_w / 2) - center_img[0]
                rotation_matrix[1, 2] += (new_h / 2) - center_img[1]

                # Apply rotation to the mask 
                rotated_image = cv2.warpAffine(
                                    binary_mask,
                                    rotation_matrix,
                                    (new_w, new_h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,  # Black background
                                    borderValue=0
                )

                # Apply rotation to the skeleton as well
                rotated_skeleton = cv2.warpAffine(
                    skeleton,
                    rotation_matrix,
                    (new_w, new_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # --- PASS 2 ---

                # Generate SKELETON
                skeleton = cv2.ximgproc.thinning(binary_mask)

                # Find coordinates of pixels where skeleton has a value of 255
                ys, xs = np.where(skeleton == 255)
                # Combine into a coordinate matrix (each row is [x, y])
                points = np.column_stack((xs, ys))

                # Ensure there are enough points
                if len(points) > 0:
                    mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))
                    
                    # First eigenvector == principal direction.
                    principal_axis = eigenvectors[0]  
                    
                    # Center
                    center = mean[0]  
                    
                    # To visualize the axis, define two points along the axis
                    scale = 100  
                    pt1 = (int(center[0] - scale * principal_axis[0]), int(center[1] - scale * principal_axis[1]))
                    pt2 = (int(center[0] + scale * principal_axis[0]), int(center[1] + scale * principal_axis[1]))
                # else:
                #     print("No points found in the skeleton")


                # Draw the line in color
                skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                cv2.line(skeleton_color, pt1, pt2, (0, 0, 255), thickness=2)

                # Get coordinates of the white points of the skeleton
                y, x = np.where(skeleton > 0)

                # If there are enough points, apply PCA
                if len(x) > 1:
                    points = np.column_stack((x, y))  # Convert to (x, y)

                    # Apply PCA
                    mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)

                    # Principal vector
                    principal_vector = eigenvectors[0]

                    # Calculate angle in degrees with respect to the horizontal
                    angle = np.arctan2(principal_vector[1], principal_vector[0])
                    angle_degrees = np.degrees(angle)

                    # Orientation adjustment: Ensure the axis is always vertical (close to 90° or -90°)
                    if abs(angle_degrees) > 45:  
                        angle_degrees = 90 - angle_degrees  
                else:
                    angle_degrees = 0  

                # --------- ROTATE THE IMAGE -----------------
                (h, w) = binary_mask.shape[:2]
                center = (w // 2, h // 2)

                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)

                # Apply rotation
                rotated_image = cv2.warpAffine(binary_mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                rotated_skeleton = cv2.warpAffine(skeleton, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)#####

                # Show the rotated image
                plt.figure(figsize=(8, 8))
                plt.imshow(rotated_image, cmap='gray')

                #-------Determine if the implant is top (Superior) or bottom (Inferior)--------

                (h, w) = rotated_image.shape[:2]
                mid_x = w // 2
                left_block_x = rotated_image[:, :mid_x]
                right_block_x = rotated_image[:, mid_x:]

                # Extract keypoints for each block
                left_top_x = get_top_corner(left_block_x, side="left", offset_x=0, tol=3)
                left_bottom_x = get_bottom_corner(left_block_x, side="left", offset_x=0, tol=2)
                right_top_x = get_top_corner(right_block_x, side="right", offset_x=mid_x, tol=3)
                right_bottom_x = get_bottom_corner(right_block_x, side="right", offset_x=mid_x, tol=2)

                # Verify that all points have been obtained
                if not None in (left_top_x, left_bottom_x, right_top_x, right_bottom_x):
                    # Calculate horizontal distances between top and bottom corners
                    top_distance = np.linalg.norm(np.array(left_top_x) - np.array(right_top_x))
                    bottom_distance = np.linalg.norm(np.array(left_bottom_x) - np.array(right_bottom_x))
                    
                    # Compare the distances
                    # If top distance is greater, it is assumed the implant extends more towards the top (hence it is "Inferior")
                    if top_distance > bottom_distance:
                        position = "Inferior"
                    else:
                        position = "Superior"
                    
                    # print(f"The implant is in the zone: {position}")
    
                # If the mask is "Superior", rotate 180°
                if position == "Superior":
                    rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)

                #--------- Determine if the implant is straight or U-shaped ----------------

                tipo_implante, drop1, drop2, peak1, peak2 = classify_mask_from_array_improved3(rotated_image)[:5]

                # Straighten properly if it is straight #
                _, rotated_bin = cv2.threshold(rotated_image, 127, 255, cv2.THRESH_BINARY)
                
                # Extract only the string ("Tipo Recto" or "Tipo U") => the first position of the tuple
                shape_label = tipo_implante[0]
                
                # Straighten the mask only if it is "Tipo Recto"
                if tipo_implante == "Tipo Recto":
                    corrected_mask, angle_used = enderezar_mascara_recto_regresion_lineal(rotated_bin, shape_label, debug=False)
                else:
                    corrected_mask = rotated_bin
                    angle_used = 0.0

                #----------------OBTAIN TOP AND BOTTOM KEYPOINTS----------------------------
                
                # rotated_image is the mask of the implant
                (h, w) = corrected_mask.shape[:2]

                # Division into two vertical blocks: left and right
                mid_x = w // 2
                left_block = corrected_mask[:, :mid_x]
                right_block = corrected_mask[:, mid_x:]

                # Extract keypoints for each block:
                left_bottom = get_bottom_corner(left_block, side="left", offset_x=0, tol=2)
                left_top = get_top_corner(left_block, side="left", offset_x=0, tol=3)
                right_bottom = get_bottom_corner(right_block, side="right", offset_x=mid_x, tol=2)
                right_top = get_top_corner(right_block, side="right", offset_x=mid_x, tol=3)

                #-------FIND CHANGE POINT IN THE CENTRAL AXIS------

                if position == "Superior":
                    rotated_skeleton = cv2.rotate(rotated_skeleton, cv2.ROTATE_180)

                # Get the coordinates of the white pixels in the rotated skeleton
                y_coords, x_coords = np.where(rotated_skeleton == 255)
                # if len(x_coords) == 0:
                #     print("No white pixels in the rotated skeleton.")

                # Apply PCA to find the center and principal axis of the object
                points = np.column_stack((x_coords, y_coords))  # Each row: [x, y]
                if points.shape[0] == 0:
                        # print(f"Image {i}: Not enough points to calculate PCA. Skipping...")
                        break

                mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
                center_pca = mean[0]      # Center of the object
                principal_vector = eigenvectors[0]  # Principal direction

                # To visualize the central axis, two points along the principal vector
                scale = 100  
                pt1 = (int(center_pca[0] - scale * principal_vector[0]),
                   int(center_pca[1] - scale * principal_vector[1]))
                pt2 = (int(center_pca[0] + scale * principal_vector[0]),
                    int(center_pca[1] + scale * principal_vector[1]))

                # Use center to define the central column of the object in the rotated image
                central_x = int(center_pca[0])

                # Search for the first "white" pixel in rotated_image in that column, using a threshold
                threshold = 200  
                change_point = None
                for y in range(rotated_image.shape[0]):
                    if rotated_image[y, central_x] > threshold:
                        change_point = (central_x, y)
                        break
                
                skeleton_color = cv2.cvtColor(rotated_skeleton, cv2.COLOR_GRAY2BGR)
                cv2.line(skeleton_color, pt1, pt2, (0, 0, 255), thickness=2)

                #-------------------------EXTRACT THREAD KEYPOINTS--------------------------

                left_block = corrected_mask[:, :mid_x]
                right_block = corrected_mask[:, mid_x:]

                # Get keypoints for both sides
                peaks_left, valleys_left = get_keypoints(left_block, is_left=True, offset_x=0)
                peaks_right, valleys_right = get_keypoints(right_block, is_left=False, offset_x=mid_x)

                # Filter the points so they are only below the line
                if change_point:
                    threshold_y = change_point[1]
                    peaks_left = [(x, y) for x, y in peaks_left if y > threshold_y]
                    valleys_left = [(x, y) for x, y in valleys_left if y > threshold_y]
                    peaks_right = [(x, y) for x, y in peaks_right if y > threshold_y]
                    valleys_right = [(x, y) for x, y in valleys_right if y > threshold_y]

                #---Calculate regression line-------

                # Get keypoints and regression parameters for each side
                result_left = get_keypoints_with_regression(peaks_left, valleys_left)
                if result_left is None:
                    continue  # Skip to next image if no data

                result_right = get_keypoints_with_regression(peaks_right, valleys_right)
                if result_right is None:
                    continue  # Skip to next image if no data

                # Unpack after ensuring it is not None
                peaks_left, valleys_left, slope_left, intercept_left, angle_left, xs_valleys_left, ys_valleys_left = result_left
                peaks_right, valleys_right, slope_right, intercept_right, angle_right, xs_valleys_right, ys_valleys_right = result_right

                # Filter the points, keeping only those below change_point 
                if change_point:
                    threshold_y = change_point[1]
                    peaks_left = [(x, y) for x, y in peaks_left if y > threshold_y]
                    valleys_left = [(x, y) for x, y in valleys_left if y > threshold_y]
                    peaks_right = [(x, y) for x, y in peaks_right if y > threshold_y]
                    valleys_right = [(x, y) for x, y in valleys_right if y > threshold_y]

                # Calculate the extremes for the regression line on the left side
                if len(xs_valleys_left) > 0:#
                    x_min_left = np.min(xs_valleys_left)
                    x_max_left = np.max(xs_valleys_left)
                    y_min_left = slope_left * x_min_left + intercept_left
                    y_max_left = slope_left * x_max_left + intercept_left
                else:#
                    x_min_left, x_max_left, y_min_left, y_max_left = 0, 0, 0, 0#

                # Calculate the extremes for the regression line on the right side
                if len(xs_valleys_right) > 0:
                    x_min_right = np.min(xs_valleys_right)
                    x_max_right = np.max(xs_valleys_right)
                    y_min_right = slope_right * x_min_right + intercept_right
                    y_max_right = slope_right * x_max_right + intercept_right
                else:
                    x_min_right, x_max_right, y_min_right, y_max_right = 0, 0, 0, 0#

                # Correct only the x coordinate of the bottom points
                if left_bottom is not None:
                    left_bottom_corr = corregir_x_a_linea_regresion(left_bottom, slope_left, intercept_left, corrected_mask)
                else:
                    left_bottom_corr = None

                if right_bottom is not None:
                    right_bottom_corr = corregir_x_a_linea_regresion(right_bottom, slope_right, intercept_right,corrected_mask)
                else:
                    right_bottom_corr = None

                height, width = rotated_image.shape[:2]
                x_min_left, y_min_left, x_max_left, y_max_left = clip_line(x_min_left, y_min_left, x_max_left, y_max_left, width, height)
                x_min_right, y_min_right, x_max_right, y_max_right = clip_line(x_min_right, y_min_right, x_max_right, y_max_right, width, height)

                #----------PLOT JOINED POINTS------
                
                plt.figure(figsize=(8, 8))
                plt.imshow(corrected_mask, cmap='gray')

                # Top and bottom corner points
                if tipo_implante == "Tipo Recto":
                    if right_top is not None:
                        plt.scatter(right_top[0], right_top[1], color='cyan', marker='^', s=80, label='Top Right')
                    if left_top is not None:
                        plt.scatter(left_top[0], left_top[1], color='magenta', marker='^', s=80, label='Top Left')
                    if right_bottom is not None:
                        plt.scatter(right_bottom_corr[0], right_bottom_corr[1], color='gold', marker='v', s=80, label='Bottom Right')
                    if left_bottom is not None:
                        plt.scatter(left_bottom_corr[0], left_bottom_corr[1], color='lime', marker='v', s=80, label='Bottom Left')

                if tipo_implante == "Tipo U":
                    plt.scatter(peak1[0], peak1[1], c='cyan', s=60, marker='o', label='Peak1')
                    plt.scatter(peak2[0], peak2[1], c='magenta', s=60, marker='o', label='Peak2')
                    if right_bottom is not None:
                        plt.scatter(right_bottom_corr[0], right_bottom_corr[1], color='gold', marker='v', s=80, label='Bottom Right')
                    if left_bottom is not None:
                        plt.scatter(left_bottom_corr[0], left_bottom_corr[1], color='lime', marker='v', s=80, label='Bottom Left')

                # Thread peaks
                first_peak = True
                for (x_peak, y_peak) in peaks_left + peaks_right:
                    if first_peak:
                        plt.scatter(x_peak, y_peak, color='pink', marker='o', s=30, alpha=0.6, label='Thread peaks')
                        first_peak = False
                    else:
                        plt.scatter(x_peak, y_peak, color='pink', marker='o', alpha=0.6, s=30)

                # Thread valleys 
                first_valley = True
                for (x_valley, y_valley) in valleys_left + valleys_right:
                    if first_valley:
                        plt.scatter(x_valley, y_valley, color='brown', marker='o', alpha=0.6,  s=30, label='Thread valleys')
                        first_valley = False
                    else:
                        plt.scatter(x_valley, y_valley, color='brown', marker='o', alpha=0.6,  s=30)

                # Regression line passing through the valleys:
                # defined for the left side
                plt.plot([x_min_left, x_max_left], [y_min_left, y_max_left], color='red', linestyle='--', linewidth=2, 
                        label=f"Left Valley Regression (Angle: {angle_left:.2f}°)")
                # defined for the right side
                plt.plot([x_min_right, x_max_right], [y_min_right, y_max_right], color='blue', linestyle='--', linewidth=2, 
                        label=f"Right Valley Regression (Angle: {angle_right:.2f}°)")

                # Interior points
                if tipo_implante == "Tipo U" and all(p is not None for p in [peak1, peak2, drop1, drop2]):
                    plt.scatter(drop1[0], drop1[1], c='orange',   s=60, marker='o', label='Drop1')
                    plt.scatter(drop2[0], drop2[1], c='blue',  s=60, marker='o', label='Drop2')

                annotated_img_np = np.array(annotated_img)
                
                # --- View on original cropped image ---
                (h, w) = annotated_img_np.shape[:2]
                center = (w // 2, h // 2)

                # Transform sets of points
                peaks_left_original  = transform_points_back(peaks_left,  center, -angle_degrees, angle_used, position)
                peaks_right_original = transform_points_back(peaks_right, center, -angle_degrees,angle_used, position)
                valleys_left_original  = transform_points_back(valleys_left,  center, -angle_degrees, angle_used, position)
                valleys_right_original = transform_points_back(valleys_right, center, -angle_degrees,angle_used, position)

                # Transform mask corners (top and bottom corner points)
                if tipo_implante == "Tipo U":
                        left_top_original = transform_points_back([peak1], center, -angle_degrees, angle_used, position)[0]
                        right_top_original = transform_points_back([peak2], center, -angle_degrees, angle_used, position)[0]
                        drop1_original = transform_points_back([drop1], center, -angle_degrees, angle_used, position)[0]
                        drop2_original = transform_points_back([drop2], center, -angle_degrees, angle_used, position)[0]
                else:
                        if right_top is not None:
                            right_top_original = transform_points_back([right_top], center, -angle_degrees,angle_used, position)[0]
                        if left_top is not None:
                            left_top_original = transform_points_back([left_top], center, -angle_degrees,angle_used, position)[0]

                if right_bottom_corr is not None:
                    right_bottom_original = transform_points_back([right_bottom_corr], center, -angle_degrees, angle_used, position)[0]
                if left_bottom is not None:
                    left_bottom_original = transform_points_back([left_bottom_corr], center, -angle_degrees,angle_used,  position)[0]

                # Transform the regression line for the valleys:
                line_left_trans = transform_points_back([(x_min_left, y_min_left), (x_max_left, y_max_left)], center, -angle_degrees,angle_used, position)
                line_right_trans = transform_points_back([(x_min_right, y_min_right), (x_max_right, y_max_right)], center, -angle_degrees,angle_used, position)

                # Plot on the original image
                plt.figure(figsize=(8, 8))
                plt.imshow(annotated_img_np)

                # Draw interior points
                if tipo_implante == "Tipo U":
                            plt.scatter(drop1_original[0], drop1_original[1], c='red', label='Drop1', zorder=3)
                            plt.scatter(drop2_original[0], drop2_original[1], c='blue', label='Drop2', zorder=3)

                # Draw peaks 
                first_peak = True
                for (x, y) in peaks_left_original + peaks_right_original:
                    if first_peak:
                        plt.scatter(x, y, color='pink', s=30, alpha=0.8, label='Peaks')
                        first_peak = False
                    else:
                        plt.scatter(x, y, color='pink', s=30, alpha=0.8)

                # Draw valleys 
                first_valley = True
                for (x, y) in valleys_left_original + valleys_right_original:
                    if first_valley:
                        plt.scatter(x, y, color='brown', s=30, alpha=0.8, label='Valleys')
                        first_valley = False
                    else:
                        plt.scatter(x, y, color='brown', s=30, alpha=0.8)

                # Draw top and bottom corner points
                if right_top is not None:
                    plt.scatter(right_top_original[0], right_top_original[1], color='cyan', marker='^', s=80, label='Top Right')
                if left_top is not None:
                    plt.scatter(left_top_original[0], left_top_original[1], color='magenta', marker='^', s=80, label='Top Left')
                if right_bottom_corr is not None:
                    plt.scatter(right_bottom_original[0], right_bottom_original[1], color='gold', marker='v', s=80, label='Bottom Right')
                if left_bottom_corr is not None:
                    plt.scatter(left_bottom_original[0], left_bottom_original[1], color='lime', marker='v', s=80, label='Bottom Left')

                # Draw regression line
                plt.plot([line_left_trans[0][0], line_left_trans[1][0]],
                        [line_left_trans[0][1], line_left_trans[1][1]],
                        color='blue', linestyle='--', linewidth=2, 
                        label=f"Left Valley Regression (Angle: {angle_left:.2f}°)")
                plt.plot([line_right_trans[0][0], line_right_trans[1][0]],
                        [line_right_trans[0][1], line_right_trans[1][1]],
                        color='blue', linestyle='--', linewidth=2, 
                        label=f"Right Valley Regression (Angle: {angle_right:.2f}°)")

                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.title("Points transformed back to the original image")
                plt.axis("off")
                plt.close()

                brand = annotation_df.loc[file_id]['labels'][i]
                # print(brand)
                implant_diameter_mm = annotation_df.loc[file_id]['diameter'][i]
                implant_diameter_mm = float(annotation_df.loc[file_id]['diameter'][i])
                # print(implant_diameter_mm)

                # calculate pixel/cm ratio of the plate
                # Extract original dimensions (in pixels) from annotation_df or sample_img
                orig_w = annotation_df.loc[file_id]['width']
                orig_h = annotation_df.loc[file_id]['height']

                if orig_h >= orig_w:
                    # Vertical case: 3cm x 4cm plate
                    pixel_per_cm_width = orig_w / 3.0   # width in pixels / 3 cm
                    pixel_per_cm_height = orig_h / 4.0  # height in pixels / 4 cm
                else:
                    # Horizontal case: the plate has been rotated
                    pixel_per_cm_width = orig_w / 4.0   # width in pixels / 4 cm
                    pixel_per_cm_height = orig_h / 3.0  # height in pixels / 3 cm

                pixel_per_cm = (pixel_per_cm_width + pixel_per_cm_height) / 2.0

                # Calculate the implant diameter in pixels using the keypoints
                implant_diameter_px = right_top[0] - left_top[0]

                # check if pixel_per_cm and implant_diameter_mm are valid
                if pixel_per_cm is None or implant_diameter_mm is None:
                    # print(f"Warning: pixel_per_cm ({pixel_per_cm}) or implant diameter ({implant_diameter_mm}) are None for the image. Skipping image...")
                    continue  

                # expected_px: expected size in pixels of the implant according to the plate
                expected_px = implant_diameter_mm * (pixel_per_cm / 10)

                # Calculate the distance at which the X-ray was taken using rule of three:
                distance_cm = 2.0 * (expected_px / implant_diameter_px)

                # Define output directories
                images_out_dir = base_dir / "cropped_images"
                masks_out_dir = base_dir / "masks"

                # Create output directories if they do not exist
                images_out_dir.mkdir(parents=True, exist_ok=True)
                masks_out_dir.mkdir(parents=True, exist_ok=True)
                
                if position == "Superior":
                    current_sample_img = current_sample_img.rotate(180, expand=True)
                    x_min = current_sample_img.width - (x_min + w)
                    y_min = sample_img.height - (y_min + h)

                x_max = x_min + w
                y_max = y_min + h
                center_bbox = (x_min + w // 2, y_min + h // 2)

                # Calculate the real angle applied to the mask
                total_angle = angle_used - angle_degrees

                # Apply the same rotation to the real cropped image
                sample_rotated_img, new_center = rotate_image_around_point(current_sample_img, total_angle, center=center_bbox)

                # Use original width and height with margin (optional)
                margin = 3
                new_x_min = new_center[0] - w // 2 - margin
                new_y_min = new_center[1] - h // 2 - margin
                new_x_max = new_center[0] + w // 2 + margin
                new_y_max = new_center[1] + h // 2 + margin

                cropped_rotated_img = sample_rotated_img.crop((new_x_min, new_y_min, new_x_max, new_y_max))

                # Ensure mask format
                if corrected_mask.max() == 1:
                    corrected_mask = (corrected_mask * 255).astype(np.uint8)

                # Equalize sizes if needed
                if cropped_rotated_img.size != Image.fromarray(corrected_mask).size:
                    cropped_rotated_img = cropped_rotated_img.resize(corrected_mask.shape[::-1], resample=Image.BILINEAR)

                # Save files
                image_name = f"{file_id}_{i}"
                cropped_rotated_img.save(images_out_dir / f"{image_name}.png")
                Image.fromarray(corrected_mask).convert("L").save(masks_out_dir / f"{image_name}_seg.png")

                # Prepare dictionary with features to save:
                features = {
                    "image_name": image_name,
                    "orig_width": annotation_df.loc[file_id]['width'],
                    "orig_height": annotation_df.loc[file_id]['height'],
                    "pixel_per_cm": pixel_per_cm,
                    "bbox_width": orig_width_bbox,
                    "bbox_height": orig_height_bbox,
                    "implant_bbox_ratio": implant_bbox_ratio,
                    # Brand
                    "brand": brand,
                    "implant_diameter_mm": implant_diameter_mm,
                    "distance_cm": distance_cm,
                    # Anatomical keypoints
                    "left_top": left_top,
                    "right_top": right_top,
                    "left_bottom": left_bottom_corr,
                    "right_bottom": right_bottom_corr,
                    "interior_left": left_top,  
                    "interior_right": right_top, 
                    # Valley regression angles 
                    "angle_left_valley": angle_left,
                    "angle_right_valley": angle_right,
                    "position": position
                }
                
                # Update keypoints according to the implant type
                if tipo_implante == "Tipo U":
                    # For "U Type" use drop1 and drop2, and replace left_top and right_top with peak1/peak2
                    features["interior_left"] = drop1  
                    features["interior_right"] = drop2
                    features["left_top"] = peak1
                    features["right_top"] = peak2
                elif tipo_implante == "Tipo Recto":
                    # For "Straight Type", interior_left/right are taken as left_top/right_top
                    features["interior_left"] = left_top
                    features["interior_right"] = right_top
                    
                # Accumulate the features in the list
                features_list.append(features)

    # If features were extracted, return them; otherwise, another value or annotated image can be returned
    if features_list:
        return features_list
    else:
        # If cropped_img is not defined, skip the image returning None
        if 'cropped_img' not in locals():
            # print("Warning: 'cropped_img' is not defined. Skipping image.")
            return None
        
    return cropped_img  # Return the annotated image if it exists


def read_annotations(base_dir):
    base_dir = Path(base_dir)
        
    # Image and annotation directory in each set
    img_dir = base_dir / "images"
    print(f"Processing image directory: {base_dir}")
    print(f"Processing image directory: {img_dir}")
    annot_dir = base_dir
    
    annotation_file_path = list(annot_dir.glob("*.json"))[0]
    print(f"Processing json as annotation: {annotation_file_path}")
    
    # Load the annotation file
    with open(annotation_file_path) as f:
        _coco = json.load(f)

    # Extract image information
    images_df = pd.DataFrame(_coco.get('images', []))[['file_name', 'height', 'width', 'id']]

    # Extract annotations
    annotations_df = pd.DataFrame(_coco.get('annotations', []))

    # Extract categories
    categories_df = pd.DataFrame(_coco.get('categories', []))
    categories_df.set_index('id', inplace=True)

    # Extract diameter from attributes
    annotations_df['diameter'] = annotations_df['attributes'].apply(
        lambda x: x.get('DIAMETER') if isinstance(x, dict) else None
    )

    # Keep only needed columns
    annotations_df = annotations_df[['image_id', 'segmentation', 'bbox', 'category_id', 'diameter', 'area']].copy()

    # Add category name to annotations
    annotations_df = annotations_df[annotations_df['category_id'].notna()].copy()
    annotations_df['label'] = annotations_df['category_id'].apply(
        lambda x: categories_df.loc[x]['name'] if pd.notna(x) and x in categories_df.index else None
    )

    # print(annotations_df)

    # Merge image and annotation information
    annotation_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')
    annotation_df.drop('id', axis=1, inplace=True)

    # Extract image_id from the file name
    annotation_df['image_id'] = annotation_df['file_name'].apply(lambda x: x.split('.')[0])
    annotation_df.set_index('image_id', inplace=True)

    # Group annotations by image, so each image has lists of bounding boxes, labels, etc.
    annotation_df = annotation_df.groupby('image_id').agg({
        'segmentation': list,
        'bbox':         list,
        'category_id':  list,
        'label':        list,
        'diameter':     list,
        'area':         list,
        'file_name':    'first',
        'height':       'first',
        'width':        'first'
    })
    annotation_df.rename(columns={'bbox': 'bboxes', 'label': 'labels'}, inplace=True)
    return annotation_df, img_dir

#-----------------------------CONFIGURATION----------------------------------------------------

# Anyone else running this code only needs to change these directories.

# Define path where the image and mask dataset is stored
_here      = Path(__file__).parent.resolve()
OUTPUT_DIR = _here / "outputs" 
BASE_DIR = OUTPUT_DIR / "dataset_split"


#-----------------------------PROCESSING SCRIPT------------------------------------------------
def main():

    splits = [
        Path(os.path.join(BASE_DIR, d))
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d != "models" 
    ]
  # features of all images


    num_exixting_images = 0
    total_images = 0

    # Process each split
    for split in splits:
        print(f"Processing directory: {split}")

        results = []
        annotation_df, img_dir = read_annotations(split)
        total_images += len(annotation_df)

        # Iterate through each annotated image in this set
        for image_id, row in annotation_df.iterrows():
            img_file = img_dir / row['file_name']
            if not img_file.exists():
                print(f"The image {img_file} does not exist, skipping.")
                continue

            num_exixting_images += 1

            sample_img = Image.open(img_file).convert("RGB")
            labels_for_image = row['labels']

            features = process_implant(image_id, sample_img, annotation_df, labels_for_image, split)

            if features:
                if isinstance(features, list):
                    results.extend(features)
                else:
                    results.append(features)
        
        print(f"Processed {num_exixting_images}/{total_images} images in directory '{split}'.")

        # Convert results into a DataFrame and save to CSV
        valid_results = [res for res in results if isinstance(res, dict)]
        df_features = pd.DataFrame(valid_results)

        df_features.to_csv(split / "features.csv", index=False)

        print(f"Dataset saved to '{split}'.")


if __name__ == "__main__":
    main()