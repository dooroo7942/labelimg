#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template Matching Module for LabelImg
OpenCV-based template matching with scale and rotation support
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

try:
    from PyQt5.QtCore import QPointF
except ImportError:
    from PyQt4.QtCore import QPointF


class TemplateManager:
    """Manages template storage and retrieval for each class"""

    def __init__(self, base_dir):
        """
        Initialize template manager

        Args:
            base_dir: Base directory for storing templates (e.g., data/templates/)

        Folder structure:
            templates/
            ├── images/          # Template images (for matching)
            │   ├── capacitor/
            │   │   └── template_xxx.png
            │   └── resistor/
            │       └── template_xxx.png
            └── metadata/        # Metadata files (separate)
                ├── capacitor/
                │   └── metadata.json
                └── resistor/
                    └── metadata.json
        """
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, 'images')
        self.metadata_dir = os.path.join(base_dir, 'metadata')
        self._ensure_dir_exists(self.images_dir)
        self._ensure_dir_exists(self.metadata_dir)

    def _ensure_dir_exists(self, path):
        """Create directory if it doesn't exist"""
        if not os.path.exists(path):
            os.makedirs(path)

    def get_class_image_dir(self, class_name):
        """Get the image directory path for a specific class"""
        class_dir = os.path.join(self.images_dir, class_name)
        self._ensure_dir_exists(class_dir)
        return class_dir

    def get_class_metadata_dir(self, class_name):
        """Get the metadata directory path for a specific class"""
        class_dir = os.path.join(self.metadata_dir, class_name)
        self._ensure_dir_exists(class_dir)
        return class_dir

    def save_template(self, image, class_name, bbox):
        """
        Save a template image from the bounding box region

        Args:
            image: numpy array (BGR format from OpenCV)
            class_name: Name of the class for this template
            bbox: Tuple of (x, y, width, height)

        Returns:
            Path to the saved template file
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Boundary check
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            print(f"[Template] Invalid bbox: x={x}, y={y}, w={w}, h={h}")
            return None

        # Extract template region
        template = image[y:y+h, x:x+w].copy()

        if template.size == 0:
            print(f"[Template] Empty template region")
            return None

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"template_{timestamp}.png"

        # Save to images folder
        class_image_dir = self.get_class_image_dir(class_name)
        filepath = os.path.join(class_image_dir, filename)

        # Save template image (handle Korean/Unicode paths)
        try:
            # Use imencode + file write to handle Unicode paths
            _, img_encoded = cv2.imencode('.png', template)
            with open(filepath, 'wb') as f:
                f.write(img_encoded.tobytes())
            success = True
        except Exception as e:
            print(f"[Template] Save failed: {e}")
            success = False

        if not success:
            print(f"[Template] Failed to save: {filepath}")
            print(f"[Template] Template shape: {template.shape}, dtype: {template.dtype}")
            return None

        print(f"[Template] Saved: {filepath} (shape: {template.shape})")

        # Save metadata to separate folder
        class_metadata_dir = self.get_class_metadata_dir(class_name)
        metadata_path = os.path.join(class_metadata_dir, "metadata.json")
        metadata = self._load_metadata(metadata_path)
        metadata[filename] = {
            "width": w,
            "height": h,
            "class": class_name,
            "created": timestamp
        }
        self._save_metadata(metadata_path, metadata)

        return filepath

    def _load_metadata(self, path):
        """Load metadata from JSON file"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self, path, data):
        """Save metadata to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_all_templates(self):
        """
        Get all templates organized by class

        Returns:
            Dict: {class_name: [(template_path, template_image), ...]}
        """
        templates = {}

        if not os.path.exists(self.images_dir):
            return templates

        for class_name in os.listdir(self.images_dir):
            class_dir = os.path.join(self.images_dir, class_name)
            if os.path.isdir(class_dir):
                templates[class_name] = []
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        template_path = os.path.join(class_dir, filename)
                        # Handle Korean/Unicode paths
                        template_img = imread_unicode(template_path)
                        if template_img is not None:
                            templates[class_name].append((template_path, template_img))

        return templates

    def get_classes(self):
        """Get list of all classes that have templates"""
        classes = []
        if os.path.exists(self.images_dir):
            for name in os.listdir(self.images_dir):
                if os.path.isdir(os.path.join(self.images_dir, name)):
                    classes.append(name)
        return classes


class TemplateMatcher:
    """Performs template matching with scale and rotation support"""

    def __init__(self, threshold=0.8, scale_tolerance=0.2, rotation_tolerance=15):
        """
        Initialize template matcher

        Args:
            threshold: Matching threshold (0.0 - 1.0)
            scale_tolerance: Scale variation tolerance (0.0 - 1.0, e.g., 0.2 = ±20%)
            rotation_tolerance: Rotation tolerance in degrees (0 - 180)
        """
        self.threshold = threshold
        self.scale_tolerance = scale_tolerance
        self.rotation_tolerance = rotation_tolerance

    def set_parameters(self, threshold=None, scale_tolerance=None, rotation_tolerance=None):
        """Update matching parameters"""
        if threshold is not None:
            self.threshold = threshold
        if scale_tolerance is not None:
            self.scale_tolerance = scale_tolerance
        if rotation_tolerance is not None:
            self.rotation_tolerance = rotation_tolerance

    def match_template(self, image, template, existing_boxes=None):
        """
        Find all matches of a template in the image

        Args:
            image: Source image (numpy array, BGR)
            template: Template image (numpy array, BGR)
            existing_boxes: List of existing bounding boxes to avoid overlap

        Returns:
            List of matches: [(x, y, w, h, confidence, scale, rotation), ...]
        """
        if image is None or template is None:
            return []

        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            return []

        matches = []

        # Generate scale range
        min_scale = max(0.1, 1.0 - self.scale_tolerance)
        max_scale = min(3.0, 1.0 + self.scale_tolerance)
        scale_steps = max(1, int(self.scale_tolerance * 10))
        scales = np.linspace(min_scale, max_scale, scale_steps * 2 + 1)

        # Generate rotation range
        if self.rotation_tolerance > 0:
            rotation_steps = max(1, int(self.rotation_tolerance / 5))
            rotations = np.linspace(-self.rotation_tolerance, self.rotation_tolerance, rotation_steps * 2 + 1)
        else:
            rotations = [0]

        # Convert to grayscale for matching
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template

        original_h, original_w = gray_template.shape[:2]

        for scale in scales:
            for rotation in rotations:
                # Scale template
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)

                if new_w < 5 or new_h < 5:
                    continue
                if new_w > gray_image.shape[1] or new_h > gray_image.shape[0]:
                    continue

                scaled_template = cv2.resize(gray_template, (new_w, new_h))

                # Rotate template if needed
                if rotation != 0:
                    rotated_template = self._rotate_template(scaled_template, rotation)
                    if rotated_template is None:
                        continue
                else:
                    rotated_template = scaled_template

                th, tw = rotated_template.shape[:2]
                if tw > gray_image.shape[1] or th > gray_image.shape[0]:
                    continue

                # Perform template matching
                try:
                    result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue

                # Find locations above threshold
                locations = np.where(result >= self.threshold)

                for pt in zip(*locations[::-1]):
                    x, y = pt
                    confidence = result[y, x]

                    # Calculate bounding box
                    bbox = (int(x), int(y), tw, th)

                    # Check overlap with existing boxes
                    if existing_boxes and self._has_overlap(bbox, existing_boxes):
                        continue

                    # Check overlap with already found matches
                    if self._has_overlap(bbox, [(m[0], m[1], m[2], m[3]) for m in matches]):
                        # Keep the one with higher confidence
                        should_add = True
                        for i, m in enumerate(matches):
                            if self._calculate_iou(bbox, (m[0], m[1], m[2], m[3])) > 0.3:
                                if confidence > m[4]:
                                    matches[i] = (x, y, tw, th, confidence, scale, rotation)
                                should_add = False
                                break
                        if not should_add:
                            continue

                    matches.append((x, y, tw, th, confidence, scale, rotation))

        # Sort by confidence
        matches.sort(key=lambda x: x[4], reverse=True)

        # Apply NMS
        matches = self._non_max_suppression(matches)

        return matches

    def _rotate_template(self, template, angle):
        """Rotate template by given angle"""
        h, w = template.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(template, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        return rotated

    def _has_overlap(self, bbox, existing_boxes, iou_threshold=0.3):
        """Check if bbox overlaps with any existing box"""
        for existing in existing_boxes:
            if self._calculate_iou(bbox, existing) > iou_threshold:
                return True
        return False

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _non_max_suppression(self, matches, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove overlapping matches"""
        if len(matches) == 0:
            return []

        # Sort by confidence (already sorted)
        result = []

        for match in matches:
            bbox = (match[0], match[1], match[2], match[3])

            # Check if this match overlaps with any selected match
            overlap = False
            for selected in result:
                selected_bbox = (selected[0], selected[1], selected[2], selected[3])
                if self._calculate_iou(bbox, selected_bbox) > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                result.append(match)

        return result

    def match_all_templates(self, image, templates_by_class, existing_boxes=None):
        """
        Match all templates from all classes against the image

        Args:
            image: Source image (numpy array, BGR)
            templates_by_class: Dict {class_name: [(path, template_img), ...]}
            existing_boxes: List of existing bounding boxes

        Returns:
            List of matches: [(x, y, w, h, class_name, confidence), ...]
        """
        all_matches = []

        # Collect existing boxes
        if existing_boxes is None:
            existing_boxes = []

        for class_name, templates in templates_by_class.items():
            for template_path, template_img in templates:
                matches = self.match_template(image, template_img, existing_boxes)

                for match in matches:
                    x, y, w, h, confidence, scale, rotation = match
                    all_matches.append((x, y, w, h, class_name, confidence))
                    # Add to existing boxes to prevent overlap
                    existing_boxes.append((x, y, w, h))

        return all_matches


def qimage_to_numpy(qimage):
    """Convert QImage to numpy array (BGR format for OpenCV)"""
    from PyQt5.QtGui import QImage

    # Get image dimensions
    width = qimage.width()
    height = qimage.height()

    if width == 0 or height == 0:
        print(f"[qimage_to_numpy] Invalid image size: {width}x{height}")
        return None

    # Convert to RGB32 format (most compatible)
    qimage = qimage.convertToFormat(QImage.Format_RGB32)

    # Get bytes per line (may include padding)
    bytes_per_line = qimage.bytesPerLine()

    # Get pointer to image data
    ptr = qimage.bits()
    if ptr is None:
        print("[qimage_to_numpy] Failed to get image bits")
        return None

    ptr.setsize(height * bytes_per_line)

    # Create numpy array
    arr = np.array(ptr, dtype=np.uint8).reshape(height, bytes_per_line)

    # Extract only the image data (remove padding if any)
    # RGB32 format is BGRA (4 bytes per pixel)
    arr = arr[:, :width * 4].reshape(height, width, 4)

    # Make a copy to own the data
    arr = arr.copy()

    # Convert BGRA to BGR (remove alpha channel)
    bgr = arr[:, :, :3]

    print(f"[qimage_to_numpy] Converted: {width}x{height}, shape={bgr.shape}, dtype={bgr.dtype}")

    return bgr


def numpy_to_qimage(arr):
    """Convert numpy array (BGR) to QImage"""
    from PyQt5.QtGui import QImage

    if len(arr.shape) == 2:
        # Grayscale
        height, width = arr.shape
        bytes_per_line = width
        return QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        # Color (BGR to RGB)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = 3 * width
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)


def imread_unicode(filepath):
    """Read image from Unicode/Korean path"""
    try:
        with open(filepath, 'rb') as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[imread_unicode] Failed to read {filepath}: {e}")
        return None
