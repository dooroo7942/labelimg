#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import json
from datetime import datetime

try:
    from PyQt5.QtCore import QPointF
except ImportError:
    from PyQt4.QtCore import QPointF


class TemplateMatchingEngine:
    """OpenCV-based template matching engine for LabelImg"""

    def __init__(self, templates_dir="data/templates"):
        self.templates_dir = templates_dir
        self.templates = {}  # {class_name: [template_data, ...]}
        self._ensure_templates_dir()

    def _ensure_templates_dir(self):
        """Create templates directory if it doesn't exist"""
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)

    def save_template(self, image, bbox, class_name):
        """
        Save a template image from the bounding box region

        Args:
            image: numpy array (BGR format from OpenCV)
            bbox: tuple (x, y, w, h) or list of 4 QPointF points
            class_name: string label for the template

        Returns:
            str: path to saved template
        """
        # Convert QPointF list to bbox if necessary
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if hasattr(bbox[0], 'x'):  # QPointF objects
                xs = [p.x() for p in bbox]
                ys = [p.y() for p in bbox]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs) - x)
                h = int(max(ys) - y)
            else:
                x, y, w, h = bbox
        else:
            raise ValueError("Invalid bbox format")

        # Ensure valid crop region
        h_img, w_img = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box dimensions")

        # Crop the template region
        template = image[y:y+h, x:x+w].copy()

        # Create class directory
        class_dir = os.path.join(self.templates_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"template_{timestamp}.png"
        filepath = os.path.join(class_dir, filename)

        # Save template image
        cv2.imwrite(filepath, template)

        # Save metadata
        metadata = {
            'original_size': (w, h),
            'created': timestamp,
            'class_name': class_name
        }
        metadata_file = os.path.join(class_dir, f"template_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Reload templates for this class
        self._load_class_templates(class_name)

        return filepath

    def _load_class_templates(self, class_name):
        """Load all templates for a specific class"""
        class_dir = os.path.join(self.templates_dir, class_name)
        if not os.path.exists(class_dir):
            self.templates[class_name] = []
            return

        templates = []
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(class_dir, filename)
                template_img = cv2.imread(filepath)
                if template_img is not None:
                    templates.append({
                        'image': template_img,
                        'path': filepath,
                        'size': template_img.shape[:2]
                    })

        self.templates[class_name] = templates

    def load_all_templates(self):
        """Load all templates from all class directories"""
        self.templates = {}
        if not os.path.exists(self.templates_dir):
            return

        for class_name in os.listdir(self.templates_dir):
            class_dir = os.path.join(self.templates_dir, class_name)
            if os.path.isdir(class_dir):
                self._load_class_templates(class_name)

    def get_template_classes(self):
        """Get list of classes that have templates"""
        return list(self.templates.keys())

    def match_templates(self, image, threshold=0.8, scale_tolerance=0.2,
                       rotation_tolerance=15, existing_boxes=None):
        """
        Perform template matching on the image

        Args:
            image: numpy array (BGR format)
            threshold: matching confidence threshold (0.0-1.0)
            scale_tolerance: scale variation tolerance (0.0-1.0, e.g., 0.2 = ±20%)
            rotation_tolerance: rotation tolerance in degrees (0-180)
            existing_boxes: list of existing bounding boxes to skip overlap

        Returns:
            list of Detection objects
        """
        if existing_boxes is None:
            existing_boxes = []

        detections = []

        # Load templates if not loaded
        if not self.templates:
            self.load_all_templates()

        # Convert image to grayscale for matching
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        for class_name, templates in self.templates.items():
            for template_data in templates:
                template = template_data['image']

                # Convert template to grayscale
                if len(template.shape) == 3:
                    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                else:
                    gray_template = template

                # Multi-scale and multi-rotation matching
                matches = self._multi_scale_rotation_match(
                    gray_image, gray_template,
                    threshold, scale_tolerance, rotation_tolerance
                )

                for match in matches:
                    x, y, w, h, confidence = match

                    # Check for overlap with existing boxes
                    if not self._overlaps_existing(x, y, w, h, existing_boxes):
                        detection = Detection(
                            x=x, y=y, w=w, h=h,
                            confidence=confidence,
                            class_name=class_name
                        )
                        detections.append(detection)

        # Apply NMS to remove duplicate detections across classes
        detections = self._apply_nms(detections, iou_threshold=0.5)

        return detections

    def _multi_scale_rotation_match(self, image, template, threshold,
                                    scale_tolerance, rotation_tolerance):
        """Perform matching with scale and rotation variations"""
        matches = []
        h, w = template.shape[:2]
        img_h, img_w = image.shape[:2]

        # Generate scale factors
        min_scale = max(0.1, 1.0 - scale_tolerance)
        max_scale = min(3.0, 1.0 + scale_tolerance)
        scale_step = 0.1 if scale_tolerance > 0 else 1.0
        scales = np.arange(min_scale, max_scale + scale_step, scale_step)

        # Generate rotation angles
        if rotation_tolerance > 0:
            angle_step = max(5, rotation_tolerance / 4)
            angles = np.arange(-rotation_tolerance, rotation_tolerance + angle_step, angle_step)
        else:
            angles = [0]

        for scale in scales:
            for angle in angles:
                # Scale and rotate template
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)

                # Skip if template is too small or too large
                if scaled_w < 10 or scaled_h < 10:
                    continue
                if scaled_w > img_w or scaled_h > img_h:
                    continue

                # Resize template
                scaled_template = cv2.resize(template, (scaled_w, scaled_h))

                # Rotate template if needed
                if angle != 0:
                    rotated_template = self._rotate_image(scaled_template, angle)
                else:
                    rotated_template = scaled_template

                # Get rotated template dimensions
                rot_h, rot_w = rotated_template.shape[:2]

                # Skip if rotated template is larger than image
                if rot_w > img_w or rot_h > img_h:
                    continue

                # Perform template matching
                try:
                    result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue

                # Find locations above threshold
                locations = np.where(result >= threshold)

                for pt_y, pt_x in zip(*locations):
                    confidence = result[pt_y, pt_x]
                    matches.append((pt_x, pt_y, rot_w, rot_h, float(confidence)))

        # Apply local NMS
        matches = self._nms_matches(matches, iou_threshold=0.3)

        return matches

    def _rotate_image(self, image, angle):
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        return rotated

    def _nms_matches(self, matches, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to matches"""
        if len(matches) == 0:
            return []

        # Sort by confidence
        matches = sorted(matches, key=lambda x: x[4], reverse=True)

        selected = []
        for match in matches:
            x1, y1, w1, h1, conf1 = match

            keep = True
            for sel in selected:
                x2, y2, w2, h2, conf2 = sel
                iou = self._compute_iou(x1, y1, w1, h1, x2, y2, w2, h2)
                if iou > iou_threshold:
                    keep = False
                    break

            if keep:
                selected.append(match)

        return selected

    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply NMS across all detections"""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        selected = []
        for det in detections:
            keep = True
            for sel in selected:
                iou = self._compute_iou(
                    det.x, det.y, det.w, det.h,
                    sel.x, sel.y, sel.w, sel.h
                )
                if iou > iou_threshold:
                    keep = False
                    break

            if keep:
                selected.append(det)

        return selected

    def _compute_iou(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Compute Intersection over Union"""
        # Convert to (x1, y1, x2, y2) format
        box1 = (x1, y1, x1 + w1, y1 + h1)
        box2 = (x2, y2, x2 + w2, y2 + h2)

        # Compute intersection
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _overlaps_existing(self, x, y, w, h, existing_boxes, iou_threshold=0.3):
        """Check if detection overlaps with existing bounding boxes"""
        for box in existing_boxes:
            if hasattr(box, 'points'):
                # Shape object with QPointF points
                xs = [p.x() for p in box.points]
                ys = [p.y() for p in box.points]
                bx = min(xs)
                by = min(ys)
                bw = max(xs) - bx
                bh = max(ys) - by
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                bx, by, bw, bh = box
            else:
                continue

            iou = self._compute_iou(x, y, w, h, bx, by, bw, bh)
            if iou > iou_threshold:
                return True

        return False


class Detection:
    """Represents a template matching detection result"""

    def __init__(self, x, y, w, h, confidence, class_name):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.class_name = class_name

    def to_points(self):
        """Convert to list of 4 QPointF points (top-left, top-right, bottom-right, bottom-left)"""
        return [
            QPointF(self.x, self.y),
            QPointF(self.x + self.w, self.y),
            QPointF(self.x + self.w, self.y + self.h),
            QPointF(self.x, self.y + self.h)
        ]

    def to_bbox(self):
        """Return (x, y, w, h) tuple"""
        return (self.x, self.y, self.w, self.h)

    def center(self):
        """Return center point"""
        return (self.x + self.w / 2, self.y + self.h / 2)

    def __repr__(self):
        return f"Detection(class={self.class_name}, bbox=({self.x},{self.y},{self.w},{self.h}), conf={self.confidence:.2f})"
