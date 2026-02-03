#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Detector Module for LabelImg
Provides YOLOv8 training and inference functionality
"""

import os
import shutil
from pathlib import Path

# Lazy import for ultralytics to avoid import errors when not needed
YOLO = None
YOLO_AVAILABLE = None

def _check_yolo():
    """Lazy check for YOLO availability"""
    global YOLO, YOLO_AVAILABLE
    if YOLO_AVAILABLE is None:
        try:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO
            YOLO_AVAILABLE = True
        except (ImportError, OSError) as e:
            YOLO_AVAILABLE = False
            print(f"Warning: ultralytics not available. {str(e)}")
    return YOLO_AVAILABLE


class YoloDetector:
    """YOLOv8 detector for training and inference"""

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.classes = []

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load a trained YOLO model"""
        if not _check_yolo():
            raise RuntimeError("ultralytics not available")

        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        return False

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def detect(self, image_path, conf_threshold=0.5):
        """
        Run detection on an image
        Returns list of detections: [(class_name, x, y, w, h, confidence), ...]
        """
        if not self.model:
            return []

        results = self.model(image_path, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = result.names[cls_id]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                detections.append((class_name, x, y, w, h, conf))

        return detections

    @staticmethod
    def count_class_images(image_dir, classes_file):
        """
        Count labeled images per class
        Returns dict: {class_name: count}
        """
        counts = {}

        # Load class names
        if not os.path.exists(classes_file):
            return counts

        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]

        for cls_name in class_names:
            counts[cls_name] = 0

        # Count labels
        label_dir = image_dir  # YOLO labels in same dir as images
        if not os.path.exists(label_dir):
            return counts

        for txt_file in Path(label_dir).glob("*.txt"):
            if txt_file.name == "classes.txt":
                continue

            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(class_names):
                            counts[class_names[cls_id]] += 1

        return counts

    @staticmethod
    def prepare_dataset(image_dir, output_dir, classes_file, train_ratio=0.8):
        """
        Prepare YOLO dataset structure from labeled images
        Creates: output_dir/images/train, output_dir/images/val, output_dir/labels/train, output_dir/labels/val
        Returns: path to data.yaml
        """
        import random

        # Create directories
        for split in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

        # Load class names
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]

        # Find all image-label pairs
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        pairs = []

        for img_file in Path(image_dir).iterdir():
            if img_file.suffix.lower() in image_extensions:
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    pairs.append((img_file, label_file))

        # Shuffle and split
        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        # Copy files
        for pairs_list, split in [(train_pairs, 'train'), (val_pairs, 'val')]:
            for img_file, label_file in pairs_list:
                shutil.copy(img_file, os.path.join(output_dir, 'images', split, img_file.name))
                shutil.copy(label_file, os.path.join(output_dir, 'labels', split, label_file.name))

        # Create data.yaml
        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")

        return yaml_path, len(train_pairs), len(val_pairs)

    @staticmethod
    def train(data_yaml, output_dir, epochs=100, batch_size=16, img_size=640):
        """
        Train YOLOv8 model
        Returns path to best.pt
        """
        if not _check_yolo():
            raise RuntimeError("ultralytics not available. Run 'pip install ultralytics'")

        # Start with pretrained YOLOv8n (nano - fastest)
        model = YOLO('yolov8n.pt')

        # Train
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=output_dir,
            name='train',
            exist_ok=True,
            verbose=True
        )

        # Return path to best model
        best_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
        return best_path if os.path.exists(best_path) else None


def check_yolo_available():
    """Check if YOLO is available"""
    return _check_yolo()
