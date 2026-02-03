#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretrain Dialog for LabelImg
Shows class statistics and starts YOLO training
"""

import os

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.yolo_detector import YoloDetector, check_yolo_available


class PretrainDialog(QDialog):
    """Dialog for YOLO pretraining"""

    def __init__(self, parent=None, image_dir=None, classes_file=None):
        super(PretrainDialog, self).__init__(parent)
        self.image_dir = image_dir
        self.classes_file = classes_file
        self.training_thread = None

        self.setWindowTitle("YOLO Pretrain")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self.init_ui()
        self.load_statistics()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("클래스별 라벨링 현황")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["클래스", "이미지 수"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.stats_table)

        # Total count
        self.total_label = QLabel("총 라벨: 0개")
        layout.addWidget(self.total_label)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.load_statistics)
        btn_layout.addWidget(self.refresh_btn)

        btn_layout.addStretch()

        self.run_btn = QPushButton("Run Training")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        self.run_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.run_btn)

        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_statistics(self):
        """Load and display class statistics"""
        if not self.image_dir or not self.classes_file:
            self.status_label.setText("이미지 폴더를 먼저 열어주세요.")
            return

        if not os.path.exists(self.classes_file):
            self.status_label.setText("classes.txt 파일이 없습니다.")
            return

        counts = YoloDetector.count_class_images(self.image_dir, self.classes_file)

        self.stats_table.setRowCount(len(counts))
        total = 0

        for row, (cls_name, count) in enumerate(counts.items()):
            self.stats_table.setItem(row, 0, QTableWidgetItem(cls_name))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(count)))
            total += count

        self.total_label.setText(f"총 라벨: {total}개")

        if total == 0:
            self.status_label.setText("라벨링된 이미지가 없습니다. 먼저 라벨링을 진행해주세요.")
            self.run_btn.setEnabled(False)
        elif total < 10:
            self.status_label.setText(f"⚠️ 최소 10개 이상의 라벨이 필요합니다. (현재: {total}개)")
            self.run_btn.setEnabled(False)
        else:
            self.status_label.setText(f"✓ 학습 준비 완료 ({total}개 라벨)")
            self.run_btn.setEnabled(True)

    def start_training(self):
        """Start YOLO training"""
        if not check_yolo_available():
            QMessageBox.critical(self, "Error",
                                 "ultralytics가 설치되어 있지 않습니다.\n"
                                 "pip install ultralytics 명령으로 설치해주세요.")
            return

        # Create output directory
        pretrain_dir = os.path.join(os.path.dirname(self.image_dir), "Pretrain")
        dataset_dir = os.path.join(pretrain_dir, "dataset")
        os.makedirs(pretrain_dir, exist_ok=True)

        self.status_label.setText("데이터셋 준비 중...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.run_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Prepare dataset
            data_yaml, train_count, val_count = YoloDetector.prepare_dataset(
                self.image_dir, dataset_dir, self.classes_file
            )

            self.status_label.setText(f"학습 시작 (train: {train_count}, val: {val_count})...")
            QApplication.processEvents()

            # Start training in thread
            self.training_thread = TrainingThread(data_yaml, pretrain_dir)
            self.training_thread.progress.connect(self.on_training_progress)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            self.training_thread.start()

        except Exception as e:
            self.status_label.setText(f"오류: {str(e)}")
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)

    def on_training_progress(self, message):
        """Handle training progress updates"""
        self.status_label.setText(message)

    def on_training_finished(self, model_path):
        """Handle training completion"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)

        if model_path:
            # Copy to Pretrain/best.pt
            pretrain_dir = os.path.join(os.path.dirname(self.image_dir), "Pretrain")
            final_path = os.path.join(pretrain_dir, "best.pt")

            import shutil
            shutil.copy(model_path, final_path)

            self.status_label.setText(f"✓ 학습 완료! 모델 저장: {final_path}")
            QMessageBox.information(self, "학습 완료",
                                    f"YOLO 모델 학습이 완료되었습니다.\n\n"
                                    f"모델 경로: {final_path}\n\n"
                                    f"이제 'Run Yolo' 버튼으로 객체 인식을 사용할 수 있습니다.")
        else:
            self.status_label.setText("학습 실패: 모델 파일을 찾을 수 없습니다.")

    def on_training_error(self, error_msg):
        """Handle training error"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"오류: {error_msg}")
        QMessageBox.critical(self, "학습 오류", error_msg)


class TrainingThread(QThread):
    """Thread for YOLO training"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # model_path
    error = pyqtSignal(str)

    def __init__(self, data_yaml, output_dir):
        super().__init__()
        self.data_yaml = data_yaml
        self.output_dir = output_dir

    def run(self):
        try:
            self.progress.emit("YOLO 학습 중... (시간이 걸릴 수 있습니다)")

            model_path = YoloDetector.train(
                self.data_yaml,
                self.output_dir,
                epochs=100,
                batch_size=16,
                img_size=640
            )

            self.finished.emit(model_path or "")
        except Exception as e:
            self.error.emit(str(e))
