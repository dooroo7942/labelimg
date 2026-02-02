#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template Matching Widget for LabelImg
Provides GUI controls for template matching functionality
"""

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class TemplateMatchingWidget(QWidget):
    """Widget containing template matching controls"""

    # Signals
    saveTemplateRequested = pyqtSignal()
    openTemplateFolderRequested = pyqtSignal()
    matchingRequested = pyqtSignal()
    parametersChanged = pyqtSignal(float, float, float)  # threshold, scale, rotation
    realtimeModeChanged = pyqtSignal(bool)
    hideBoxesRequested = pyqtSignal(bool)  # True = hide, False = show

    def __init__(self, parent=None):
        super(TemplateMatchingWidget, self).__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("Template Matching")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # Buttons row 1
        btn_layout1 = QHBoxLayout()

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setToolTip("Open templates folder in file explorer")
        self.open_folder_btn.clicked.connect(self.openTemplateFolderRequested.emit)
        btn_layout1.addWidget(self.open_folder_btn)

        self.save_template_btn = QPushButton("Save Template")
        self.save_template_btn.setToolTip("Save selected bounding box as template")
        self.save_template_btn.clicked.connect(self.saveTemplateRequested.emit)
        self.save_template_btn.setEnabled(False)
        btn_layout1.addWidget(self.save_template_btn)

        layout.addLayout(btn_layout1)

        # Buttons row 2
        btn_layout2 = QHBoxLayout()

        self.match_btn = QPushButton("Run Matching")
        self.match_btn.setToolTip("Execute template matching on current image")
        self.match_btn.clicked.connect(self.matchingRequested.emit)
        self.match_btn.setEnabled(False)
        btn_layout2.addWidget(self.match_btn)

        self.realtime_checkbox = QCheckBox("Real-time")
        self.realtime_checkbox.setToolTip("Enable real-time matching (auto-match on image change)")
        self.realtime_checkbox.stateChanged.connect(self._on_realtime_changed)
        btn_layout2.addWidget(self.realtime_checkbox)

        layout.addLayout(btn_layout2)

        # Info label for hide shortcut
        hide_info_label = QLabel("Press ` (backtick) to hide boxes")
        hide_info_label.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(hide_info_label)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Match Threshold:")
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(80)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self._on_parameter_changed)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("0.80")
        self.threshold_value_label.setMinimumWidth(35)
        threshold_layout.addWidget(self.threshold_value_label)

        layout.addLayout(threshold_layout)

        # Scale tolerance slider
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale Tolerance:")
        scale_layout.addWidget(scale_label)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(100)
        self.scale_slider.setValue(20)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_slider.setTickInterval(10)
        self.scale_slider.valueChanged.connect(self._on_parameter_changed)
        scale_layout.addWidget(self.scale_slider)

        self.scale_value_label = QLabel("20%")
        self.scale_value_label.setMinimumWidth(35)
        scale_layout.addWidget(self.scale_value_label)

        layout.addLayout(scale_layout)

        # Rotation tolerance slider
        rotation_layout = QHBoxLayout()
        rotation_label = QLabel("Rotation Tolerance:")
        rotation_layout.addWidget(rotation_label)

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setMinimum(0)
        self.rotation_slider.setMaximum(180)
        self.rotation_slider.setValue(15)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(15)
        self.rotation_slider.valueChanged.connect(self._on_parameter_changed)
        rotation_layout.addWidget(self.rotation_slider)

        self.rotation_value_label = QLabel("15")
        self.rotation_value_label.setMinimumWidth(35)
        rotation_layout.addWidget(self.rotation_value_label)

        layout.addLayout(rotation_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        # Add stretch to push everything to top
        layout.addStretch()

        self.setLayout(layout)

        # Debounce timer for real-time matching
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._trigger_realtime_match)

    def _on_parameter_changed(self):
        """Handle parameter slider changes"""
        threshold = self.threshold_slider.value() / 100.0
        scale = self.scale_slider.value() / 100.0
        rotation = self.rotation_slider.value()

        # Update labels
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self.scale_value_label.setText(f"{int(scale * 100)}%")
        self.rotation_value_label.setText(f"{rotation}")

        # Emit signal
        self.parametersChanged.emit(threshold, scale, rotation)

        # Trigger real-time matching if enabled
        if self.realtime_checkbox.isChecked():
            self.debounce_timer.start(300)

    def _on_realtime_changed(self, state):
        """Handle real-time checkbox state change"""
        is_checked = state == Qt.Checked
        self.realtimeModeChanged.emit(is_checked)

        if is_checked:
            self.match_btn.setEnabled(False)
            self._trigger_realtime_match()
        else:
            self.match_btn.setEnabled(True)

    def _trigger_realtime_match(self):
        """Trigger matching for real-time mode"""
        if self.realtime_checkbox.isChecked():
            self.matchingRequested.emit()

    def get_parameters(self):
        """Get current matching parameters"""
        return {
            'threshold': self.threshold_slider.value() / 100.0,
            'scale_tolerance': self.scale_slider.value() / 100.0,
            'rotation_tolerance': self.rotation_slider.value()
        }

    def set_parameters(self, threshold=None, scale_tolerance=None, rotation_tolerance=None):
        """Set matching parameters"""
        if threshold is not None:
            self.threshold_slider.setValue(int(threshold * 100))
        if scale_tolerance is not None:
            self.scale_slider.setValue(int(scale_tolerance * 100))
        if rotation_tolerance is not None:
            self.rotation_slider.setValue(int(rotation_tolerance))

    def set_save_template_enabled(self, enabled):
        """Enable/disable save template button"""
        self.save_template_btn.setEnabled(enabled)

    def set_matching_enabled(self, enabled):
        """Enable/disable matching button"""
        if not self.realtime_checkbox.isChecked():
            self.match_btn.setEnabled(enabled)

    def set_status(self, message):
        """Set status message"""
        self.status_label.setText(message)

    def is_realtime_enabled(self):
        """Check if real-time mode is enabled"""
        return self.realtime_checkbox.isChecked()

    def trigger_realtime_match_if_enabled(self):
        """Trigger matching if real-time mode is enabled"""
        if self.realtime_checkbox.isChecked():
            self.debounce_timer.start(300)

    def request_hide_boxes(self, hide):
        """Request to hide/show boxes (called from main window)"""
        print(f"[TemplateMatchingWidget] request_hide_boxes: {hide}")
        self.hideBoxesRequested.emit(hide)
