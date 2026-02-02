#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom QListWidget for label list with keyboard shortcuts
"""

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class LabelListWidget(QListWidget):
    """Custom QListWidget with Delete key and Ctrl+A support"""

    # Signal emitted when delete is requested for selected items
    deleteSelectedRequested = pyqtSignal()

    def __init__(self, parent=None):
        super(LabelListWidget, self).__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure widget can receive focus

        # Create Ctrl+A shortcut that only works when this widget has focus
        self.select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.select_all_shortcut.setContext(Qt.WidgetShortcut)  # Only active when widget has focus
        self.select_all_shortcut.activated.connect(self._select_all)

        # Note: Delete key is handled by labelImg.py's delete action
        # We emit deleteSelectedRequested signal when delete_selected() is called

        print("[LabelListWidget] Initialized with ExtendedSelection mode and Ctrl+A shortcut")

    def _select_all(self):
        """Select all items in the list"""
        print("[LabelListWidget] Ctrl+A shortcut activated - selecting all")
        self.selectAll()

    def delete_selected(self):
        """Called externally to delete selected items"""
        selected = self.selectedItems()
        print(f"[LabelListWidget] delete_selected called - {len(selected)} items selected")
        if selected:
            print("[LabelListWidget] Emitting deleteSelectedRequested signal")
            self.deleteSelectedRequested.emit()
            return True
        return False

    def keyPressEvent(self, event):
        """Handle key press events"""
        key = event.key()
        modifiers = event.modifiers()
        print(f"[LabelListWidget] Key pressed: {key}, modifiers: {int(modifiers)}")

        # Ctrl+A: Select all (backup for shortcut)
        if key == Qt.Key_A and (modifiers & Qt.ControlModifier):
            print("[LabelListWidget] Ctrl+A detected via keyPressEvent - selecting all")
            self.selectAll()
            event.accept()
            return

        # Pass other events to parent
        super(LabelListWidget, self).keyPressEvent(event)
