from math import sqrt
from libs.ustr import ustr
import hashlib
import re
import sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    QT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    QT5 = False


def new_icon(icon):
    return QIcon(':/' + icon)


def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class Struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


# 클래스 인덱스 기반 색상 (빨주노초파남보 순서) - 연한 버전
CLASS_COLORS = [
    (255, 100, 100),   # 빨강 (연한)
    (255, 180, 100),   # 주황 (연한)
    (255, 255, 100),   # 노랑 (연한)
    (100, 255, 100),   # 초록 (연한)
    (100, 180, 255),   # 파랑 (연한)
    (100, 100, 255),   # 남색 (연한)
    (200, 100, 255),   # 보라 (연한)
]

# 클래스 목록을 저장할 전역 변수
_class_list = []

def set_class_list(class_list):
    """클래스 목록 설정 (labelImg.py에서 호출)"""
    global _class_list
    _class_list = list(class_list) if class_list else []

def get_color_for_class(class_name, alpha=100):
    """클래스 이름에 따른 색상 반환 (인덱스 기반)"""
    global _class_list
    if class_name in _class_list:
        idx = _class_list.index(class_name)
        r, g, b = CLASS_COLORS[idx % len(CLASS_COLORS)]
        return QColor(r, g, b, alpha)
    # 클래스 목록에 없으면 해시 기반 색상 사용
    return generate_color_by_text_hash(class_name, alpha)

def generate_color_by_text_hash(text, alpha=100):
    """텍스트 해시 기반 색상 생성 (기존 방식)"""
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, alpha)

def generate_color_by_text(text):
    """라벨 리스트 배경색용 (연한 색상)"""
    return get_color_for_class(text, alpha=100)

def get_line_color_for_class(class_name):
    """바운딩 박스용 색상 (더 진한 색상)"""
    return get_color_for_class(class_name, alpha=200)


def have_qstring():
    """p3/qt5 get rid of QString wrapper as py3 has native unicode str type"""
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


# QT4 has a trimmed method, in QT5 this is called strip
if QT5:
    def trimmed(text):
        return text.strip()
else:
    def trimmed(text):
        return text.trimmed()
