import sys
import cv2
import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QSlider,
    QComboBox,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sensitive_detector import SensitiveTextDetector

class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__()

        self.original_pixmap = None
        self.start_point = None
        self.selection_rect = None

        # OCR 候选项与已确认的候选项编号。
        self.candidates = []
        self.selected_candidate_indexes = set()

        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.clear_selection()
        self.clear_candidates()

    def set_candidates(self, candidates):
        """接收 OCR 候选项；默认均不选中，等待用户确认。"""
        self.candidates = candidates
        self.selected_candidate_indexes = set()
        self.update()

    def clear_candidates(self):
        self.candidates = []
        self.selected_candidate_indexes = set()
        self.update()

    def get_selected_candidates(self):
        return [
            candidate
            for index, candidate in enumerate(self.candidates)
            if index in self.selected_candidate_indexes
        ]

    def clear_selection(self):
        self.start_point = None
        self.selection_rect = None
        self.update()

    def get_image_rect(self):
        """计算图片在当前画布中实际显示的位置和大小。"""
        if self.original_pixmap is None:
            return QRect()

        image_size = self.original_pixmap.size()
        image_size.scale(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        left = (self.width() - image_size.width()) // 2
        top = (self.height() - image_size.height()) // 2

        return QRect(QPoint(left, top), image_size)

    def get_selection_in_image_coordinates(self):
        """将手动选框坐标转换为原始图片坐标。"""
        if self.original_pixmap is None or self.selection_rect is None:
            return None

        image_rect = self.get_image_rect()
        selected_rect = self.selection_rect.intersected(image_rect)

        if selected_rect.isEmpty():
            return None

        original_width = self.original_pixmap.width()
        original_height = self.original_pixmap.height()

        scale_x = original_width / image_rect.width()
        scale_y = original_height / image_rect.height()

        left = int((selected_rect.left() - image_rect.left()) * scale_x)
        top = int((selected_rect.top() - image_rect.top()) * scale_y)
        right = int((selected_rect.right() + 1 - image_rect.left()) * scale_x)
        bottom = int((selected_rect.bottom() + 1 - image_rect.top()) * scale_y)

        return left, top, right, bottom

    def get_candidate_display_rect(self, candidate):
        """将 OCR 原图坐标换算为画布中的显示坐标。"""
        if self.original_pixmap is None:
            return QRect()

        left, top, right, bottom = candidate["box"]
        image_rect = self.get_image_rect()

        scale_x = image_rect.width() / self.original_pixmap.width()
        scale_y = image_rect.height() / self.original_pixmap.height()

        display_left = image_rect.left() + int(left * scale_x)
        display_top = image_rect.top() + int(top * scale_y)
        display_right = image_rect.left() + int(right * scale_x)
        display_bottom = image_rect.top() + int(bottom * scale_y)

        return QRect(
            display_left,
            display_top,
            max(1, display_right - display_left),
            max(1, display_bottom - display_top),
        )

    def find_candidate_at(self, position):
        """返回鼠标位置命中的候选项编号；未命中则返回 None。"""
        for index, candidate in enumerate(self.candidates):
            if self.get_candidate_display_rect(candidate).contains(position):
                return index

        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f9fafb"))

        if self.original_pixmap is None:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "请点击上方“选择图片”导入 PNG、JPG 或 JPEG 图片",
            )
            return

        image_rect = self.get_image_rect()
        painter.drawPixmap(image_rect, self.original_pixmap)

        # 绘制 OCR 候选框。
        for index, candidate in enumerate(self.candidates):
            rect = self.get_candidate_display_rect(candidate)
            is_selected = index in self.selected_candidate_indexes

            if is_selected:
                color = QColor("#16a34a")
                fill_color = QColor(22, 163, 74, 50)
            else:
                color = QColor("#f97316")
                fill_color = QColor(249, 115, 22, 45)

            painter.setPen(QPen(color, 3))
            painter.setBrush(fill_color)
            painter.drawRect(rect)

            painter.setPen(color)
            painter.drawText(
                rect.adjusted(3, 3, -3, -3),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                candidate["type"],
            )

        # 绘制手动框选区域。
        if self.selection_rect is not None:
            painter.setPen(QPen(QColor("#ef4444"), 3))
            painter.setBrush(QColor(239, 68, 68, 50))
            painter.drawRect(self.selection_rect)

    def mousePressEvent(self, event: QMouseEvent):
        if self.original_pixmap is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position().toPoint()

            # 优先处理 OCR 候选框点击。
            candidate_index = self.find_candidate_at(position)

            if candidate_index is not None:
                if candidate_index in self.selected_candidate_indexes:
                    self.selected_candidate_indexes.remove(candidate_index)
                else:
                    self.selected_candidate_indexes.add(candidate_index)

                self.update()
                return

            image_rect = self.get_image_rect()

            if image_rect.contains(position):
                self.start_point = position
                self.selection_rect = QRect(position, position)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.start_point is None:
            return

        image_rect = self.get_image_rect()
        position = event.position().toPoint()

        x = max(image_rect.left(), min(position.x(), image_rect.right()))
        y = max(image_rect.top(), min(position.y(), image_rect.bottom()))
        current_point = QPoint(x, y)

        self.selection_rect = QRect(
            self.start_point,
            current_point,
        ).normalized()

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图片隐私打码助手")
        self.resize(1100, 800)

        # original_image 永远保存原图；working_image 才会被修改。
        self.original_image = None
        self.working_image = None
        self.history = []
        self.text_detector = None
        self.sensitive_candidates = []

        self.open_button = QPushButton("选择图片")
        self.open_button.setFixedHeight(42)
        self.open_button.clicked.connect(self.open_image)

        self.mask_mode_combo = QComboBox()
        self.mask_mode_combo.addItem("模糊", "blur")
        self.mask_mode_combo.addItem("马赛克", "mosaic")
        self.mask_mode_combo.addItem("黑色遮挡", "black")
        self.mask_mode_combo.setFixedHeight(42)

        self.blur_button = QPushButton("应用打码")
        self.blur_button.setFixedHeight(42)
        self.blur_button.clicked.connect(self.apply_blur)

        self.qr_button = QPushButton("自动遮挡二维码")
        self.qr_button.setFixedHeight(42)
        self.qr_button.clicked.connect(self.mask_qr_code)

        self.face_button = QPushButton("自动模糊人脸")
        self.face_button.setFixedHeight(42)
        self.face_button.clicked.connect(self.blur_faces)

        self.scan_text_button = QPushButton("扫描敏感文字")
        self.scan_text_button.setFixedHeight(42)
        self.scan_text_button.clicked.connect(self.scan_sensitive_text)

        self.mask_candidates_button = QPushButton("遮挡已选候选项")
        self.mask_candidates_button.setFixedHeight(42)
        self.mask_candidates_button.clicked.connect(
            self.mask_selected_candidates
        )

        self.clear_button = QPushButton("清除选区")
        self.clear_button.setFixedHeight(42)
        self.clear_button.clicked.connect(self.clear_selection)

        self.undo_button = QPushButton("撤销上一步")
        self.undo_button.setFixedHeight(42)
        self.undo_button.clicked.connect(self.undo_last_action)

        self.reset_button = QPushButton("恢复原图")
        self.reset_button.setFixedHeight(42)
        self.reset_button.clicked.connect(self.reset_to_original)

        self.save_button = QPushButton("导出打码副本")
        self.save_button.setFixedHeight(42)
        self.save_button.clicked.connect(self.save_image)

        self.strength_label = QLabel("打码强度：")
        self.strength_value_label = QLabel("18")

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(1, 50)
        self.strength_slider.setValue(18)
        self.strength_slider.valueChanged.connect(self.update_strength_label)

        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_slider, stretch=1)
        strength_layout.addWidget(self.strength_value_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.mask_mode_combo)
        button_layout.addWidget(self.blur_button)
        button_layout.addWidget(self.qr_button)
        button_layout.addWidget(self.face_button)
        button_layout.addWidget(self.scan_text_button)
        button_layout.addWidget(self.mask_candidates_button)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)

        self.file_name_label = QLabel("暂未选择图片")
        self.file_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_canvas = ImageCanvas()
        self.image_canvas.setStyleSheet(
            """
            QWidget {
                border: 2px dashed #9ca3af;
                border-radius: 10px;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addLayout(button_layout)
        layout.addLayout(strength_layout)
        layout.addWidget(self.file_name_label)
        layout.addWidget(self.image_canvas, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def pil_image_to_pixmap(self, image):
        """将 Pillow 图片转换为 PySide6 可显示的 QPixmap。"""
        rgba_image = image.convert("RGBA")
        image_data = rgba_image.tobytes("raw", "RGBA")

        qimage = QImage(
            image_data,
            rgba_image.width,
            rgba_image.height,
            rgba_image.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()

        return QPixmap.fromImage(qimage)

    def open_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择需要打码的图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg)",
        )

        if not image_path:
            return

        try:
            self.original_image = Image.open(image_path).convert("RGBA")
            self.working_image = self.original_image.copy()
            self.history = []
        except OSError:
            QMessageBox.warning(self, "读取失败", "这张图片无法读取，请换一张图片尝试。")
            return

        pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(pixmap)
        self.file_name_label.setText(f"当前图片：{Path(image_path).name}")

    def apply_blur(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        selection = self.image_canvas.get_selection_in_image_coordinates()

        if selection is None:
            QMessageBox.information(self, "提示", "请先在图片上拖动鼠标，框选需要模糊的区域。")
            return

        left, top, right, bottom = selection

        if right <= left or bottom <= top:
            QMessageBox.information(self, "提示", "选区太小，请重新框选。")
            return

        selected_area = self.working_image.crop((left, top, right, bottom))
        mask_mode = self.mask_mode_combo.currentData()
        strength = self.strength_slider.value()

        if mask_mode == "blur":
            # radius 越大，模糊程度越强。
            processed_area = selected_area.filter(
                ImageFilter.GaussianBlur(radius=strength)
            )

        elif mask_mode == "mosaic":
            # 先缩小图片，再用最近邻算法放大，形成马赛克效果。
            block_size = max(2, strength * 2)

            small_width = max(1, selected_area.width // block_size)
            small_height = max(1, selected_area.height // block_size)

            tiny_area = selected_area.resize(
            (small_width, small_height),
            Image.Resampling.NEAREST,
            )

            processed_area = tiny_area.resize(
                selected_area.size,
                Image.Resampling.NEAREST,
            )

        elif mask_mode == "black":
            processed_area = Image.new(
                "RGBA",
                selected_area.size,
                (0, 0, 0, 255),
            )

        self.history.append(self.working_image.copy())
        self.working_image.paste(processed_area, (left, top))

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.file_name_label.setText("已完成打码。可继续框选其他区域。")

    def update_strength_label(self, value):
        self.strength_value_label.setText(str(value))
    def clear_selection(self):
        self.image_canvas.clear_selection()
    def scan_sensitive_text(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        try:
            # 延迟创建，避免打开程序时就加载 OCR 模型。
            if self.text_detector is None:
                self.text_detector = SensitiveTextDetector()

            self.sensitive_candidates = self.text_detector.detect(
                self.working_image
            )
        except Exception as error:
            QMessageBox.warning(
                self,
                "扫描失败",
                f"无法完成文字识别：\n{error}",
            )
            return

        self.image_canvas.set_candidates(self.sensitive_candidates)
        if not self.sensitive_candidates:
            QMessageBox.information(
                self,
                "扫描结果",
                "未发现格式明确的手机号、邮箱或身份证号。\n"
                "这不代表图片中绝对不存在敏感信息，请继续人工检查。",
            )
            return

        result_lines = []

        for item in self.sensitive_candidates:
            result_lines.append(
                f"{item['type']}：{item['masked_text']} "
                f"（置信度 {item['score']:.0%}）"
            )

        QMessageBox.information(
            self,
            "扫描结果",
            "以下内容已用橙色框标在图片中。"
            "点击候选框可选中或取消选择，绿色表示已选中：\n\n"
            + "\n".join(result_lines),
        )
    def get_partial_mask_box(self, candidate):
        """
        尽量只遮挡敏感文本中间部分，保留“手机号：”等说明文字，
        以及手机号前 3 位、后 4 位等可展示信息。

        如果无法确定文本在整行中的位置，为保护隐私，会回退为遮挡整行。
        """
        left, top, right, bottom = candidate["box"]

        raw_text = candidate.get("raw_text", "")
        sensitive_text = candidate["text"]
        sensitive_type = candidate["type"]

        start_index = raw_text.find(sensitive_text)

        # 无法精确定位时，宁可全部遮挡，也不要漏掉隐私信息。
        if start_index == -1 or not raw_text:
            return left, top, right, bottom

        if sensitive_type == "手机号":
            # 保留前 3 位和后 4 位，例如 138****8000。
            hidden_start = start_index + 3
            hidden_end = start_index + len(sensitive_text) - 4

        elif sensitive_type == "身份证号":
            # 保留前 6 位和后 4 位。
            hidden_start = start_index + 6
            hidden_end = start_index + len(sensitive_text) - 4

        elif sensitive_type == "邮箱":
            # 保留邮箱首字符和 @ 后面的域名。
            at_position = sensitive_text.find("@")

            if at_position <= 1:
                return left, top, right, bottom

            hidden_start = start_index + 1
            hidden_end = start_index + at_position

        else:
            return left, top, right, bottom

        # OCR 坐标通常是一整行文字，按字符比例估算敏感部分位置。
        total_characters = len(raw_text)
        line_width = right - left

        mask_left = left + int(line_width * hidden_start / total_characters)
        mask_right = left + int(line_width * hidden_end / total_characters)

        return mask_left, top, mask_right, bottom
    def mask_selected_candidates(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        selected_candidates = self.image_canvas.get_selected_candidates()

        if not selected_candidates:
            QMessageBox.information(
                self,
                "提示",
                "请先点击橙色候选框，选中需要遮挡的内容。",
            )
            return

        # 所有候选项作为一次操作，可整体撤销。
        self.history.append(self.working_image.copy())
        draw = ImageDraw.Draw(self.working_image)

        for candidate in selected_candidates:
            left, top, right, bottom = self.get_partial_mask_box(candidate)

            # 给识别框增加一点边距，避免遗漏文字边缘。
            padding = 3
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(self.working_image.width, right + padding)
            bottom = min(self.working_image.height, bottom + padding)

            draw.rectangle(
                (left, top, right, bottom),
                fill=(0, 0, 0, 255),
            )

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.sensitive_candidates = []
        self.file_name_label.setText("已遮挡选中的敏感文字候选项。")
    def blur_faces(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        # OpenCV 的人脸检测需要灰度图片。
        rgb_image = np.array(self.working_image.convert("RGB"))
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        face_detector = cv2.CascadeClassifier(cascade_path)

        faces = face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
        )

        if len(faces) == 0:
            QMessageBox.information(
                self,
                "未发现人脸",
                "没有识别到清晰的正脸。你仍可以手动框选并打码。",
            )
            eturn

        confirmation = QMessageBox.question(
            self,
            "确认自动模糊",
            f"检测到 {len(faces)} 张人脸，是否自动模糊？\n"
            "操作后仍可点击“撤销上一步”恢复。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        self.history.append(self.working_image.copy())

        for face in faces:
            # OpenCV 返回 NumPy 数值；转成普通 int，供 Pillow 使用。
            x, y, face_width, face_height = [
                int(value)
                for value in face
            ]

            # 适度扩大范围，避免只模糊到脸部中央。
            padding = int(max(face_width, face_height) * 0.15)

            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(
                self.working_image.width,
                x + face_width + padding,
            )
            bottom = min(
                self.working_image.height,
                y + face_height + padding,
            )

            face_area = self.working_image.crop(
                (left, top, right, bottom)
            )

            blur_radius = int(
                max(8, min(face_width, face_height) // 5)
            )

            blurred_face = face_area.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )

            self.working_image.paste(blurred_face, (left, top))

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.sensitive_candidates = []
        self.file_name_label.setText(
            f"已自动模糊 {len(faces)} 张人脸。"
        )
    def mask_qr_code(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        # OpenCV 使用 BGR 格式；Pillow 使用 RGB 格式，因此需要转换。
        rgb_image = np.array(self.working_image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        detector = cv2.QRCodeDetector()
        _, points, _ = detector.detectAndDecode(bgr_image)

        if points is None:
            QMessageBox.information(
                self,
                "未发现二维码",
                "没有在当前图片中识别到二维码。你仍可以手动框选并打码。",
            )
            return

        # points 是二维码四个角的坐标，取其最小/最大值即可得到包围框。
        points = np.squeeze(points)

        left = max(0, int(points[:, 0].min()) - 8)
        top = max(0, int(points[:, 1].min()) - 8)
        right = min(self.working_image.width, int(points[:, 0].max()) + 8)
        bottom = min(self.working_image.height, int(points[:, 1].max()) + 8)

        # 保存历史，因此用户可点击“撤销上一步”恢复。
        self.history.append(self.working_image.copy())

        draw = ImageDraw.Draw(self.working_image)
        draw.rectangle(
            (left, top, right, bottom),
            fill=(0, 0, 0, 255),
        )

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.file_name_label.setText("已自动遮挡检测到的二维码。")
    def undo_last_action(self):
        if not self.history:
            QMessageBox.information(self, "提示", "当前没有可撤销的操作。")
            return

        self.working_image = self.history.pop()

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.file_name_label.setText("已撤销上一次打码操作。")
    def reset_to_original(self):
        if self.original_image is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        confirmation = QMessageBox.question(
            self,
            "确认恢复原图",
            "这会清除当前图片上的所有打码和模糊效果，是否继续？",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        self.working_image = self.original_image.copy()
        self.history = []
        self.sensitive_candidates = []

        updated_pixmap = self.pil_image_to_pixmap(self.working_image)
        self.image_canvas.set_image(updated_pixmap)

        self.file_name_label.setText("已恢复为原图。")
    def save_image(self):
        if self.working_image is None:
            QMessageBox.information(self, "提示", "请先选择并处理一张图片。")
            return

        default_path = str(Path.cwd() / "masked_image.png")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出打码后的图片",
            default_path,
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)",
        )

        if not save_path:
            return

        output_path = Path(save_path)

        # 用户未输入扩展名时，默认保存为 PNG。
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")

        image_to_save = self.working_image

        # JPEG 不支持透明通道，保存前必须转为 RGB。
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            image_to_save = self.working_image.convert("RGB")

        try:
            image_to_save.save(output_path)
        except OSError as error:
            QMessageBox.warning(self, "导出失败", f"无法保存图片：\n{error}")
            return

        self.file_name_label.setText(f"已导出：{output_path.name}")
        QMessageBox.information(
            self,
            "导出成功",
            f"打码副本已保存到：\n{output_path}",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())