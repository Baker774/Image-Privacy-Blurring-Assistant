import re

from rapidocr import RapidOCR


class SensitiveTextDetector:
    def __init__(self):
        # 首次实际扫描时才会加载 OCR 模型。
        self.ocr = RapidOCR()

    def detect(self, image):
        """
        接收 Pillow 图片，返回疑似敏感信息列表。
        每项包含：类别、脱敏展示文本、置信度、文字区域坐标。
        """
        result = self.ocr(image)

        if result.boxes is None or result.txts is None:
            return []

        candidates = []

        for box, text, score in zip(
            result.boxes,
            result.txts,
            result.scores,
        ):
            text = str(text).strip()
            score = float(score)

            # 置信度太低的文本不作为候选，减少误报。
            if score < 0.75:
                continue

            sensitive_result = self.extract_sensitive_text(text)

            if sensitive_result is None:
                continue

            sensitive_type, sensitive_text = sensitive_result

            left = int(min(point[0] for point in box))
            top = int(min(point[1] for point in box))
            right = int(max(point[0] for point in box))
            bottom = int(max(point[1] for point in box))

            candidates.append(
                {
                    "type": sensitive_type,
                    "raw_text": text,
                    "text": sensitive_text,
                    "masked_text": self.mask_text(sensitive_text, sensitive_type),
                    "score": score,
                    "box": (left, top, right, bottom),
                }
            )

        return candidates

    def extract_sensitive_text(self, text):
        """
        从整段 OCR 文本中提取敏感信息。
        例如“手机号：13800138000”会提取为“13800138000”。
        """
        compact_text = re.sub(r"[\s-]", "", text)

        phone_match = re.search(
            r"(?<!\d)(?:\+?86)?1[3-9]\d{9}(?!\d)",
            compact_text,
        )
        if phone_match:
            return "手机号", phone_match.group()

        email_match = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text,
        )
        if email_match:
            return "邮箱", email_match.group()

        id_match = re.search(
            r"(?<!\d)\d{17}[\dXx](?!\d)",
            compact_text,
        )
        if id_match:
            return "身份证号", id_match.group()

        return None

    def mask_text(self, text, sensitive_type):
        """仅用于界面显示，避免在扫描结果里完整展示敏感信息。"""
        normalized = re.sub(r"[\s-]", "", text)

        if sensitive_type == "手机号":
            return f"{normalized[:3]}****{normalized[-4:]}"

        if sensitive_type == "邮箱":
            name, domain = text.split("@", maxsplit=1)
            return f"{name[:1]}***@{domain}"

        if sensitive_type == "身份证号":
            return f"{normalized[:6]}********{normalized[-4:]}"

        return "已识别敏感信息"