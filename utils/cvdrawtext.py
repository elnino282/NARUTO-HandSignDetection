#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from functools import lru_cache
from PIL import ImageFont, ImageDraw, Image


class CvDrawText:
    def __init__(self):
        pass

    @staticmethod
    @lru_cache(maxsize=64)
    def _font(font_path, font_size):
        return ImageFont.truetype(str(font_path), max(1, int(font_size)))

    @classmethod
    def text_bounds(cls, text, font_path, font_size):
        font = cls._font(font_path, font_size)
        if hasattr(font, 'getbbox'):
            return font.getbbox(text)
        # Pillow 6.x compatibility.
        left, top = font.getoffset(text)
        right, bottom = font.getsize(text)
        return left, top, right, bottom

    @classmethod
    def text_size(cls, text, font_path, font_size):
        left, top, right, bottom = cls.text_bounds(text, font_path, font_size)
        return right - left, bottom - top

    @classmethod
    def fit_text(cls, text, font_path, font_size, width, height):
        """Return a fitting font size, or 0 if the region cannot hold the text."""
        for size in range(max(1, int(font_size)), 0, -1):
            text_width, text_height = cls.text_size(text, font_path, size)
            if text_width <= width and text_height <= height:
                return size
        return 0

    @classmethod
    def history_text(cls, names, font_path, font_size, width):
        """Keep the newest complete names without modifying the recognition queue."""
        visible = list(names)
        shortened = False
        while visible:
            text = ('... ' if shortened else '') + ' \u2192 '.join(visible)
            if cls.text_size(text, font_path, font_size)[0] <= width:
                return text
            if len(visible) == 1:
                return text  # The drawing helper fits this final name if needed.
            visible.pop(0)
            shortened = True
        return ''

    @classmethod
    def puttext_fitted(cls, cv_image, text, region, font_path, font_size,
                       color=(0, 0, 0)):
        """Draw within (x, y, width, height), accounting for font bearings."""
        if not text:
            return cv_image
        x, y, width, height = map(int, region)
        x2 = min(cv_image.shape[1], x + width)
        y2 = min(cv_image.shape[0], y + height)
        x, y = max(0, x), max(0, y)
        width, height = x2 - x, y2 - y
        if width <= 0 or height <= 0:
            return cv_image
        size = cls.fit_text(text, font_path, font_size, width, height)
        if not size:
            return cv_image
        left, top, _, _ = cls.text_bounds(text, font_path, size)
        # A region-sized canvas also contains antialiasing at the edges.
        region_image = cls.puttext(cv_image[y:y2, x:x2], text, (-left, -top),
                                   font_path, size, color)
        cv_image[y:y2, x:x2] = region_image
        return cv_image

    @classmethod
    def puttext(cls,
                cv_image,
                text,
                point,
                font_path,
                font_size,
                color=(0, 0, 0)):
        font = cls._font(font_path, font_size)

        cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)

        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)

        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv.cvtColor(cv_rgb_result_image,
                                          cv.COLOR_RGB2BGR)

        return cv_bgr_result_image
