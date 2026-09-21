"""Regression coverage for the English configuration and display migration."""

import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import warnings
from collections import deque

import numpy as np

from utils.hand_signs import Jutsu, load_labels, load_jutsu, match_jutsu
from utils.cvdrawtext import CvDrawText


ROOT = Path(__file__).resolve().parents[1]
FONT = str(ROOT / 'utils/font/KouzanMouhitsu.ttf')
# Original Japanese sequences converted using their original label row indices.
ORIGINAL_SEQUENCES = [
    (6, 3, 9, 12, 7, 3),
    (6, 8, 9, 12, 7, 3),
    (8, 6, 3),
    (11, 12, 10, 9, 8),
    (3, 6, 5, 11),
    (1, 3, 11, 2, 4, 3),
    (5, 3, 4),
    (3, 6, 11, 5, 13),
    (6, 5, 4, 3),
    (3, 2, 5, 4, 10, 5, 8),
    (6, 12, 8, 4, 11, 1, 10, 7, 6, 13),
    (2, 9, 4, 1, 12, 10, 2, 7, 10, 1, 3, 11, 3, 6, 2, 8, 6, 12,
     8, 1, 15, 9, 10, 5, 10, 2, 7, 8, 3, 6, 1, 9, 4, 12, 5, 8, 1,
     2, 9, 10, 15, 1, 12, 10),
    (8, 7, 6, 5, 1, 2, 3),
    (8, 12, 2, 11, 6),
]
EXPECTED_NAMES = [
    'None', 'Rat', 'Ox', 'Tiger', 'Hare', 'Dragon', 'Snake', 'Horse',
    'Ram', 'Monkey', 'Bird', 'Dog', 'Boar', 'Hand Claps', 'Unknown', 'Mizunoe',
]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.labels = load_labels(ROOT / 'setting/labels.csv')
        self.jutsu = load_jutsu(self.labels, ROOT / 'setting/jutsu.csv')

    def test_model_class_mapping(self):
        self.assertEqual(self.labels, dict(enumerate(EXPECTED_NAMES)))
        for model_class in range(15):
            self.assertEqual(self.labels[model_class + 1], EXPECTED_NAMES[model_class + 1])

    def test_original_sequences_and_order(self):
        self.assertEqual([j.sign_ids for j in self.jutsu], ORIGINAL_SEQUENCES)
        self.assertEqual(len(self.jutsu), 14)
        self.assertEqual(len(self.jutsu[11].sign_ids), 44)
        for index, sequence in enumerate(ORIGINAL_SEQUENCES):
            with self.subTest(index=index):
                self.assertEqual(match_jutsu(deque(sequence, maxlen=44), self.jutsu), index)
        self.assertEqual(self.jutsu[11].display_name, 'Water Style: Water Dragon Jutsu')
        self.assertEqual(self.jutsu[2].display_name, 'Clone Jutsu')

    def test_only_complete_sequences_match(self):
        self.assertIsNone(match_jutsu([], self.jutsu))
        for sequence in ORIGINAL_SEQUENCES:
            for history in (sequence[:-1], (0,) + sequence, sequence + (0,),
                            tuple(reversed(sequence)), (99,)):
                with self.subTest(history=history):
                    self.assertIsNone(match_jutsu(history, self.jutsu))

    def test_first_match_and_display_name_independence(self):
        renamed = Jutsu('Different style', 'Different name', (6, 3, 9, 12, 7, 3))
        self.assertEqual(match_jutsu(renamed.sign_ids, [renamed, self.jutsu[0]]), 0)

    def test_invalid_labels(self):
        cases = [
            ('name,id\nRat,1\n', 'header'),
            ('id,name\n0,None\n0,Rat\n', 'duplicate'),
            ('id,name\n-1,Rat\n', 'negative'),
            ('id,name\n0,None\n2,Ox\n', 'consecutive'),
            ('id,name\nx,Rat\n', 'invalid sign ID'),
            ('id,name\n0,\n', 'expected id,name'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            for contents, message in cases:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, message):
                        load_labels(path)

    def test_invalid_techniques(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'jutsu.csv'
            for contents, message in [
                (',Example,99\n', 'unknown sign ID 99'),
                (',Example,1,\n', 'invalid sign ID'),
                (',Example\n', 'expected style,name,sign_id'),
                ('', 'no jutsu'),
            ]:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, message):
                        load_jutsu(self.labels, path)


class RenderingTests(unittest.TestCase):
    def test_longest_technique_fits_at_both_resolutions(self):
        techniques = load_jutsu(load_labels(ROOT / 'setting/labels.csv'),
                                ROOT / 'setting/jutsu.csv')
        for width, height in ((960, 540), (640, 480)):
            for technique in techniques:
                with self.subTest(width=width, name=technique.display_name):
                    available = (width - 10, height // 10 - 8)
                    size = CvDrawText.fit_text(technique.display_name, FONT,
                                               height // 22, *available)
                    self.assertGreater(size, 0)
                    actual = CvDrawText.text_size(technique.display_name, FONT, size)
                    self.assertLessEqual(actual[0], available[0])
                    self.assertLessEqual(actual[1], available[1])

    def test_history_shows_newest_signs_without_mutation(self):
        history = deque(['Hand Claps', 'Dragon', 'Mizunoe'] * 6, maxlen=18)
        before = list(history)
        for width, height in ((960, 540), (640, 480)):
            text = CvDrawText.history_text(history, FONT, height // 22, width - 10)
            self.assertTrue(text.startswith('... '))
            self.assertTrue(text.endswith('Mizunoe'))
            self.assertIn(' \u2192 ', text)
            self.assertLessEqual(CvDrawText.text_size(text, FONT, height // 22)[0], width - 10)
            self.assertEqual(list(history), before)

    def test_text_is_contained_in_small_or_clipped_regions(self):
        for region in ((10, 10, 30, 12), (-5, -4, 60, 30), (70, 70, 60, 30)):
            canvas = np.zeros((80, 100, 3), dtype=np.uint8)
            result = CvDrawText.puttext_fitted(canvas, 'Hand Claps', region,
                                               FONT, 40, (255, 255, 255))
            ys, xs = np.nonzero(np.any(result, axis=2))
            self.assertGreater(len(xs), 0)
            x, y, width, height = region
            self.assertGreaterEqual(xs.min(), max(0, x))
            self.assertGreaterEqual(ys.min(), max(0, y))
            self.assertLess(xs.max(), min(100, x + width))
            self.assertLess(ys.max(), min(80, y + height))
        canvas = np.zeros((80, 100, 3), dtype=np.uint8)
        self.assertFalse(CvDrawText.puttext_fitted(
            canvas, 'Tiger', (5, 5, 0, 0), FONT, 20).any())


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ninjutsu = importlib.import_module('Ninjutsu_demo')

    def test_legacy_language_cli(self):
        for arguments, expected in (([], None), (['--use_jutsu_lang_en'], True),
                                    (['--use_jutsu_lang_en', 'True'], True),
                                    (['--use_jutsu_lang_en', 'False'], False)):
            with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['demo'] + arguments):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    args = self.ninjutsu.get_args()
                self.assertEqual(args.use_jutsu_lang_en, expected)
                self.assertEqual(args.fps, 30)
                self.assertEqual(len(caught), int(bool(arguments)))
                if caught:
                    self.assertIn('always uses English', str(caught[0].message))
        with patch.object(sys, 'argv', ['demo', '--use_jutsu_lang_en', 'invalid']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.ninjutsu.get_args()

    def test_match_timing_is_preserved(self):
        jutsu = load_jutsu(load_labels())
        with patch.object(self.ninjutsu.time, 'time', return_value=123):
            for index, sequence in enumerate(ORIGINAL_SEQUENCES):
                self.assertEqual(self.ninjutsu.check_jutsu(sequence, jutsu, 0, 0), (index, 123))
            self.assertEqual(self.ninjutsu.check_jutsu([], jutsu, 2, 50), (2, 50))

    def test_render_full_frames(self):
        labels = load_labels()
        jutsu = load_jutsu(labels)
        for width, height in ((960, 540), (640, 480)):
            for active in (False, True):
                with self.subTest(width=width, active=active):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    with patch.object(self.ninjutsu.time, 'time', return_value=100):
                        result = self.ninjutsu.draw_debug_image(
                            frame, FONT, 30, labels, [[10, 10, 45, 40]], [0.99], [12],
                            0.7, False, True, jutsu, deque([13, 5, 15] * 6), 5, 7,
                            99 if active else 0)
                    self.assertEqual(result.shape, (height + height // 18 + height // 10, width, 3))
                    self.assertTrue(result[-height // 10:].any())

    def test_all_demo_loops_consume_new_labels(self):
        # Synthetic input/detections exercise each complete loop without camera access.
        for name in ('simple_demo', 'simple_demo_without_post', 'Ninjutsu_demo'):
            with self.subTest(name=name), contextlib.ExitStack() as stack:
                module = importlib.import_module(name)
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
                camera = Mock()
                camera.read.return_value = (True, frame)
                camera.get.return_value = 30
                stack.enter_context(patch.object(sys, 'argv', [name]))
                stack.enter_context(patch.object(module.cv, 'VideoCapture', return_value=camera))
                stack.enter_context(patch.object(module.cv, 'VideoWriter'))
                stack.enter_context(patch.object(module.cv, 'imshow'))
                stack.enter_context(patch.object(module.cv, 'destroyAllWindows'))
                # The first iteration draws a detection; the second exits.
                stack.enter_context(patch.object(module.cv, 'waitKey', side_effect=[-1, 27]))
                stack.enter_context(patch.object(module.time, 'sleep'))
                model = stack.enter_context(patch.object(module, 'YoloxONNX'))
                model.return_value.inference.return_value = (
                    np.array([[10, 10, 100, 100]]), np.array([0.99]), np.array([12]))
                module.main()
                camera.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
