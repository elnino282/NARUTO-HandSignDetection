#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils import CvDrawText
from utils.hand_signs import load_labels, load_jutsu, match_jutsu
from model.yolox.yolox_onnx import YoloxONNX


def legacy_language_value(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('expected True or False')
    return value.lower() == 'true'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default='model/yolox/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.7,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)

    parser.add_argument("--use_display_score", type=bool, default=False)
    parser.add_argument("--erase_bbox", type=bool, default=False)
    parser.add_argument(
        "--use_jutsu_lang_en", nargs='?', const=True, default=None,
        type=legacy_language_value,
        help='Deprecated compatibility option; the interface always uses English.',
    )

    parser.add_argument("--chattering_check", type=int, default=1)

    parser.add_argument("--use_fullscreen", type=bool, default=False)

    args = parser.parse_args()
    if args.use_jutsu_lang_en is not None:
        warnings.warn('--use_jutsu_lang_en is deprecated; the interface always uses English.',
                      FutureWarning, stacklevel=2)

    return args


def main():
    # Parse arguments #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height
    cap_device = args.device
    if args.file is not None:  # Use a video file when supplied
        cap_device = args.file

    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    sign_interval = args.sign_interval
    jutsu_display_time = args.jutsu_display_time

    use_display_score = args.use_display_score
    erase_bbox = args.erase_bbox

    chattering_check = args.chattering_check

    use_fullscreen = args.use_fullscreen

    # Prepare camera ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model ############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        # providers=['CPUExecutionProvider'],
    )

    # FPS measurement #########################################################
    cvFpsCalc = CvFpsCalc()

    # Load font ##########################################################
    # https://opentype.jp/kouzanmouhitufont.htm
    font_path = './utils/font/KouzanMouhitsu.ttf'

    # Load labels ###########################################################
    labels = load_labels()
    jutsu = load_jutsu(labels)

    # Display and recognition histories ##############################################
    sign_max_display = 18
    sign_max_history = 44
    sign_display_queue = deque(maxlen=sign_max_display)
    sign_history_queue = deque(maxlen=sign_max_history)

    chattering_check_queue = deque(maxlen=chattering_check)
    for index in range(-1, -1 - chattering_check, -1):
        chattering_check_queue.append(index)

    # Initialize state #########################################################
    sign_interval_start = 0  # Start of sign interval
    jutsu_index = 0  # Displayed jutsu index
    jutsu_start_time = 0  # Start of jutsu display
    frame_count = 0  # Frame counter

    window_name = 'NARUTO HandSignDetection Ninjutsu Demo'
    if use_fullscreen:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        start_time = time.time()

        # Capture frame #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        debug_image = copy.deepcopy(frame)

        if (frame_count % (skip_frame + 1)) != 0:
            continue

        # Measure FPS ##############################################################
        fps_result = cvFpsCalc.get()

        # Detect hand signs #############################################################
        bboxes, scores, class_ids = yolox.inference(frame)

        # Update detection history ####################################################
        for _, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # Discard results below the confidence threshold
            if score < score_th:
                continue

            # Require consecutive detections of the same sign to suppress flicker
            chattering_check_queue.append(class_id)
            if len(set(chattering_check_queue)) != 1:
                continue

            # Queue only signs different from the previous sign
            if len(sign_display_queue
                   ) == 0 or sign_display_queue[-1] != class_id:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                sign_interval_start = time.time()  # Time of last sign detection

        # Clear history after the sign interval expires ####################
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()

        # Match a jutsu #########################################################
        jutsu_index, jutsu_start_time = check_jutsu(
            sign_history_queue,
            jutsu,
            jutsu_index,
            jutsu_start_time,
        )

        # Handle keyboard input ###########################################################
        key = cv.waitKey(1)
        if key == 99:  # C: clear sign history
            sign_display_queue.clear()
            sign_history_queue.clear()
        if key == 27:  # ESC: exit
            break

        # Render frame #############################################################
        debug_image = draw_debug_image(
            debug_image,
            font_path,
            fps_result,
            labels,
            bboxes,
            scores,
            class_ids,
            score_th,
            erase_bbox,
            use_display_score,
            jutsu,
            sign_display_queue,
            jutsu_display_time,
            jutsu_index,
            jutsu_start_time,
        )
        if use_fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN,
                                 cv.WINDOW_FULLSCREEN)
        cv.imshow(window_name, debug_image)
        # cv.moveWindow(window_name, 100, 100)

        # Limit FPS #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()


def check_jutsu(sign_history_queue, jutsu, jutsu_index, jutsu_start_time):
    match_index = match_jutsu(sign_history_queue, jutsu)
    if match_index is not None:
        return match_index, time.time()
    return jutsu_index, jutsu_start_time


def draw_debug_image(
    debug_image,
    font_path,
    fps_result,
    labels,
    bboxes,
    scores,
    class_ids,
    score_th,
    erase_bbox,
    use_display_score,
    jutsu,
    sign_display_queue,
    jutsu_display_time,
    jutsu_index,
    jutsu_start_time,
):
    frame_width, frame_height = debug_image.shape[1], debug_image.shape[0]

    # Draw bounding boxes when enabled ###################
    if not erase_bbox:
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # Discard boxes below the confidence threshold
            if score < score_th:
                continue

            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            # Draw a square using the longer side of the bounding box
            x_len = x2 - x1
            y_len = y2 - y1
            square_len = x_len if x_len >= y_len else y_len
            square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
            square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
            square_x2 = square_x1 + square_len
            square_y2 = square_y1 + square_len
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (255, 255, 255), 4)
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (0, 0, 0), 2)

            # Fit the English name inside the lower half of the visible box.
            left, top = max(0, square_x1), max(0, square_y1)
            right, bottom = min(frame_width, square_x2), min(frame_height, square_y2)
            box_width, box_height = right - left, bottom - top
            label_top = top + box_height // 2
            debug_image = CvDrawText.puttext_fitted(
                debug_image, labels[class_id],
                (left + 3, label_top, box_width - 6, bottom - label_top - 3),
                font_path, max(1, square_len // 2), (185, 0, 0))

            if use_display_score:
                debug_image = CvDrawText.puttext_fitted(
                    debug_image, '{:.3f}'.format(score),
                    (left + 3, top + 3, box_width - 6, box_height // 2 - 3),
                    font_path, max(1, square_len // 8), (185, 0, 0))

    # Create the FPS header and fit text to the available height.
    header_image = np.zeros((max(1, int(frame_height / 18)), frame_width, 3), np.uint8)
    header_image = CvDrawText.puttext_fitted(
        header_image, "FPS: " + str(fps_result),
        (5, 2, frame_width - 10, header_image.shape[0] - 4),
        font_path, max(1, int(frame_height / 20)), (255, 255, 255))

    footer_image = np.zeros((max(1, int(frame_height / 10)), frame_width, 3), np.uint8)
    text_width, text_height = frame_width - 10, footer_image.shape[0] - 8
    font_size = max(1, int(frame_height / 22))
    if (time.time() - jutsu_start_time) < jutsu_display_time:
        footer_text = jutsu[jutsu_index].display_name
    else:
        footer_text = CvDrawText.history_text(
            [labels[sign_id] for sign_id in sign_display_queue],
            font_path, font_size, text_width)
    footer_image = CvDrawText.puttext_fitted(
        footer_image, footer_text, (5, 4, text_width, text_height),
        font_path, font_size, (255, 255, 255))

    # Attach the header and footer ######################################
    debug_image = cv.vconcat([header_image, debug_image])
    debug_image = cv.vconcat([debug_image, footer_image])

    return debug_image


if __name__ == '__main__':
    main()
