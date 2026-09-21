# NARUTO-HandSignDetection

Deep Sharingan: Naruto hand-sign recognition using YOLOX object detection.
This repository provides trained models and Python demos for recognizing hand signs
and matching their sequences to ninjutsu from <span id="cite_ref-2">Naruto</span> [2](#cite_note-2).
The current YOLOX demos, configuration files, and documentation use English.
The original SSD/EfficientDet demos remain unchanged in [`_legacy/v2`](./_legacy/v2).

<img src="https://user-images.githubusercontent.com/37477845/95489944-78d5ed00-09d2-11eb-96f6-a687b012c413.gif" width="45%"> <img src="https://user-images.githubusercontent.com/37477845/95645297-97360880-0af8-11eb-9134-d92cbfb5fe42.gif" width="40%">

Right image: © NARUTO Episode 9, *Kakashi, Sharingan Warrior!*,
Masashi Kishimoto / Shueisha / Studio Pierrot.
The original demonstration omits bounding-box overlays on this image because the
author considered them a possible modification under
<span id="cite_ref-1">Article 20 of Japan's Copyright Act</span> [1](#cite_note-1).
These historical images may still contain Japanese text; they do not show the new English interface.

## Overview

Most ninjutsu require a sequence of hand signs. Different nature transformations
also feature characteristic signs, such as Tiger for Fire Style and Boar for Earth Style.
The demos detect individual signs and display a technique when the complete sign
history matches a configured sequence. YOLOX-Nano replaces the earlier
EfficientDet-D0 implementation to improve inference speed.

## Requirements

- ONNX Runtime 1.10.0 or later
- OpenCV 3.4.2 or later
- NumPy (used by the model wrappers)
- Pillow 6.1.0 or later, for `Ninjutsu_demo.py`
- TensorFlow 2.3.0 or later, only for legacy SSD/EfficientDet demos or the post-processing conversion workflow

Install the current demo dependencies in your Python environment:

```bash
python -m pip install onnxruntime opencv-python numpy Pillow
```

## Usage

Run from the repository root so relative model, CSV, and font paths resolve:

```bash
python simple_demo.py
python simple_demo_without_post.py
python Ninjutsu_demo.py
python Ninjutsu_demo.py --file path/to/video.mp4
```

Press **Esc** to exit. In the ninjutsu demo, press **c** to clear the hand-sign history.
English names are always used. Ninjutsu appear as `Fire Style: Fireball Jutsu`;
history entries are separated by arrows. If the history does not fit, the footer
shows `...` followed by the newest signs. Recognition still uses the full history
(up to 44 signs), independently of the display queue (up to 18 signs).

### Demo files

| File | Purpose | Default model |
| --- | --- | --- |
| `simple_demo.py` | Detection with Python post-processing | `model/yolox/yolox_nano.onnx` |
| `simple_demo_without_post.py` | Detection with post-processing embedded in ONNX; writes `output.mp4` | `model/yolox/yolox_nano_with_post.onnx` |
| `Ninjutsu_demo.py` | Detection, sign history, and ninjutsu matching | `model/yolox/yolox_nano.onnx` |

### Options shared by all three demos

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `0` | Camera device number |
| `--file` | `None` | Video input; takes precedence over the camera |
| `--width` | `960` | Requested capture width |
| `--height` | `540` | Requested capture height |
| `--fps` | `30` | Target processing FPS, subject to inference speed |
| `--skip_frame` | `0` | Process every N+1 frames |
| `--model` | See table above | ONNX model path |
| `--input_shape` | `416,416` | Inference input shape |
| `--score_th` | `0.7` | Class confidence threshold |
| `--with_p6` | Disabled | Enable P6 in FPN/PAN; flag without a value |

`simple_demo.py` and `Ninjutsu_demo.py` also accept `--nms_th` (default `0.45`)
and `--nms_score_th` (default `0.1`). The embedded-post-processing demo does not expose these options.

### Additional ninjutsu options

| Option | Default | Meaning |
| --- | --- | --- |
| `--sign_interval` | `2.0` | Seconds without a new sign before clearing history |
| `--jutsu_display_time` | `5` | Seconds to display a matched technique |
| `--use_display_score` | `False` | Show detection confidence |
| `--erase_bbox` | `False` | Hide bounding-box overlays |
| `--chattering_check` | `1` | Consecutive detections required to accept a sign |
| `--use_fullscreen` | `False` | Experimental fullscreen display |
| `--use_jutsu_lang_en [True/False]` | Omitted | Deprecated compatibility option; always displays English |

For the existing `--use_display_score`, `--erase_bbox`, and `--use_fullscreen`
options, pass `True` to enable and omit the option to disable. Their original
`type=bool` parsing is retained: passing the string `False` also enables them.

`--use_jutsu_lang_en`, `--use_jutsu_lang_en True`, and `--use_jutsu_lang_en False`
are all accepted and issue an English deprecation warning. None changes the language.

## Configuration and matching

`utils/hand_signs.py` loads the same configuration for all current demos.
Display names are independent of recognition IDs.

### Labels: `setting/labels.csv`

The file has an `id,name` header. IDs are consecutive integers starting at 0;
duplicates, missing IDs, and empty names are rejected.

```csv
id,name
0,None
1,Rat
2,Ox
3,Tiger
```

This example shows only the first rows. The complete mapping is:

| ID | Name | ID | Name |
| --- | --- | --- | --- |
| 0 | None | 8 | Ram |
| 1 | Rat | 9 | Monkey |
| 2 | Ox | 10 | Bird |
| 3 | Tiger | 11 | Dog |
| 4 | Hare | 12 | Boar |
| 5 | Dragon | 13 | Hand Claps |
| 6 | Snake | 14 | Unknown |
| 7 | Horse | 15 | Mizunoe |

Keep these IDs unchanged: a model class maps to `int(class_id) + 1`.
Changing a display name does not change recognition.

### Techniques: `setting/jutsu.csv`

Each row is `style,name,sign_id,...`, with **no header** and no trailing empty cells.
The style may be empty. A name and at least one known sign ID are required.

```csv
Fire Style,Fireball Jutsu,6,3,9,12,7,3
,Clone Jutsu,8,6,3
```

Matching compares the entire history to each sequence in file order; the first
exact match wins. A suffix or incomplete sequence does not count. The file contains
14 rows, including both Fireball sequences and the 44-sign Water Dragon sequence.
Water Dragon is classified as **Water Style**.

### Migrating custom CSV files

1. Back up custom files before replacing them. The old label format had English
   and Japanese names without a header. Assign each existing row its zero-based
   row index as `id`, retain this order, and write `id,name` with English names.
2. Build a lookup from each old Japanese label to that same numeric ID.
3. For each old technique row, keep its English style (old column B) and English
   name (old column D). Convert every nonempty sign from old column E onward using
   the lookup. Remove trailing blank cells and the Japanese style/name columns.
4. Preserve technique row order and alternative sequences. Use `Water Style` for
   Water Dragon. Update custom CSV consumers to the new schema at the same time.

The current loaders require the new schema. Legacy demos continue to use their own
original CSV files under `_legacy/v2/setting`.

## Dataset

The training dataset is private; trained models are public. The original author
cites <span id="cite_ref-3">Article 47-7 of Japan's Copyright Act</span> [3](#cite_note-3)
for the dataset distribution policy. Training images include the author's photos,
anime images, and the public
<span id="cite_ref-4">Naruto hand-sign dataset</span> [4](#cite_note-4).

Backgrounds and clothing can reduce detection accuracy. Reports of false detections,
with their conditions and sample images where possible, help improve training.
The original project requests examples for the twelve animal signs, Mizunoe, and Hand Claps;
submitted examples may be added to the training dataset.

### Hand signs

The model recognizes 14 kinds of hand signs: the twelve animal signs, Mizunoe,
and Hand Claps. `None` and `Unknown` remain reserved entries in the label mapping.

<table>
	<tbody>
		<tr>
			<td width="25%">Rat</td>
			<td width="25%">Ox</td>
			<td width="25%">Tiger</td>
			<td width="25%">Hare</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611897-6d032d00-0a9d-11eb-86c4-de1c50c0d7b6.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611906-6ffe1d80-0a9d-11eb-9054-4e68c42e52ca.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611912-712f4a80-0a9d-11eb-8cb8-fc7097e16f60.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611915-72607780-0a9d-11eb-9995-66524ce4f978.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">Dragon</td>
			<td width="25%">Snake</td>
			<td width="25%">Horse</td>
			<td width="25%">Ram</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611920-7391a480-0a9d-11eb-8e74-db39acf90f83.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611922-742a3b00-0a9d-11eb-8a21-8bdf207db9bb.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611928-755b6800-0a9d-11eb-86c0-67605ffd6e9b.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611930-768c9500-0a9d-11eb-81c6-067b632dc43d.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">Monkey</td>
			<td width="25%">Bird</td>
			<td width="25%">Dog</td>
			<td width="25%">Boar</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611931-77252b80-0a9d-11eb-97d6-e3efc6f1aac3.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611935-77bdc200-0a9d-11eb-95e1-feb8bf7f61de.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611936-78eeef00-0a9d-11eb-90b3-f565e4763c50.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611938-7a201c00-0a9d-11eb-9d5f-1daf2405f20f.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">Mizunoe</td>
			<td width="25%">Hand Claps</td>
			<td width="25%">-</td>
			<td width="25%">-</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611947-7c827600-0a9d-11eb-97ae-9d7eabc58cd5.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611943-7b514900-0a9d-11eb-97be-4fda80d17879.jpg" width="100%"></td>
			<td></td>
			<td></td>
		</tr>
	</tbody>
</table>

### Dataset size

- Total images: 10,026 (including 2,651 anime images)
- Annotated images: 7,098
- Unannotated images: 2,928
- Annotation boxes: 8,941

<img src="https://user-images.githubusercontent.com/37477845/163701957-529d7510-88e4-420f-8099-5dd6f6d9a8cf.png" width="35%"> <img src="https://user-images.githubusercontent.com/37477845/163701962-a4818eba-3b4f-4c92-a7b9-f2331b7f3f23.png" width="50%">

## Repository layout

- `model/yolox`: trained ONNX models and inference wrappers
- `setting`: English label names and ID-based technique sequences
- `utils`: CSV loading, sequence matching, FPS measurement, and text rendering
- `utils/font/KouzanMouhitsu.ttf`: original bundled font, renamed for an English path
- `post_process_gen_tools`: tools for merging post-processing into ONNX
- `_legacy/v2`: unchanged SSD/EfficientDet implementation and its documentation
- `tests`: configuration, matching, rendering, and CLI regression tests

## Tests

With the demo dependencies installed, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Application example

[Sign VADERS — 15th UE4 Petit Contest](https://www.youtube.com/watch?v=K4-E5SseVtI)

## Acknowledgements

The original EfficientDet training used
<span id="cite_ref-5">karaage's Object Detection API tutorial</span> [5](#cite_note-5).
Karaage also introduced Deep Sharingan on
<span id="cite_ref-6">his blog</span> [6](#cite_note-6).
YOLOX training used
<span id="cite_ref-7">YOLOX-Colaboratory-Training-Sample</span> [7](#cite_note-7).
Thank you to these authors for their work.

## References
1. [^](#cite_ref-1)<span id="cite_note-1">Japan: [Copyright Law Article 20 "Right to maintain identity"](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#183)</span>
1. [^](#cite_ref-2)<span id="cite_note-2">[NARUTO](https://www.shonenjump.com/j/rensai/naruto.html)Masashi Kishimoto/Shueisha 1999-2014</span>
1. [^](#cite_ref-3)<span id="cite_note-3">Japan: [Copyright Act Article 47-7 "Transfer of reproductions made due to restrictions on reproduction rights"](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#407)</span>
1. [^](#cite_ref-4)<span id="cite_note-4">Kaggle Public dataset: [naruto-hand-sign-dataset](https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset)</span>
1. [^](#cite_ref-5)<span id="cite_note-5">Karaage-san's blog: [Training an object detector on custom data with the Object Detection API (TensorFlow 2.x)](https://qiita.com/karaage0703/items/8567cc192e151bac3e50)</span>
1. [^](#cite_ref-6)<span id="cite_note-6">Karaage-san's blog: [Experience Naruto with AI: playing with Deep Sharingan](https://karaage.hatenadiary.jp/entry/2020/10/16/073000)</span>
1. [^](#cite_ref-7)<span id="cite_note-7">[Kazuhito00/YOLOX-Colaboratory-Training-Sample](https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample)</span>

## Author

[Kazuhito Takahashi](https://twitter.com/KzhtTkhs)

## License

NARUTO-HandSignDetection is distributed under the [MIT license](LICENSE).

### Font license and attribution

The bundled [KouzanMouhitsu font](https://opentype.jp/kouzanmouhitufont.htm)
retains its original licensing terms. Only its local filename was changed;
its binary contents are unchanged. Consult the font's original page for its terms.
