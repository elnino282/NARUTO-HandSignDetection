"""Read English display names and match language-independent hand-sign IDs."""

import csv
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Jutsu:
    style: str
    name: str
    sign_ids: Tuple[int, ...]

    @property
    def display_name(self):
        return f'{self.style}: {self.name}' if self.style else self.name


def load_labels(path='setting/labels.csv'):
    """Return an ID-to-name mapping; IDs must retain the model's class order."""
    labels = {}
    with open(path, encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'name']:
            raise ValueError(f'{path}: expected CSV header id,name')
        for row in reader:
            try:
                sign_id = int(row['id'])
            except (TypeError, ValueError):
                raise ValueError(f'{path}:{reader.line_num}: invalid sign ID') from None
            if sign_id < 0 or sign_id in labels:
                raise ValueError(
                    f'{path}:{reader.line_num}: duplicate or negative sign ID {sign_id}')
            if None in row or not row['name'] or not row['name'].strip():
                raise ValueError(f'{path}:{reader.line_num}: expected id,name')
            labels[sign_id] = row['name'].strip()
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError(f'{path}: sign IDs must be consecutive, starting at 0')
    return labels


def load_jutsu(labels, path='setting/jutsu.csv'):
    """Read headerless style,name,sign_id,... rows in their matching order."""
    techniques = []
    with open(path, encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        for row in reader:
            if len(row) < 3 or not row[1].strip():
                raise ValueError(f'{path}:{reader.line_num}: expected style,name,sign_id,...')
            try:
                sign_ids = tuple(int(value) for value in row[2:])
            except ValueError:
                raise ValueError(f'{path}:{reader.line_num}: invalid sign ID') from None
            for sign_id in sign_ids:
                if sign_id not in labels:
                    raise ValueError(
                        f'{path}:{reader.line_num}: unknown sign ID {sign_id}')
            techniques.append(Jutsu(row[0].strip(), row[1].strip(), sign_ids))
    if not techniques:
        raise ValueError(f'{path}: no jutsu defined')
    return techniques


def match_jutsu(sign_history, techniques):
    """Return the first exact match, or None; display names never affect matching."""
    history = tuple(sign_history)
    if history:
        for index, technique in enumerate(techniques):
            if history == technique.sign_ids:
                return index
    return None
