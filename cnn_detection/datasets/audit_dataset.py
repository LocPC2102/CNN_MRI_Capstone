"""Audit AISSLab Axial detection images and optional Pascal-VOC-style XML files.

Run from project root:
    python src/cnn_detector_axial/datasets/audit_dataset.py

The dataset may have five axial images per patient (L1-L2 through L5-S1),
but not every image necessarily has an XML. Missing XML is recorded as
'unannotated' and is NEVER assumed to mean a negative/normal image.
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = [
    "patient_id", "image_path", "xml_path", "annotation_status",
    "image_width", "image_height", "object_count", "classes", "levels",
    "boxes_json", "objects_json", "error",
]


def _text(node, path: str, default: str = "") -> str:
    found = node.find(path)
    return (found.text or "").strip() if found is not None and found.text else default


def parse_xml(xml_path: Path, image_width: int, image_height: int):
    """Return object records and validation errors from a VOC-like XML."""
    root = ET.parse(xml_path).getroot()
    objects, errors = [], []

    for idx, obj in enumerate(root.findall("object")):
        name = _text(obj, "name")
        box = obj.find("bndbox")
        if not name:
            errors.append(f"object[{idx}] missing class name")
            continue
        if box is None:
            errors.append(f"object[{idx}] missing bndbox")
            continue

        try:
            # VOC-style coordinates are represented as xyxy for torchvision.
            xmin = float(_text(box, "xmin"))
            ymin = float(_text(box, "ymin"))
            xmax = float(_text(box, "xmax"))
            ymax = float(_text(box, "ymax"))
        except (TypeError, ValueError):
            errors.append(f"object[{idx}] non-numeric bbox")
            continue

        if not (xmin < xmax and ymin < ymax):
            errors.append(f"object[{idx}] invalid bbox order: {(xmin, ymin, xmax, ymax)}")
            continue
        if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
            errors.append(
                f"object[{idx}] bbox outside image bounds: "
                f"{(xmin, ymin, xmax, ymax)} for {image_width}x{image_height}"
            )
            continue

        level = _text(obj, "level")
        ap_diameter = _text(obj, "ap_diameter")
        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "level": level,
            "ap_diameter": ap_diameter,
        })

    return objects, errors


def audit(data_dir: Path, out_csv: Path, class_map_path: Path, report_path: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    rows = []
    class_counts, level_counts, patient_image_counts = Counter(), Counter(), Counter()
    status_counts = Counter()
    image_count = xml_count = 0

    patient_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    # Expected layout: data_dir/<patient_id>/*.png and *.xml.
    # If images are directly under data_dir, use that directory as one group.
    if not patient_dirs:
        patient_dirs = [data_dir]

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        image_files = sorted(
            p for p in patient_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        xml_by_stem = {
            p.stem: p for p in patient_dir.rglob("*.xml") if p.is_file()
        }
        image_stems = {p.stem for p in image_files}
        image_count += len(image_files)
        xml_count += len(xml_by_stem)

        # Record images, including images that do not have XML.
        for image_path in image_files:
            xml_path = xml_by_stem.get(image_path.stem)
            row = {
                "patient_id": patient_id,
                "image_path": image_path.as_posix(),
                "xml_path": xml_path.as_posix() if xml_path else "",
                "annotation_status": "",
                "image_width": "",
                "image_height": "",
                "object_count": 0,
                "classes": "",
                "levels": "",
                "boxes_json": "[]",
                "objects_json": "[]",
                "error": "",
            }

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                row["image_width"], row["image_height"] = width, height
                patient_image_counts[patient_id] += 1

                if xml_path is None:
                    row["annotation_status"] = "unannotated"
                    row["error"] = "XML missing; annotation status unknown, not a negative sample"
                else:
                    try:
                        objects, errors = parse_xml(xml_path, width, height)
                        row["object_count"] = len(objects)
                        row["classes"] = "|".join(sorted({o["name"] for o in objects}))
                        row["levels"] = "|".join(sorted({o["level"] for o in objects if o["level"]}))
                        row["boxes_json"] = json.dumps([o["bbox"] for o in objects], ensure_ascii=False)
                        row["objects_json"] = json.dumps(objects, ensure_ascii=False)
                        if errors:
                            row["annotation_status"] = "invalid"
                            row["error"] = " ; ".join(errors)
                        elif not objects:
                            row["annotation_status"] = "empty_xml"
                            row["error"] = "XML has no valid object; confirm whether this means a true negative"
                        else:
                            row["annotation_status"] = "annotated"
                        for obj in objects:
                            class_counts[obj["name"]] += 1
                            if obj["level"]:
                                level_counts[obj["level"]] += 1
                    except (ET.ParseError, OSError) as exc:
                        row["annotation_status"] = "invalid"
                        row["error"] = f"XML parse/read error: {exc}"
            except Exception as exc:
                row["annotation_status"] = "invalid"
                row["error"] = f"Image read error: {exc}"

            status_counts[row["annotation_status"]] += 1
            rows.append(row)

        # XML without a matching image is also an audit issue.
        for stem, xml_path in xml_by_stem.items():
            if stem not in image_stems:
                rows.append({
                    "patient_id": patient_id, "image_path": "", "xml_path": xml_path.as_posix(),
                    "annotation_status": "invalid", "image_width": "", "image_height": "",
                    "object_count": 0, "classes": "", "levels": "", "boxes_json": "[]",
                    "objects_json": "[]", "error": "XML has no image with matching filename stem",
                })
                status_counts["invalid"] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    class_names = sorted(class_counts)
    class_map = {name: i + 1 for i, name in enumerate(class_names)}  # 0 reserved for background
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    class_map_path.write_text(json.dumps(class_map, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# AISSLab Axial Detection Dataset Audit", "",
        f"- Data directory: `{data_dir}`",
        f"- Patient directories scanned: {len(patient_dirs)}",
        f"- Images found: {image_count}",
        f"- XML files found: {xml_count}",
        f"- Manifest rows (includes unmatched XML rows): {len(rows)}", "",
        "## Image annotation status",
    ]
    report_lines += [f"- {key}: {value}" for key, value in sorted(status_counts.items())]
    report_lines += ["", "## Object counts by class"]
    report_lines += [f"- {key}: {value}" for key, value in sorted(class_counts.items())]
    report_lines += ["", "## Object counts by level"]
    report_lines += [f"- {key}: {value}" for key, value in sorted(level_counts.items())]
    report_lines += [
        "", "## Important interpretation notes",
        "- `unannotated` means XML is absent; it does not mean normal/negative.",
        "- `empty_xml` means an XML exists but has no objects; verify dataset annotation policy before training it as a negative.",
        "- `invalid` rows must be reviewed before training.",
        "- The generated class IDs start at 1; ID 0 is reserved for detector background.",
        "- Patient-level splitting should be used to prevent data leakage.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Patients scanned : {len(patient_dirs)}")
    print(f"Images found     : {image_count}")
    print(f"XML files found  : {xml_count}")
    print(f"Status counts    : {dict(status_counts)}")
    print(f"Classes          : {dict(class_counts)}")
    print(f"Levels           : {dict(level_counts)}")
    print(f"Manifest         : {out_csv}")
    print(f"Class map        : {class_map_path}")
    print(f"Report           : {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="DATA/raw/AISSLab_Axial/Localization_and_Severity_Classification",
    )
    parser.add_argument("--out", default="DATA/metadata/detection_manifest.csv")
    parser.add_argument("--class-map", default="DATA/metadata/detection_class_map.json")
    parser.add_argument("--report", default="reports/detection_audit_report.md")
    args = parser.parse_args()
    audit(Path(args.data_dir), Path(args.out), Path(args.class_map), Path(args.report))
