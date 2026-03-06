import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# ===================== CONFIG =====================

DEFAULT_MODEL_PATH = "runs/detect/palm_symbols_run2/weights/best.pt"

PALM_MARGIN_TOP = 0.12
PALM_MARGIN_BOTTOM = 0.08
PALM_MARGIN_LEFT = 0.18
PALM_MARGIN_RIGHT = 0.18

PRESETS = {
    "Soft (more symbols)": dict(post_conf=0.0015, min_area=5000, nms_iou=0.45),
    "Medium (balanced)": dict(post_conf=0.0015, min_area=8000, nms_iou=0.45),
    "Strong (cleaner)": dict(post_conf=0.0020, min_area=10000, nms_iou=0.45),
    "Very strong (few but precise)": dict(post_conf=0.0030, min_area=12000, nms_iou=0.45),
}

# ================= SYMBOL MEANINGS (COUNT-BASED) =================

SYMBOL_MEANINGS_BY_COUNT = {

    "Downward lines": {
        "low": "Few downward lines indicate normal palm structure.",
        "medium": "Multiple downward lines show sustained effort over time.",
        "high": "Dominant downward lines indicate strong persistence and long-term endurance."
    },

    "Upward lines": {
        "low": "Occasional upward lines suggest mild growth tendencies.",
        "medium": "Several upward lines indicate steady improvement.",
        "high": "Strong upward patterns show high ambition and positive growth."
    },

    "Vertical lines": {
        "low": "Few vertical lines indicate basic discipline and routine.",
        "medium": "Multiple vertical lines suggest consistent effort and focus.",
        "high": "Strong vertical dominance indicates high self-control and determination."
    },

    "Transverse lines": {
        "low": "Few transverse lines indicate minor disturbances.",
        "medium": "Repeated transverse lines suggest periodic obstacles.",
        "high": "Dense transverse lines indicate frequent interruptions or challenges."
    },

    "Crosses": {
        "low": "Few crosses indicate minor life events.",
        "medium": "Multiple crosses suggest notable turning points.",
        "high": "Strong cross dominance shows impactful experiences."
    },

    "Triangle": {
        "low": "Occasional triangle indicates basic analytical ability.",
        "medium": "Repeated triangles suggest structured thinking.",
        "high": "Strong triangular patterns indicate high intelligence and planning skills."
    },

    "Chains": {
        "low": "Few chain patterns show stable emotional state.",
        "medium": "Chains indicate emotional sensitivity.",
        "high": "Dense chains suggest frequent emotional fluctuations."
    },

    "Grilles": {
        "low": "Few grille patterns indicate mild mental strain.",
        "medium": "Multiple grilles suggest overthinking or stress.",
        "high": "Dense grilles indicate high mental pressure or complexity."
    },

    "Tassels": {
        "low": "Simple line endings with minimal complexity.",
        "medium": "Overlapping tassels indicate mixed influences.",
        "high": "Dense tassels show high structural complexity."
    },

    "Breaks": {
        "low": "Minor interruptions in palm lines.",
        "medium": "Noticeable changes handled effectively.",
        "high": "Frequent breaks indicate repeated transitions."
    },

    "Tridents": {
        "low": "Isolated trident suggests multi-skill ability.",
        "medium": "Repeated tridents indicate balanced talents.",
        "high": "Strong trident presence shows success through multiple paths."
    }
}

# ==================================================


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(str(Path(model_path)))


def compute_palm_roi(width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(width * PALM_MARGIN_LEFT)
    x2 = int(width * (1.0 - PALM_MARGIN_RIGHT))
    y1 = int(height * PALM_MARGIN_TOP)
    y2 = int(height * (1.0 - PALM_MARGIN_BOTTOM))
    return x1, y1, x2, y2


def nms_numpy(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(xyxy) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = xyxy.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return np.array(keep)


def get_count_level(count: int) -> str:
    if count >= 4:
        return "high"
    elif count >= 2:
        return "medium"
    return "low"


def build_interpretation_table(counts: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for symbol, count in counts.items():
        level = get_count_level(count)
        meaning = SYMBOL_MEANINGS_BY_COUNT.get(symbol, {}).get(
            level, "General palm pattern indication."
        )
        rows.append({
            "Symbol": symbol,
            "Count": count,
            "Meaning": meaning
        })
    return pd.DataFrame(rows)


def run_yolo_on_palm_roi(
    image_bgr: np.ndarray,
    model,
    class_names: Dict[int, str],
    selected_symbols: List[str],
    post_conf: float,
    min_area: float,
    nms_iou: float,
    imgsz: int,
):
    h, w = image_bgr.shape[:2]
    px1, py1, px2, py2 = compute_palm_roi(w, h)

    roi = image_bgr[py1:py2, px1:px2]
    results = model.predict(roi, imgsz=imgsz, conf=0.0001, verbose=False)[0]

    annotated = image_bgr.copy()
    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)

    if results.boxes is None:
        return annotated, {}

    xyxy = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    xyxy[:, [0, 2]] += px1
    xyxy[:, [1, 3]] += py1

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    mask = (scores >= post_conf) & (areas >= min_area)

    xyxy, scores, cls_ids = xyxy[mask], scores[mask], cls_ids[mask]

    allowed_ids = [i for i, n in class_names.items() if n in selected_symbols]
    mask = np.isin(cls_ids, allowed_ids)
    xyxy, scores, cls_ids = xyxy[mask], scores[mask], cls_ids[mask]

    keep = nms_numpy(xyxy, scores, nms_iou)
    xyxy, scores, cls_ids = xyxy[keep], scores[keep], cls_ids[keep]

    counts = {}
    for box, score, cid in zip(xyxy, scores, cls_ids):
        name = class_names[cid]
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{name} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated, counts


# ===================== STREAMLIT UI =====================

st.set_page_config(page_title="Palm Symbol Detection", layout="wide", page_icon="✋")
st.sidebar.title("Model & Options")

model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
imgsz = st.sidebar.selectbox("Inference size", [0, 512, 640], index=2)

preset_name = st.sidebar.selectbox("Preset", PRESETS.keys(), index=2)
preset_cfg = PRESETS[preset_name]

model = load_model(model_path) if Path(model_path).exists() else None
class_names = model.names if model else {}

selected_symbols = st.sidebar.multiselect(
    "Select symbols",
    options=list(class_names.values()),
    default=list(class_names.values())
)

st.title("Palm Symbol Detection – Palm Region Only")

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Webcam"])


def display_results(img_bgr, counts):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width="stretch")
    if counts:
        df = build_interpretation_table(counts)
        st.subheader("Palm Symbol Interpretation")
        st.dataframe(df, width="stretch")
    else:
        st.info("No symbols detected inside palm region.")


with tab1:
    img = st.file_uploader("Upload palm image", type=["jpg", "png"])
    if img and st.button("Run Detection"):
        img_bgr = cv2.cvtColor(np.array(Image.open(img)), cv2.COLOR_RGB2BGR)
        out, counts = run_yolo_on_palm_roi(
            img_bgr,
            model,
            class_names,
            selected_symbols,
            preset_cfg["post_conf"],
            preset_cfg["min_area"],
            preset_cfg["nms_iou"],
            imgsz,
        )
        display_results(out, counts)

with tab2:
    cam = st.camera_input("Capture palm")
    if cam and st.button("Run Webcam Detection"):
        cam_bgr = cv2.cvtColor(np.array(Image.open(cam)), cv2.COLOR_RGB2BGR)
        out, counts = run_yolo_on_palm_roi(
            cam_bgr,
            model,
            class_names,
            selected_symbols,
            preset_cfg["post_conf"],
            preset_cfg["min_area"],
            preset_cfg["nms_iou"],
            imgsz,
        )
        display_results(out, counts)

st.caption(
    "Detection is limited to the palm region. Meanings are descriptive summaries based on symbol frequency."
)