import cv2
import numpy as np
from pathlib import Path
from typing import Dict
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import datetime

# ===================== CONFIG =====================

DEFAULT_MODEL_PATH = "runs/detect/palm_symbols_run2/weights/best.pt"

PALM_MARGIN_TOP = 0.08
PALM_MARGIN_BOTTOM = 0.08
PALM_MARGIN_LEFT = 0.10
PALM_MARGIN_RIGHT = 0.10

PRESETS = {
    "Soft (more symbols)": dict(conf=0.05, min_area=2000, nms_iou=0.50),
    "Medium (balanced)": dict(conf=0.07, min_area=4000, nms_iou=0.45),
    "Strong (cleaner)": dict(conf=0.10, min_area=6000, nms_iou=0.40),
    "Very strong (few but precise)": dict(conf=0.15, min_area=8000, nms_iou=0.35),
}

# ================= SYMBOL MEANINGS =================

SYMBOL_MEANINGS_BY_COUNT = {
    "Downward lines": {
        "low": "Normal life challenges.",
        "medium": "Persistent effort and resilience.",
        "high": "Strong endurance and long-term stability."
    },
    "Upward lines": {
        "low": "Basic growth tendencies.",
        "medium": "Steady career development.",
        "high": "Strong ambition and rapid success growth."
    },
    "Vertical lines": {
        "low": "Basic discipline.",
        "medium": "Consistent hard work.",
        "high": "High leadership and authority potential."
    },
    "Triangle": {
        "low": "Basic analytical ability.",
        "medium": "Structured thinking.",
        "high": "High intelligence and planning ability."
    },
    "Crosses": {
        "low": "Minor life events.",
        "medium": "Important turning points.",
        "high": "Major life transformations."
    },
    "Tridents": {
        "low": "Hidden talents.",
        "medium": "Balanced skills.",
        "high": "Success through multiple talents."
    }
}

# ================= HELPER FUNCTIONS =================

def get_count_level(count):
    if count >= 4:
        return "high"
    elif count >= 2:
        return "medium"
    return "low"

def build_interpretation_table(counts: Dict[str, int]):
    rows = []
    for symbol, count in counts.items():
        level = get_count_level(count)
        meaning = SYMBOL_MEANINGS_BY_COUNT.get(symbol, {}).get(level, "General palm indication.")
        rows.append({
            "Symbol": symbol,
            "Count": count,
            "Level": level.capitalize(),
            "Meaning": meaning
        })
    return pd.DataFrame(rows)

# ================= HOROSCOPE ENGINE =================

def calculate_life_path_number(dob):
    digits = [int(d) for d in dob.strftime("%d%m%Y")]
    total = sum(digits)
    while total > 9:
        total = sum(int(d) for d in str(total))
    return total

def get_zodiac_sign(day, month):
    zodiac_ranges = {
        "Aries": ((3, 21), (4, 19)),
        "Taurus": ((4, 20), (5, 20)),
        "Gemini": ((5, 21), (6, 20)),
        "Cancer": ((6, 21), (7, 22)),
        "Leo": ((7, 23), (8, 22)),
        "Virgo": ((8, 23), (9, 22)),
        "Libra": ((9, 23), (10, 22)),
        "Scorpio": ((10, 23), (11, 21)),
        "Sagittarius": ((11, 22), (12, 21)),
        "Capricorn": ((12, 22), (1, 19)),
        "Aquarius": ((1, 20), (2, 18)),
        "Pisces": ((2, 19), (3, 20)),
    }

    for sign, ((sm, sd), (em, ed)) in zodiac_ranges.items():
        if (month == sm and day >= sd) or (month == em and day <= ed):
            return sign
    return "Capricorn"

def generate_full_horoscope(counts, dob):

    zodiac = get_zodiac_sign(dob.day, dob.month)
    life_path = calculate_life_path_number(dob)

    output = []
    output.append(f"Zodiac Sign: {zodiac}")
    output.append(f"Life Path Number: {life_path}")

    # Career growth
    if counts.get("Upward lines", 0) >= 2:
        output.append("Strong career growth and success opportunities.")

    # Leadership
    if counts.get("Vertical lines", 0) >= 2:
        output.append("High discipline and leadership potential.")

    # Financial / stability
    if counts.get("Downward lines", 0) >= 3:
        output.append("Strong professional and financial growth phase.")

    # Life path personality
    if life_path in [1,3,5]:
        output.append("Independent and innovative life journey.")
    elif life_path in [2,6]:
        output.append("Relationship-centered and emotionally driven life.")
    else:
        output.append("Spiritual growth and transformation life path.")

    return "\n".join(output)

# ================= NMS =================

def nms_numpy(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
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

    return keep

# ================= STREAMLIT UI =================

st.set_page_config(page_title="AI Palm + Horoscope System", layout="wide")

st.title("🔮 AI Palm Symbol Detection + Advanced Horoscope")

st.markdown("""
### Now your AI system will:
1. Detect palm symbols using YOLO  
2. Analyze symbol frequency  
3. Take personal details  
4. Calculate zodiac sign, life path number, and palm-based personality  
5. Generate full horoscope prediction  
""")

# Inputs
name = st.text_input("Full Name")
dob = st.date_input("Date of Birth", min_value=datetime.date(1900,1,1), max_value=datetime.date.today())
gender = st.selectbox("Gender", ["Male","Female","Other"])

preset_name = st.selectbox("Detection Preset", PRESETS.keys())
preset = PRESETS[preset_name]

uploaded = st.file_uploader("Upload Palm Image", type=["jpg","png"])

@st.cache_resource
def load_model(path):
    return YOLO(str(Path(path)))

model = load_model(DEFAULT_MODEL_PATH)
class_names = model.names

if uploaded and st.button("Run Full AI Prediction"):

    img_bgr = cv2.cvtColor(np.array(Image.open(uploaded)), cv2.COLOR_RGB2BGR)
    h,w = img_bgr.shape[:2]

    x1 = int(w * PALM_MARGIN_LEFT)
    x2 = int(w * (1 - PALM_MARGIN_RIGHT))
    y1 = int(h * PALM_MARGIN_TOP)
    y2 = int(h * (1 - PALM_MARGIN_BOTTOM))

    roi = img_bgr[y1:y2, x1:x2]

    results = model.predict(roi, imgsz=640, conf=preset["conf"], verbose=False)[0]

    annotated = img_bgr.copy()

    # Yellow boundary
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 3)

    counts = {}

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        boxes[:,[0,2]] += x1
        boxes[:,[1,3]] += y1

        keep = nms_numpy(boxes, scores, preset["nms_iou"])

        for i in keep:
            box = boxes[i]
            cls_id = cls_ids[i]
            symbol = class_names[cls_id]
            counts[symbol] = counts.get(symbol,0)+1

            x1b,y1b,x2b,y2b = box.astype(int)
            cv2.rectangle(annotated,(x1b,y1b),(x2b,y2b),(0,255,0),2)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

    if counts:
        st.subheader("✋ Palm Interpretation")
        st.dataframe(build_interpretation_table(counts), width="stretch")

        st.subheader("🔥 Example Output")
        st.code(generate_full_horoscope(counts, dob))
    else:
        st.info("No palm symbols detected.")