✋ Palmistry Symbol Detection & Horoscope Interpretation

This project detects palm symbols from hand images using a YOLOv8 deep learning model and generates interpretations along with horoscope insights based on zodiac calculations.

The system combines Computer Vision + Palmistry + Zodiac Analysis to produce a descriptive life interpretation.

The application is built using:

Python

YOLOv8 (Ultralytics)

OpenCV

Streamlit

NumPy & Pandas

📂 Project Structure
Palmistry_Project
│
├── final palmistry.zip        # Dataset (images + labels)
│
├── newapp.py                  # Main Streamlit detection application
├── zodiac.py                  # Zodiac calculation & horoscope descriptions
│
└── README.md
🚀 Features

✔ Palm symbol detection using YOLOv8
✔ Automatic image rotation correction
✔ Palm region extraction (background ignored)
✔ Symbol counting and interpretation
✔ Horoscope prediction using zodiac logic
✔ Image upload and webcam capture support
✔ Interactive Streamlit user interface

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/palmistry-project.git
cd palmistry-project
2️⃣ Create Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate
Mac / Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Required Libraries
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install pillow
📁 Dataset Setup

Extract the dataset file:

final palmistry.zip

Dataset must follow YOLO format:
dataset/
│
├── images/
│   ├── train
│   ├── val
│   └── test
│
├── labels/
│   ├── train
│   ├── val
│   └── test
Each image must have a matching label file.

Example:

images/train/palm1.jpg
labels/train/palm1.txt
🧠 Train the Detection Model (Optional)

If you want to train the YOLO model using the dataset:

yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=80 imgsz=640

After training, the model will be saved in:

runs/detect/train/weights/best.pt

Update the model path in newapp.py if necessary.

▶️ Run the Application

Start the Streamlit application:

streamlit run newapp.py

Open the application in your browser:

http://localhost:8501
🖥 How to Use
📤 Upload Palm Image

Open Upload Image tab

Upload a palm image

Click Run Detection

The system will:

Detect palm symbols

Draw bounding boxes

Count detected symbols

Display palm interpretation

Generate horoscope insights

📷 Webcam Capture

Open Webcam tab

Capture palm image

Click Run Webcam Detection

🔮 Horoscope + Palm Interpretation Output

The system first displays Zodiac-based horoscope insights, then palm symbol interpretations.

Example Output
Zodiac Sign: Leo
Life Path Number: 3

Strong career growth and success opportunities.
High discipline and leadership potential.
Strong professional and financial growth phase.
Independent and innovative life journey.
🔄 Project Workflow
User Input (Palm Image / Webcam)
        │
        ▼
Image Preprocessing
 - Rotation correction
 - Image normalization
        │
        ▼
Palm Region Extraction
 - Background removed
        │
        ▼
YOLOv8 Symbol Detection
 - Bounding box detection
        │
        ▼
Symbol Counting
 - Count occurrences
        │
        ▼
Palm Interpretation
 - Meaning mapping
        │
        ▼
Zodiac Calculation
 - Zodiac sign detection
 - Life path number
        │
        ▼
Horoscope Generation
        │
        ▼
Results Display (Streamlit UI)
💡 Notes

Use clear palm images for better detection

Model automatically handles rotated images

Horoscope output is generated using logic in zodiac.py
