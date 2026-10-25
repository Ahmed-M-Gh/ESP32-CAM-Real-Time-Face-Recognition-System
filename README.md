# ESP32-CAM Real-Time Face Recognition System 👁️‍🗨️🤖

A low-latency, high-performance facial recognition system utilizing an ESP32-CAM for image capturing, a FastAPI backend server in Python, and InsightFace for precise state-of-the-art face detection and embedding comparison via WebSockets.

---

## 🏗️ System Architecture

1. The Edge (ESP32-CAM): Captures images from the environment and streams them as binary data over a persistent WebSocket connection. It also receives the recognized name to display it on an LCD/OLED screen.

2. The Server (Python / FastAPI): An async backend that instantly receives image buffers, decodes them using OpenCV, and extracts facial embeddings via InsightFace (buffalo_l).

3. Recognition Engine: Compares the live face embedding against a local dataset folder using Cosine Similarity.

---

## 📁 Repository Structure
```text
📂 esp32-cam-face-recognition/
│
├── 📂 dataset/              # Put your reference face images here (e.g., Ahmed.jpg)
├── server.py                # FastAPI WebSocket server & InsightFace pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

1. Prerequisites:

Make sure you have Python 3.8+ installed on your laptop/PC.

2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/esp32-cam-face-recognition.git
cd esp32-cam-face-recognition
```

3. Install Dependencies

Install the required libraries using pip:
```bash
pip install -r requirements.txt
```


4. Setup Your Dataset

Create a folder named dataset in the root directory (if it doesn't auto-create).

Place clear, front-facing reference images of the people you want to recognize inside the folder.

Example: dataset/Ahmed.jpg, dataset/Sara.png

Note: The filename (without extension) will be used as the recognized person's name.

5. Run the Server

Start the FastAPI server:
```bash
python server.py
```
> The server will run on 0.0.0.0:8000 and automatically encode all faces inside the dataset folder upon startup.

---

## 🔌 Next Steps

- Flash the ESP32-CAM with the Arduino code to connect to your local Wi-Fi and stream frames directly to ws://<YOUR_LAPTOP_IP>:8000/ws/face-recognition.

---

## 🛠️ Tech Stack

- Edge Device: ESP32-CAM (OV2640 Camera module)
- Backend Framework: FastAPI (Python) & WebSockets
- AI / Face Recognition: InsightFace (buffalo_l model) & OpenCV
- Math & Matching: NumPy (Cosine Similarity)