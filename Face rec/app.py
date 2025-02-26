from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import sqlite3
import base64
import datetime

app = Flask(__name__)

# Inisialisasi detektor wajah MTCNN
detector = MTCNN()

# Load model FaceNet untuk ekstraksi fitur wajah
embedder = FaceNet()

# Buka kamera laptop
cap = cv2.VideoCapture(0)

DB_PATH = "face_database.db"

def init_db():
    """Membuat tabel database jika belum ada"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image BLOB,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_image_to_db(name, image, embedding):
    """Menyimpan gambar wajah dan embedding ke database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Konversi gambar ke format base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Konversi embedding ke format base64
    embedding_base64 = base64.b64encode(embedding.tobytes()).decode('utf-8')
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO faces (name, timestamp, image, embedding) VALUES (?, ?, ?, ?)", 
                   (name, timestamp, image_base64, embedding_base64))
    
    conn.commit()
    conn.close()

def load_known_faces():
    """Muat wajah yang tersimpan di database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    known_faces = {}
    for row in rows:
        name = row[0]
        embedding = np.frombuffer(base64.b64decode(row[1]), dtype=np.float32)
        known_faces[name] = embedding
    return known_faces

def recognize_face(embedding, known_faces):
    """Cocokkan wajah dengan database"""
    best_match = "Unknown"
    best_distance = 1.0  # Threshold

    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(embedding - known_embedding)
        if distance < best_distance:
            best_match = name
            best_distance = distance

    return best_match

def generate_frames():
    known_faces = load_known_faces()

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face["box"]
            face_crop = frame[y:y+h, x:x+w]

            # Ekstrak embedding wajah
            face_embedding = embedder.embeddings([face_crop])[0]

            # Kenali wajah
            name = recognize_face(face_embedding, known_faces)

            # Gambar kotak dan label nama
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_face', methods=['POST'])
def register_face():
    """Mendaftarkan wajah dengan nama"""
    name = request.form["name"]
    
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Gagal menangkap gambar dari kamera"})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    if not faces:
        return jsonify({"error": "Wajah tidak ditemukan, coba lagi"})

    x, y, w, h = faces[0]["box"]
    face_crop = frame[y:y+h, x:x+w]

    face_embedding = embedder.embeddings([face_crop])[0]

    save_image_to_db(name, face_crop, face_embedding)
    return jsonify({"message": f"Wajah {name} telah disimpan"})

if __name__ == "__main__":
    app.run(debug=True)
