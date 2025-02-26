import cv2

cap = cv2.VideoCapture(0)  # Pastikan menggunakan kamera yang benar

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break

    cv2.imshow("Test Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
