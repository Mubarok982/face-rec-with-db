const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture");
const resultImage = document.getElementById("result");

// Aktifkan kamera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Gagal mengakses kamera:", err));

// Tangkap gambar dari kamera
captureButton.addEventListener("click", async () => {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    // Kirim gambar ke backend untuk deteksi wajah
    const response = await fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    });

    const data = await response.json();
    if (data.image) {
        resultImage.src = "data:image/jpeg;base64," + data.image;
    }
});
