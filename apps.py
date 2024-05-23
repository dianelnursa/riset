# Mengimpor pustaka yang diperlukan dari Flask untuk membuat aplikasi web
from flask import Flask, render_template, request, jsonify, flash
# Mengimpor fungsi load_model dari Keras untuk memuat model yang sudah dilatih
from keras.models import load_model

# Mengimpor utilitas dari Keras untuk memuat dan mengubah gambar menjadi array
from keras.utils import load_img, img_to_array
# Mengimpor modul Xception dari Keras untuk pemrosesan gambar
from keras_applications.xception import Xception
from keras_applications.xception import preprocess_input
from keras_applications.xception import decode_predictions

# Mengimpor TensorFlow sebagai backend Keras
import tensorflow as tf
from tensorflow import keras
# Mengimpor transformasi gambar dan input/output dari skimage
from skimage import transform, io
# Mengimpor NumPy untuk manipulasi array
import numpy as np
# Mengimpor modul os untuk operasi sistem
import os
# Mengimpor modul Image dari PIL untuk manipulasi gambar
from PIL import Image
# Mengimpor modul datetime untuk menangani tanggal dan waktu
from datetime import datetime
# Mengimpor utilitas image dari Keras untuk pemrosesan gambar
from keras.preprocessing import image
# Mengimpor modul CORS dari Flask untuk mengizinkan permintaan lintas sumber daya
from flask_cors import CORS

# Membuat instance dari aplikasi Flask
app = Flask(__name__)
# Mengatur secret key untuk keamanan sesi aplikasi
app.secret_key="qwerty0987765421"

# Memuat model pra-terlatih Xception dari file .h5
modelxception = load_model("Xception-fructus-99.23.h5")

# Mengatur folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
# Menambahkan konfigurasi folder unggahan ke aplikasi Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Menentukan ekstensi file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

# Fungsi untuk memeriksa apakah file yang diunggah memiliki ekstensi yang diizinkan
def allowed_file(filename):
    # Memeriksa apakah ada titik dalam nama file dan ekstensi file ada dalam daftar yang diizinkan
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rute untuk halaman utama
@app.route("/", methods=['GET', 'POST'])
def main():
    # Mengembalikan template HTML untuk halaman utama
    return render_template("beranda.html")

# Rute untuk halaman 'belajar'
@app.route("/belajar", methods=['GET', 'POST'])
def belajar():
    # Mengembalikan template HTML untuk halaman belajar
    return render_template("belajar.html")

# Rute untuk halaman 'classification'
@app.route("/classification", methods=['GET', 'POST'])
def classification():
    # Mengembalikan template HTML untuk halaman klasifikasi
    return render_template("classifications.html")

# Rute untuk halaman 'tentang'
@app.route("/tentang", methods=['GET', 'POST'])
def tentang():
    # Mengembalikan template HTML untuk halaman tentang developer
    return render_template("developer.html")

# Rute untuk menangani pengunggahan gambar dan prediksi
@app.route('/submit', methods=['POST'])
def predict():
    # Mengambil daftar file yang diunggah dari permintaan
    files = request.files.getlist('file')
    # Menentukan nama file sementara untuk menyimpan gambar yang diunggah
    filename = "temp_image.png"
    # Menandai apakah unggahan berhasil atau tidak
    success = False
    # Memproses setiap file dalam daftar file yang diunggah
    for file in files:
        # Memeriksa apakah file valid dan memiliki ekstensi yang diizinkan
        if file and allowed_file(file.filename):
            # Menyimpan file yang diunggah ke folder unggahan
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Menandai bahwa unggahan berhasil
            success = True
        else:
            # Menampilkan pesan kesalahan jika file tidak valid
            flash("Anda belum Mengunggah File atau Ekstensi File Salah, Silahkan ulangi unggah file dan pastikan ekstensi file sudah sesuai panduan di atas!")
            # Mengembalikan template HTML untuk halaman klasifikasi dengan pesan kesalahan
            return render_template("classifications.html")
    
    # Menentukan path untuk gambar yang diunggah
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Mengonversi gambar ke format RGB menggunakan PIL
    img = Image.open(img_url).convert('RGB')
    # Mendapatkan waktu saat ini untuk membuat nama file yang unik
    now = datetime.now()
    # Menentukan path untuk menyimpan gambar yang akan diprediksi
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    # Menyimpan gambar yang telah diubah ke RGB
    image_predict = predict_image_path
    img.convert('RGB').save(image_predict, format="png")
    # Menutup file gambar setelah selesai
    img.close()

    # Menyiapkan gambar untuk prediksi dengan mengubah ukuran dan mengubahnya menjadi array
    img = load_img(predict_image_path, target_size=(128, 128, 3))
    x = img_to_array(img) / 255.0  # Normalisasi nilai pixel
    x = x.reshape(1, 128, 128, 3)
    images = np.array(x)

    # Melakukan prediksi menggunakan model yang sudah dilatih
    prediction_array_xception = modelxception.predict(images)

    # Menyiapkan respon API dengan daftar nama kelas
    class_names = [
        'Amomi Fructus (Kapulaga)', 'Capsici Fructescentis Fructus (Cabai Rawit)', 
        'Cumini Fructus (Jinten Putih)', 'Piper Retrofractum Fructus (Cabai Jawa)', 
        'Piperis Albi Fructus (Lada Putih)', 'Piperis Nigri Fructus (Lada Hitam)', 
        'Tamarindus Indicia Fructus (Asam Jawa)', 
        'Tidak ada fructus yang terdeteksi, gambar tersebut bukan fructus'
    ]
    
    # Mengembalikan hasil prediksi ke halaman klasifikasi
    return render_template("classifications.html", img_path=predict_image_path, 
                           predictionxception=class_names[np.argmax(prediction_array_xception)],
                           confidenceexception='{:2.0f}%'.format(100 * np.max(prediction_array_xception)))

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
