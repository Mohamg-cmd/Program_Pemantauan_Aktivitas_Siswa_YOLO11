# Panduan Instalasi Universal

## 1. Persiapan

Ekstrak paket source code, lalu buka terminal pada folder proyek yang memuat `run_pipeline.py` dan `requirements.txt`. Pastikan Python 3.10 atau lebih baru tersedia. Python 3.11 direkomendasikan.

```bash
python --version
python -m venv .venv
```

## 2. Aktivasi environment

- Windows CMD: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Jika PowerShell memblokir skrip aktivasi, gunakan Windows CMD atau ikuti kebijakan execution policy di lingkungan Anda.

## 3. Instal dependency

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Untuk GPU NVIDIA, gunakan PyTorch yang kompatibel dengan driver dan perangkat CUDA. Untuk pemrosesan CPU, ubah `detection.device` menjadi `"cpu"` pada `config.yaml`.

## 4. Siapkan input

- Letakkan checkpoint YOLO11 di `weights/best.pt`, atau sesuaikan `source.model_path`.
- Letakkan video sumber di `data/video_part001.mp4`, atau sesuaikan `source.video_path`.
- Periksa `data/student_seats.csv`; koordinat kursi harus sesuai dengan perspektif dan resolusi video. Untuk ruang lain, gunakan `python tools/calibrate_seats.py --help` untuk melihat opsi kalibrasi.
- Dataset gambar/label tidak disertakan. Dataset hanya diperlukan jika evaluasi detector diaktifkan melalui konfigurasi.

## 5. Jalankan program

Windows: jalankan `RUN_PIPELINE.bat`. Linux/macOS: jalankan `./run_pipeline.sh`. Alternatif dari terminal pada folder proyek:

```bash
python run_pipeline.py --config config.yaml
```

Untuk menggunakan deteksi yang telah disimpan sebelumnya, pastikan `output/detections.csv` tersedia lalu gunakan `--skip-detect`. Untuk audit identitas saja, tambahkan `--identity-audit-only`.

## 6. Verifikasi instalasi

```bash
python -m compileall -q run_pipeline.py run_experiments.py src tests tools
python -m pytest -q tests
```

Pengujian otomatis memeriksa fungsi yang tercakup dalam test suite, bukan memastikan performa detector pada setiap video baru.
