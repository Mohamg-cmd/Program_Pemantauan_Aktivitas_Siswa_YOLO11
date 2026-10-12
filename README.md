# Program Pemantauan Aktivitas Siswa Berbasis YOLO11

Paket source code untuk mendeteksi dan melacak aktivitas siswa di ruang kelas, mengasosiasikan fragmen lintasan ke identitas siswa berdasarkan posisi kursi tetap, mengagregasi aktivitas secara temporal, dan menghasilkan analisis keputusan berbasis AHP–SAW.

## Alur utama

1. **Deteksi dan pelacakan**: YOLO11 mendeteksi empat kelas aktivitas dan BoT-SORT menghasilkan `track_id` sebagai identitas lintasan.
2. **Asosiasi identitas**: `src/identity.py` menggunakan peta kursi tetap untuk mengaitkan track/fragmen ke `student_id`. `track_id` tidak boleh dianggap sebagai identitas siswa.
3. **Agregasi temporal**: aktivitas diukur dalam representasi durasi dan/atau frekuensi berdasarkan FPS video.
4. **Pengambilan keputusan**: bobot kriteria dihitung dengan AHP; ranking utama dihitung dengan SAW. Analisis sensitivitas dan Monte Carlo tersedia sebagai pemeriksaan kestabilan.
5. **Pelaporan**: CSV audit dan workbook Excel dibuat di folder `output/`.

## Persyaratan

- Python 3.10 atau lebih baru (Python 3.11 direkomendasikan).
- Dependencies pada `requirements.txt`.
- Checkpoint YOLO11 `best.pt`, video sumber, dan `data/student_seats.csv` yang sesuai. Checkpoint, video sumber, dan dataset penuh tidak disertakan dalam paket ini.
- GPU NVIDIA direkomendasikan untuk inferensi cepat. Penggunaan CPU didukung melalui `detection.device: "cpu"`, tetapi dapat lebih lambat.

## Instalasi universal

Ekstrak ZIP, buka terminal di folder proyek (folder yang berisi `run_pipeline.py` dan `requirements.txt`), lalu jalankan:

```bash
python --version
python -m venv .venv
```

Aktifkan environment:

- Windows CMD: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Pasang dependency:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Untuk CUDA/NVIDIA, pastikan instalasi PyTorch sesuai driver dan GPU yang digunakan.

## Menyiapkan input

- Letakkan checkpoint hasil pelatihan di `weights/best.pt`, atau ubah `source.model_path` pada `config.yaml`.
- Letakkan video sumber di `data/video_part001.mp4`, atau ubah `source.video_path`.
- Periksa `data/student_seats.csv`. Peta ini spesifik pada geometri kamera/ruangan; gunakan `python tools/calibrate_seats.py --help` untuk kalibrasi ulang.
- Evaluasi detector menggunakan `source.data_yaml` dan dataset terpisah. Evaluasi tersebut nonaktif secara default.

## Menjalankan program

Windows: jalankan `RUN_PIPELINE.bat`. Linux/macOS: jalankan `./run_pipeline.sh`. Atau jalankan langsung dari folder proyek:

```bash
python run_pipeline.py --config config.yaml
```

Opsi umum:

```bash
python run_pipeline.py --config config.yaml --identity-audit-only
python run_pipeline.py --config config.yaml --temporal-both
python run_pipeline.py --config config.yaml --monte-carlo
python run_pipeline.py --config config.yaml --skip-detect
python run_pipeline.py --config config.yaml --track-level
```

`--skip-detect` memerlukan `output/detections.csv` yang sudah ada. `--track-level` hanya untuk debugging dan tidak boleh dianggap sebagai ranking siswa.

## Output umum

- `output/detections.csv`: deteksi per frame beserta track, kelas, confidence, dan bounding box.
- `output/track_summary.csv`: ringkasan lintasan. Jumlah track bukan jumlah siswa.
- `output/identity_assignments.csv`: hasil asosiasi track ke kursi/student.
- `output/track_to_student_summary.csv`: ringkasan fragmentasi track → student.
- `output/student_identity_audit.csv`: coverage dan audit identitas per siswa.
- `output/long_gap_audit.csv`: audit gap panjang antar fragmen.
- `output/student_activity_matrix*.csv`: matriks aktivitas tingkat siswa.
- `output/sensitivity_oat.csv` dan `output/sensitivity_montecarlo.csv`: keluaran analisis kestabilan jika aktif.
- `output/yolo11_classroom_research_report.xlsx`: workbook rekap.

## Catatan pengoperasian

- Default FPS pada konfigurasi adalah 25 untuk rekaman yang digunakan dalam implementasi ini. Ubah `temporal.fps` hanya berdasarkan metadata video yang sebenarnya; `auto` juga tersedia bila metadata dapat dibaca.
- Konfigurasi tracker default memakai `botsort.yaml` dan `tracking.with_reid: false`. Jangan menyebut penggunaan ReID kecuali konfigurasi dan eksekusi yang dipakai benar-benar mengaktifkannya.
- Matriks AHP pada konfigurasi adalah matriks yang disusun peneliti dan ditinjau pembimbing; bukan hasil elisitasi multi-pakar. Bobot default adalah konfigurasi sistem, bukan label kebenaran pendidikan.
- Jika validasi fixed-seat identity gagal, ranking student-level diblokir. Jangan mengganti identitas siswa dengan `track_id` tanpa validasi.
- Metrik MOT berbasis ground truth hanya tersedia jika trajectory ground truth dengan format yang sesuai diberikan.

## Pengujian

```bash
python -m compileall -q run_pipeline.py run_experiments.py src tests tools
python -m pytest -q tests
```

Tes unit memeriksa fungsi yang tercakup dalam suite. Tes yang lulus tidak menggantikan evaluasi dengan data/ground truth yang memadai.

## Batas isi ZIP

ZIP memuat source code, konfigurasi, peta kursi, launcher, dan pengujian. File `weights/best.pt`, `data/video_part001.mp4`, dataset gambar/label, dan hasil run besar sengaja tidak dimasukkan; siapkan secara terpisah.
