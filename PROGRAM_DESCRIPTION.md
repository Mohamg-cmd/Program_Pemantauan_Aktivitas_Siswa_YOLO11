# Deskripsi Program

**Nama:** Program Pemantauan Aktivitas Siswa Berbasis YOLO11

**Tujuan:** menyediakan pipeline perangkat lunak untuk mengidentifikasi kelas aktivitas siswa dari video kelas, menjaga atribusi aktivitas pada identitas siswa menggunakan geometri kursi tetap, mengagregasi perilaku secara temporal, dan menyusun ringkasan keputusan yang dapat diperiksa.

**Fungsi utama:**
- Deteksi kelas aktivitas memakai model YOLO11.
- Pelacakan lintasan objek memakai BoT-SORT.
- Asosiasi identitas berbasis fixed-seat dengan validasi konflik dan penggabungan fragmen track.
- Agregasi durasi/frekuensi aktivitas dan transisi antaraktivitas.
- AHP untuk bobot kriteria, SAW untuk ranking, serta opsi perbandingan SAW–WP.
- Sensitivitas one-at-a-time, Monte Carlo, ablation, analisis temporal, propagasi error, dan laporan Excel.

**Batasan:** program tidak otomatis melatih model detector; checkpoint harus disediakan. Data kursi terikat pada tampilan kamera tertentu dan wajib diperiksa ulang bila kamera/ruang berubah. `track_id` merupakan ID lintasan dan tidak identik dengan `student_id`. Tanpa ground-truth trajectory, hasil tidak boleh dipresentasikan sebagai metrik akurasi MOT.
