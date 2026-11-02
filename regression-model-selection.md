# Pemilihan Algoritma Regresi dan Evaluasi Model

Halo, saya Fakhri Hasabi, mahasiswa semester 5 Teknik Informatika di Universitas Pamulang.
Repo ini berisi catatan belajar saya soal cara memilih algoritma regresi
dan membaca metrik evaluasi (MAE, MSE, R²). Awalnya dibuat buat pegangan
sendiri waktu ngerjain tugas Machine Learning, terus saya rapikan supaya
bisa kepakai orang lain juga.

Kalau ada yang kurang tepat atau mau ditambahin, silakan buka issue
atau pull request.

Milih algoritma regresi itu sering bikin bingung karena pilihannya banyak banget, dari Linear Regression yang simpel sampai Gradient Boosting dan Neural Network. 
Nggak ada yang selalu terbaik, semuanya tergantung karakteristik datanya: jenis fitur, jumlah data, ada outlier atau nggak, dan hubungannya linear atau bukan.

Setelah model jadi, masalah berikutnya adalah baca hasil evaluasinya.
Masih banyak yang mengira MAE dan MSE harus di bawah 1, atau R² yang hampir 1 pasti bagus. 
Padahal MAE dan MSE bergantung pada skala target, dan R² yang terlalu sempurna di data nyata justru patut dicurigai sebagai overfitting.

Karena itu, catatan ini merangkum faktor yang perlu dipertimbangkan saat memilih algoritma, kegunaan tiap algoritma, cara baca MAE, MSE, dan R², serta langkah praktis memilih model. 
Isinya disusun dari beberapa buku dan jurnal (daftarnya ada di bagian referensi) dan dipakai sebagai bahan belajar dan acuan cepat.

Catatan ini saya tulis buat merangkum cara milih algoritma regresi, fungsi tiap algoritma, dan cara baca metrik evaluasi seperti MAE, MSE, dan R².

1. Hal yang Bikin Kita Milih Algoritma Tertentu
Waktu mau bikin model regresi, kita nggak bisa asal pilih algoritma. Ada tiga hal utama yang perlu dicek dulu. Pertama jenis datanya. Kalau datanya banyak outlier, Robust Regression atau Random Forest lebih aman. 
Kalau fitur-fiturnya saling berkorelasi, Ridge atau Elastic Net lebih cocok. Kedua ukuran datasetnya. Data yang kecil biasanya lebih pas pakai model sederhana kayak Linear Regression atau Ridge, sedangkan data yang besar bisa pakai Random Forest, Boosting, atau Neural Network. Ketiga soal linearitas, yaitu hubungan antara fitur dan target itu linear atau bukan. 
Kalau linear, Linear, Ridge, atau Lasso sudah cukup. Kalau non-linear, bisa coba Polynomial, model berbasis pohon, SVR, atau Neural Network.
Selain itu ada dua pertimbangan tambahan. Kalau modelnya harus jalan real-time, pilih yang ringan. 
Kalau dipakai di bidang sensitif seperti medis atau keuangan, pilih model yang gampang dijelaskan karena hasilnya harus bisa dipertanggungjawabkan.

2. Algoritma Regresi dan Kegunaannya

Linear Regression adalah yang paling dasar. Cocok buat hubungan linear dan biasanya dipakai sebagai baseline. 
Modelnya sangat cepat dan paling gampang diinterpretasikan, jadi aman dipakai di domain sensitif. Polynomial Regression dipakai kalau hubungannya non-linear tapi masih sederhana. 
Cocoknya untuk data kecil sampai sedang.

Ridge, Lasso, dan Elastic Net adalah turunan Linear Regression yang pakai penalti. Ridge memakai penalti L2 dan berguna saat fitur banyak atau saling berkorelasi. 
Lasso memakai penalti L1 sehingga bisa mengecilkan koefisien sampai 0, jadi sekaligus berfungsi sebagai seleksi fitur otomatis. 
Elastic Net menggabungkan keduanya dan cocok untuk fitur yang saling berkorelasi. Ketiganya cepat dan masih cukup mudah dijelaskan.

Ada juga beberapa algoritma untuk kasus khusus. Robust Regression (Huber, RANSAC) dipakai kalau datanya penuh outlier. GLM seperti Poisson cocok kalau targetnya berupa hitungan atau distribusinya bukan normal. 
Gaussian Process atau Bayesian Regression dipakai kalau kita butuh estimasi ketidakpastian, tapi biasanya hanya untuk data kecil dan prosesnya lambat.

Yang perlu diingat, nggak ada algoritma yang selalu paling bagus. 
Contohnya studi harga cabai di Banyumas dan Cilacap menunjukkan Lasso paling akurat untuk prediksi jangka pendek, Ridge lebih realistis untuk jangka panjang, dan Random Forest tidak bisa memprediksi nilai di luar rentang data latihnya.

Untuk hubungan non-linear, ada Decision Tree yang gampang divisualisasikan tapi rawan overfitting. Random Forest lebih tahan outlier dan bagus untuk data tabular, tapi sifatnya kotak hitam. 
Gradient Boosting, XGBoost, dan LightGBM biasanya memberi akurasi tinggi di data tabular dan cukup cepat, tapi interpretabilitasnya kurang, meski bisa dibantu dengan SHAP. 
SVR memakai kernel untuk pola non-linear dan cocok untuk data kecil sampai sedang. KNN Regression bagus untuk pola lokal di data kecil, tapi lambat saat prediksi. Neural Network (MLP) dipakai untuk pola yang sangat kompleks dan butuh data besar.

3. Cara Baca Metrik Evaluasi
MAE dan MSE. Dua metrik ini makin kecil makin bagus, dan 0 adalah kondisi ideal. Banyak yang mengira nilainya harus di bawah 1, padahal nggak ada aturan seperti itu karena nilainya tergantung skala target.
MAE satuannya sama dengan target karena ia rata-rata selisih absolut antara prediksi dan nilai asli. MSE satuannya kuadrat dan lebih sensitif terhadap outlier, jadi biasanya dipakai RMSE (akar dari MSE) supaya satuannya balik lagi ke satuan target.
MSE di bawah 1 baru bermakna kalau targetnya sudah distandarisasi atau dinormalisasi. Cara menilainya yang benar adalah dengan membandingkan hasilnya ke model baseline (prediksi rata-rata), ke standar deviasi atau rentang target, dan antar-model di dataset yang sama.

R² (koefisien determinasi). Nilai maksimumnya 1 dan nggak mungkin lebih dari itu. Makin dekat ke 1 makin bagus. Kalau R² = 1, artinya model menjelaskan seluruh variasi data, tapi di data nyata ini justru patut dicurigai sebagai overfitting. 
R² = 0 artinya model sama saja dengan menebak rata-rata. R² juga bisa negatif di data uji kalau modelnya malah lebih jelek daripada tebak rata-rata. Untuk regresi berganda, lebih baik pakai Adjusted R² karena R² biasa pasti naik tiap kali fitur ditambah, walaupun fitur itu nggak berguna. 
Soal berapa nilai yang dianggap “baik”, itu tergantung bidangnya. Ilmu sosial biasanya menerima nilai yang lebih rendah dibanding fisika atau teknik.

4. Langkah Praktis Memilih Model
Alurnya kurang lebih begini. Mulai dari model baseline dulu, misalnya Linear Regression atau prediksi rata-rata. Kalau fiturnya banyak atau saling berkorelasi, lanjut coba Ridge, Lasso, atau Elastic Net.
Kalau hubungannya ternyata non-linear, coba Random Forest atau Gradient Boosting. Setelah itu, bandingkan semua model pakai cross-validation dengan MAE, RMSE, dan R².
Terakhir, sebelum memutuskan, pertimbangkan juga kebutuhan real-time dan seberapa penting model itu harus bisa dijelaskan.

Berikut rangkuman yang tulis di repo ini, dan tentunya setiap kalimat diatas menggunakan referensi tertentu seperti lewat jurnal maupun e-book
terima kasih!! sudah melihat dan membaca repo ini!! :))

Referensi
Mubaroka, F. R., Fitriani, M. A., Pambudi, E. A., & Hakim, D. K. (2026). Chili Price Prediction Model in Banyumas and Cilacap Regencies Using Comparative Machine Learning Algorithm. Jurnal E-Komtek, 10(1), 60-72. https://doi.org/10.37339/e-komtek.v10i1.3003
Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems (NeurIPS 2022).
Rajan, M. P. (2022). An Efficient Ridge Regression Algorithm with Parameter Estimation for Data Analysis in Machine Learning. SN Computer Science, 3(2). https://doi.org/10.1007/s42979-022-01051-x
Zeng, Z. (2025). Using Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression for Predicting Real Estate Price. ICDSE, 509-515. https://doi.org/10.5220/0013700300004670
Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O’Reilly Media.
