# 🌾 Perbandingan Kinerja Arsitektur EfficientNet-B4 dan Inception-V3 pada Klasifikasi Hama Tanaman Padi

Repository ini merupakan implementasi dari penelitian skripsi yang membandingkan kinerja dua arsitektur *Convolutional Neural Network* (CNN), yaitu **EfficientNet-B4** dan **Inception-V3**, dalam mengklasifikasikan hama tanaman padi menggunakan **dataset RP11 (subset dari IP102)**. Penelitian ini bertujuan untuk mengevaluasi kemampuan kedua model dalam mengenali 11 kelas hama berdasarkan citra digital melalui pendekatan *transfer learning*.

Eksperimen dilakukan dengan menerapkan tiga skenario fine tuning, yaitu **Feature Extraction**, **Partial Fine-Tuning**, dan **Full Fine-Tuning**. Selain itu, setiap strategi diuji pada dua skenario pelatihan, yaitu **dengan augmentasi data** dan **tanpa augmentasi data**, untuk menganalisis pengaruhnya terhadap performa model. Evaluasi dilakukan menggunakan metrik **Accuracy, Precision, Recall,** dan **F1-score**, serta dilengkapi dengan analisis kesalahan (*error analysis*) dan pendekatan **Explainable Artificial Intelligence (XAI)** menggunakan **GradCAM++** untuk memvisualisasikan area citra yang menjadi dasar pengambilan keputusan model. 

<p align="center">

![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RP11_(Subset_IP102)-blue?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Rice_Pest_Classification-success?style=for-the-badge)

</p>
