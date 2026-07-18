# 🌾 Perbandingan Kinerja Arsitektur EfficientNet-B4 dan Inception-V3 pada Klasifikasi Hama Tanaman Padi

Repository ini merupakan implementasi dari penelitian skripsi yang membandingkan kinerja dua arsitektur *Convolutional Neural Network* (CNN), yaitu **EfficientNet-B4** dan **Inception-V3**, dalam mengklasifikasikan hama tanaman padi menggunakan **dataset RP11 (subset dari IP102)**. Penelitian ini bertujuan untuk mengevaluasi kemampuan kedua model dalam mengenali 11 kelas hama berdasarkan citra digital melalui pendekatan *transfer learning*.

Eksperimen dilakukan dengan menerapkan tiga skenario fine tuning, yaitu **Feature Extraction**, **Partial Fine-Tuning**, dan **Full Fine-Tuning**. Selain itu, setiap strategi diuji pada dua skenario pelatihan, yaitu **dengan augmentasi data** dan **tanpa augmentasi data**, untuk menganalisis pengaruhnya terhadap performa model. Evaluasi dilakukan menggunakan metrik **Accuracy, Precision, Recall,** dan **F1-score**, serta dilengkapi dengan analisis kesalahan (*error analysis*) dan pendekatan **Explainable Artificial Intelligence (XAI)** menggunakan **GradCAM++** untuk memvisualisasikan area citra yang menjadi dasar pengambilan keputusan model. 

## 🛠️ Tech Stack & Tools
<p align="center">

![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RP11_(Subset_IP102)-blue?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Rice_Pest_Classification-success?style=for-the-badge)

</p>

<p align="center">

![TorchVision](https://img.shields.io/badge/TorchVision-0.26-orange?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-2.0.2-013243?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?style=flat-square&logo=pandas)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=flat-square&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Evaluation-F7931E?style=flat-square&logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)
![GradCAM++](https://img.shields.io/badge/XAI-GradCAM++-purple?style=flat-square)

</p>
