Tire Condition Detection using Convolutional Neural Networks (CNN)

An end-to-end **deep learning–based image classification system** for detecting tire conditions from images.
This project uses a **Convolutional Neural Network (CNN)** trained with TensorFlow/Keras and is deployed as an interactive **Streamlit web application**.

The system classifies tire images into **three categories**:

* **Flat**
* **Full**
* **No-Tire**

---

## 📌 Project Overview

Tire condition plays a crucial role in vehicle safety. Manual inspection is time-consuming and inconsistent, while sensor-based solutions require additional hardware.
This project proposes an **image-based solution** using deep learning to automatically classify tire conditions from RGB images.

**Key highlights:**

* CNN-based image classification
* Data augmentation for better generalization
* Softmax-based decision with confidence score
* Deployed as a Streamlit web application

---

## 🧠 Model & Methodology

### Model Architecture

The CNN architecture consists of:

* 3 convolutional blocks with increasing filters (32 → 64 → 128)
* Max pooling layers for spatial downsampling
* Fully connected layer with dropout regularization
* Softmax output layer for multi-class classification

**Total parameters:** ~12.9 million

### Training Setup

* Image size: **240 × 240**
* Optimizer: **Adam**
* Loss function: **Categorical Cross-Entropy**
* Epochs: **10**
* Train / Validation split: **80% / 20%**

### Decision Rule

The model outputs class probabilities using **Softmax**:

[
p_i = \frac{e^{z_i}}{\sum_{j=1}^{3} e^{z_j}}
]

The final prediction is selected using:
[
\hat{y} = \arg\max(p)
]

The confidence score is the highest softmax probability.

---

## 📂 Project Structure

```
DL_AOL/
│
├── app/
│   └── app.py                  # Streamlit application
│
├── config/
│   └── config.yaml             # Configuration file
│
├── data/
│   └── raw/
│       ├── flat.class/
│       ├── full.class/
│       └── no-tire.class/
│
├── notebooks/
│   └── experiments.ipynb       # Training & experiments
│
├── outputs/
│   └── models/
│       └── tire_cnn.h5         # Trained CNN model
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── utils.py                # Preprocessing utilities
│
├── report/
│   └── final_reports.pdf
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Application

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd DL_AOL
```

---

### 2️⃣ Create & Activate Virtual Environment (Recommended)

**Using Conda**

```bash
conda create -n tire_cnn python=3.9 -y
conda activate tire_cnn
```

**OR using venv**

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Verify Configuration

Make sure `config/config.yaml` is correctly set:

```yaml
img_size: 240
batch_size: 32
epochs: 10
num_classes: 3

data_dir: "data/raw"
model_path: "outputs/models/tire_cnn.h5"

class_names:
  - flat
  - full
  - no-tire
```

---

### 5️⃣ Run the Streamlit App

⚠️ **Run this command from the project root (`DL_AOL`)**

```bash
streamlit run app/app.py
```

Then open:

```
http://localhost:8501
```

---

## 🖥️ Application Demo

Due to GitHub file size limitations, the demo video is hosted externally.

🎥 **Demo Video (Google Drive)**
👉(https://drive.google.com/file/d/1a2WvTVr22QpxtNe0iIirI0IuamCD8LE5/view?usp=sharing)

---

## 📊 Results

* Training accuracy reached **~98%**
* Validation accuracy peaked at **~90%**
* Loss and accuracy curves indicate stable convergence
* Mild overfitting observed in later epochs, suggesting early stopping could further improve performance

---

## ⚠️ Limitations

* Limited dataset size
* Sensitivity to extreme lighting or occlusions
* Softmax confidence is not fully calibrated

---

## 🔮 Future Work

* Apply **transfer learning** (MobileNet / EfficientNet)
* Add **Grad-CAM** visualization for explainability
* Implement **early stopping** and model checkpointing
* Expand dataset for improved robustness
* Deploy to **Streamlit Cloud**

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy, Matplotlib**
* **Streamlit**
* **YAML**
* **Git & GitHub**

---

## 📄 License

This project is for **academic and educational purposes**.
Feel free to use and adapt with proper attribution.

---

## 👤 Author

**Owen Figo**
Computer Science Student
BINUS University

📧 Email: owen.26.figo@gmail.com
🔗 GitHub: https://github.com/owen-figo
🔗 LinkedIn: https://linkedin.com/in/owenfigo


