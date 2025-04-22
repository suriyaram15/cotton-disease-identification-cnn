# 🌿 Cotton Plant Disease Identification using CNN

This project leverages a Convolutional Neural Network (CNN) to identify diseases in cotton plant leaves. The model can classify the following categories:

- Aphids
- Army Worm
- Bacterial Blight
- Powdery Mildew
- Target Spot
- Healthy

---

## 📂 Dataset

The dataset contains approximately **26,100** labeled images of cotton plant leaves affected by various diseases, including healthy samples.

📥 **Download Dataset**: [Kaggle - Cotton Plant Disease Dataset](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease/data)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/suriyaram15/cotton-disease-identification-cnn.git
cd cotton-disease-identification-cnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download and extract the dataset into the following structure:
```
cotton-disease-identification-cnn/
└── dataset/
    └── Main dataset/
        ├── train/
        │   ├── Aphids/
        │   ├── Army worm/
        │   ├── Bacterial Blight/
        │   ├── Powdery Mildew/
        │   ├── Target spot/
        │   └── Healthy/
        └── validation/
            ├── Aphids/
            ├── Army worm/
            ├── Bacterial Blight/
            ├── Powdery Mildew/
            ├── Target spot/
            └── Healthy/
```

---

## 🧠 Model Architecture

The CNN model is designed to be lightweight yet accurate, making it ideal for both web and mobile deployment.

- **4 Convolutional Layers** with ReLU & MaxPooling
- **1 Flatten Layer**
- **1 Dropout Layer** (rate = 0.5)
- **2 Dense Layers**:
  - Dense(512, activation='relu')
  - Dense(output_classes, activation='softmax')

---

## 🚀 Training the Model

Train the CNN using:
```bash
python train.py
```

> ⏱️ Training time may vary depending on hardware (GPU recommended).

---

## 📱 Optional: Convert to TFLite

For mobile or edge deployment:
```bash
python test.py
```

---

## 🔍 Making Predictions

To classify new cotton plant leaf images:
```bash
python predict.py
```

---

## 📦 Requirements

- Python 3.7+
- TensorFlow >= 2.0
- OpenCV
- NumPy
- Matplotlib
- Pillow

Install using:
```bash
pip install tensorflow opencv-python numpy matplotlib pillow
```

Alternatively, use the included `requirements.txt`.

---

## 📊 Performance

- Validation Accuracy: **90% - 95%**
- Optimized for both desktop and mobile environments (when using TFLite)
- Lightweight architecture for efficient inference

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

---

## 🙋‍♂️ Author

Developed by **Suriya Ram S**  
GitHub: [@suriyaram15](https://github.com/suriyaram15)


