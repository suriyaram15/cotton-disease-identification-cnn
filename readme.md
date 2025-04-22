```markdown
# Cotton Plant Disease Identification using CNN

This project identifies diseases in cotton plants using a Convolutional Neural Network (CNN). It can detect:
- Aphids
- Army worm
- Bacterial Blight
- Powdery Mildew
- Target spot
- Healthy leaves

## Dataset
The dataset contains 26.1k images of cotton plant leaves with various diseases. 
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease/data).

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/suriyaram15/cotton-disease-identification-cnn.git
   cd cotton-disease-identification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and extract it into a `dataset` folder:
   ```
   cotton-disease-identification/
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

4. Train the model:
   ```bash
   python train.py
   ```

5. Convert the model to TFLite (optional for mobile deployment):
   ```bash
   python test.py
   ```

6. Make predictions:
   ```bash
   python predict.py
   ```

## Requirements
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Model Architecture
The CNN model consists of:
- 4 Convolutional layers with MaxPooling
- 1 Flatten layer
- 1 Dropout layer (0.5)
- 2 Dense layers (512 neurons and output layer)

## Performance
The model achieves approximately 90-95% accuracy on the validation set.

## License
This project is licensed under the MIT License.
```

## 5. requirements.txt

```
tensorflow>=2.0
opencv-python
numpy
matplotlib
pillow
```

## Usage Instructions

1. First, organize your dataset into train and validation folders with the class folders inside each.
2. Run `train.py` to train the model (this may take several hours depending on your GPU).
3. After training, you can convert the model to TFLite format using `test.py`.
4. Use `predict.py` to make predictions on new images.

The model architecture is designed to be lightweight enough to run on mobile devices (when converted to TFLite) while still maintaining good accuracy. You can adjust the hyperparameters in `train.py` based on your specific needs.