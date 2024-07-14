# Fruit Image Classification using Convolutional Neural Network (CNN)

This project focuses on the classification of fruit images using a Convolutional Neural Network (CNN). The dataset consists of 10 different fruit classes, and the goal is to accurately classify the images into their respective categories.

## Dataset

The dataset contains images of 10 different fruits:

- Guava
- Banana
- Peaches
- Mango
- Raspberries
- Others...

The images are preprocessed and resized to 100x100 pixels. The dataset is split into training and testing sets with a ratio of 70:30.

## Dependencies

The following libraries are required to run the project:

- numpy
- pandas
- matplotlib
- cv2 (OpenCV)
- sklearn
- tensorflow
- keras

## Data Preprocessing

### Loading and Preprocessing Images

Images are loaded using OpenCV, resized to 100x100 pixels, and normalized to have values between 0 and 1.

### One-Hot Encoding

The labels are one-hot encoded using OneHotEncoder from sklearn.

## Model Architecture

The CNN model is built using Keras with TensorFlow as the backend. The architecture consists of:

- Input Layer: Input shape of (100, 100, 3)
- Convolutional Layers: Four Conv2D layers with ReLU activation and MaxPooling2D layers
- Fully Connected Layers: Three Dense layers with ReLU activation and Dropout layers to prevent overfitting
- Output Layer: Dense layer with softmax activation for multi-class classification

### Detailed Architecture

```plaintext
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 98, 98, 32)        896       
max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0         
conv2d_1 (Conv2D)            (None, 47, 47, 64)        18496     
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0         
conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856     
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0         
conv2d_3 (Conv2D)            (None, 8, 8, 256)         295168    
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 256)         0         
flatten (Flatten)            (None, 4096)              0         
dense (Dense)                (None, 64)                262208    
dropout (Dropout)            (None, 64)                0         
dense_1 (Dense)              (None, 128)               8320      
dropout_1 (Dropout)          (None, 128)               0         
dense_2 (Dense)              (None, 256)               33024     
dropout_2 (Dropout)          (None, 256)               0         
dense_3 (Dense)              (None, 10)                2570      
=================================================================
Total params: 694538
```

## Training

The model is compiled with the Adam optimizer and categorical cross-entropy loss. The training process includes:

- **Data Augmentation**: Using ImageDataGenerator to augment the training images with rotations, shifts, and flips.
- **Training**: The model is trained for 30 epochs with a batch size of 100.

