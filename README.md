

# Lung Nodule Classification with Deep Learning

## file information

video.mp4 - contains the explanation of the code 

architecture.png - shows the model architecture 

Lung_cancer(drug-hackathon).ipynb - notebook containing code with each line being commented out to define its function

## Overview

This repository presents an advanced Deep Learning (DL) model for the classification of lung nodules. The primary objective is to distinguish between benign and malignant entities with exceptional accuracy, outperforming traditional diagnostic methods. The model is constructed using the powerful EfficientNetB3 architecture, a state-of-the-art DL approach.

## Problem Statement

In the realm of medical imaging, this project addresses the challenging task of classifying lung nodules. The goal is to develop sophisticated algorithms that significantly improve the efficiency and speed of lung cancer detection. The dataset comprises images of lung nodules categorized into adenocarcinoma, large cell carcinoma, normal, and squamous cell carcinoma.

## Dataset

The dataset is organized into training and test sets, with distinct subdirectories for each class. The images undergo preprocessing, resizing, and augmentation to enhance the model's generalization. The dataset walkthrough section in the code provides insights into the structure and utilization of the dataset.

```
# Loading and Preprocessing Data
X_train = []
y_train = []
image_size = 224

# Loop through each label in the training data
for i in labels:
    folderPath = os.path.join('/kaggle/input/l01-lung-cancer/Data/train', i)
    
    # Check if the directory exists
    if os.path.exists(folderPath):
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath, j))
            img = cv2.resize(img, (image_size, image_size))
            X_train.append(img)
            y_train.append(i)
# Similar loop for test data
# ...
```

## Model Architecture

In contrast to EfficientNetB0, the model exploits the superior capabilities of EfficientNetB3. This architecture, pretrained on ImageNet, is customized for our specific classification task. It integrates global average pooling, dropout for regularization, and a dense softmax layer for the final output.

```
# Model Architecture
effnet = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
```

## Training Process

The training process encompasses loading and preprocessing the dataset, shuffling the data, and splitting it into training and testing sets. Data augmentation techniques are applied to enhance the model's generalization. The model is compiled with categorical cross-entropy loss and the Adam optimizer. Key callbacks, including early stopping and model checkpointing, ensure optimal training.

```
# Model Compilation
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("lung_nodule_model.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Model Training
history = model.fit(X_train, y_train, validation_split=0.2, epochs=12, verbose=1, batch_size=32,
                    callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping])
```

## Performance Evaluation

The model's performance is meticulously evaluated on the test set using metrics such as accuracy, precision, recall, and the confusion matrix. This step provides a comprehensive assessment of how well the model distinguishes between different lung nodule classes.

```
# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report and Confusion Matrix
print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))
```

## Model Visualization

The repository includes code to visualize the model architecture using the `plot_model` function from TensorFlow's Keras library. This diagram provides a detailed overview of the model's structure, including the different layers and their connections.

```
# Model Visualization
from tensorflow.keras.utils import plot_model

# Plot the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
```

## Conclusion

The lung nodule classification challenge addresses the critical need for advanced algorithms in medical imaging. The choice of EfficientNetB3 and meticulous training processes contributes to a model that aims for accuracy and efficiency in lung cancer detection. The detailed walkthrough, code comments, and visualizations provide a comprehensive understanding of the implemented solution.



---
