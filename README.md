
```markdown
# Image Classification with Transformer Models

This repository demonstrates how to use a Transformer model to perform image classification by leveraging patch-based image representation. The dataset is split into multiple categories, and the model classifies images into one of the defined classes using a patch-based approach.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

This notebook implements a Transformer-based model to classify images from a custom dataset into multiple predefined categories. The image dataset is first split into patches, and these patches are fed into a Transformer model for classification. The implementation is based on the `EOTransformerModel` that uses a multi-layer Transformer encoder for processing image patches.

### Key Features:
- Image dataset processing by splitting images into patches.
- Custom `EOTransformerModel` built using Transformer encoder layers.
- Training using standard classification loss (cross-entropy).
- Model evaluation with various classification metrics including accuracy, precision, recall, F1 score, and ROC-AUC.

## Requirements

The following libraries are required to run this code:

- `torch` (PyTorch)
- `numpy`
- `PIL` (Pillow)
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the dependencies with:

```bash
pip install torch numpy pillow matplotlib seaborn scikit-learn
```

## Dataset

The dataset consists of images from multiple categories. Each image belongs to one of the following classes:

- `addisplay/airpush`
- `adware++virus/kuguo`
- `downloader/jiagu`
- `troj/artemis`
- `smssend/koomer`

### Dataset Directory Structure

```plaintext
dataset/
    ├── addisplay/airpush/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── adware++virus/kuguo/
    │   ├── image1.png
    │   └── ...
    ├── downloader/jiagu/
    ├── troj/artemis/
    └── smssend/koomer/
```

Make sure that the images are placed in folders that correspond to their labels. For example, images in the `addisplay/airpush` folder should belong to the `addisplay/airpush` class.

## Model Architecture

The main model used in this notebook is a custom implementation of a Transformer-based architecture for image classification. The architecture consists of the following components:

### EOTransformerModel

- **Input Preparation**: Images are divided into patches (16x16 pixels).
- **Positional Encoding**: Added to the input patches to maintain spatial information.
- **Transformer Encoder**: The patches are processed using multi-layer Transformer encoder blocks.
- **Fully Connected Layer**: After processing by the Transformer encoder, the output is passed through a fully connected layer to predict the class label.

```python
class EOTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=16, dropout=0.1, max_len=256, device='cpu'):
        super(EOTransformerModel, self).__init__()
        ...
```

### FullTransformerModel

This model includes both the encoder and decoder layers of the Transformer. It is an extension of the `EOTransformerModel` and can be used for more complex tasks.

```python
class FullTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=16, dropout=0.1, max_len=256):
        super(FullTransformerModel, self).__init__()
        ...
```

## Training

### Hyperparameters
- `batch_size`: 32
- `learning_rate`: 1e-4
- `num_epochs`: 5
- `num_classes`: Number of classes in the dataset

The model is trained using the following procedure:
1. **DataLoader**: The dataset is loaded and divided into batches.
2. **Model Training**: The model is trained using the cross-entropy loss function and the Adam optimizer.
3. **Metrics**: During training, both loss and accuracy are logged for each epoch.

### Training Code Snippet:

```python
# Define model and optimizer
model = EOTransformerModel(input_dim=16*16*3, num_classes=len(label_mapping), device=device)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Train the model
history = train_model(model, train_dataloader, optimizer, num_epochs=num_epochs, device=device)
```

## Evaluation

After training the model, it is evaluated using the `evaluate_model_with_patching()` function. This function computes the following metrics:

- **Accuracy**
- **Precision** (Macro)
- **Recall** (Macro)
- **F1 Score** (Macro)

Additionally, the following metrics are computed using the ROC-AUC score:

- **ROC-AUC** (Macro)
- **Confusion Matrix**

### Evaluation Code Snippet:

```python
test_dataset = ImageDataset(image_dir=cwd + data_src_dir + test_dir, label_map=label_mapping, patch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model
eometrics = evaluate_model_with_patching(model, test_dataloader)
```

### Evaluation Results

After evaluating the model, the following results are displayed:

- **Accuracy**: The percentage of correctly classified samples.
- **Precision (Macro)**: The average precision across all classes.
- **Recall (Macro)**: The average recall across all classes.
- **F1 Score (Macro)**: The average F1 score across all classes.
- **ROC-AUC (Macro)**: The average ROC-AUC score across all classes.
- **Confusion Matrix**: Visualized matrix showing true vs predicted labels.

## Usage

To use the code, follow these steps:

1. **Prepare Dataset**: Organize your image dataset as described in the [Dataset](#dataset) section.
2. **Set Hyperparameters**: Modify the hyperparameters (e.g., `batch_size`, `learning_rate`, `num_epochs`) according to your needs.
3. **Run Training**: Run the training code to train the model.
4. **Evaluate the Model**: After training, evaluate the model on the test dataset.
5. **Visualize Results**: View the confusion matrix and other evaluation metrics.

## Results

After training the model and evaluating it on the test set, you will get performance metrics like accuracy, precision, recall, F1-score, and ROC-AUC. You can visualize these metrics in plots and confusion matrices.

### Example Output:

```plaintext
Epoch [1/5], Loss: 1.4323, Accuracy: 0.6250
Epoch [2/5], Loss: 1.1234, Accuracy: 0.7352
Epoch [3/5], Loss: 0.9856, Accuracy: 0.7608
Epoch [4/5], Loss: 0.8934, Accuracy: 0.7725
Epoch [5/5], Loss: 0.7867, Accuracy: 0.7854
```

## Future Improvements

- **Data Augmentation**: Introduce data augmentation techniques to improve generalization.
- **Model Optimization**: Experiment with more advanced models like Vision Transformers (ViT) for better performance.
- **Hyperparameter Tuning**: Further tune hyperparameters such as learning rate, batch size, and the number of layers in the Transformer.
- **Real-time Inference**: Optimize the model for faster inference on large image datasets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` provides a complete guide to your image classification project with Transformer models. It includes sections on the dataset, model architecture, training process, evaluation, usage, results, and future improvements.

```
