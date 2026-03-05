# CNN-FANet sEMG Gesture Classifier

CNN with channel attention for classifying 7 hand gestures from 8-channel surface EMG data.

## Dataset
UCI EMG Data for Gestures: [view here](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
EMG data collected via sEMG sensors, labeled by their corresponding hand gesture.

## Gestures
Rest, Fist, Flexion, Extension, Radial Deviation, Ulnar Deviation, Palm

## Model
Two Conv1d layers for feature extraction followed by a channel attention block (FANet-style), then global average pooling and a fully connected classifier. Input shape: `(Batch, 8, WINDOW_SIZE)`.

## Hyperparameters
Set at the top of the notebook:

| Parameter | Description |
|---|---|
| `WINDOW_TYPE` | `pure` or `majority` labeling strategy |
| `WINDOW_SIZE` | Samples per window |
| `STRIDE` | Step size between windows |
| `LEARNING_RATE` | Adam optimizer LR |
| `EPOCHS` | Training epochs |

## Usage
1. Install dependencies: `torch`, `numpy`, `pandas`, `sklearn`, `seaborn`, `matplotlib`, `tensorboard`. `tensorflow`
2. Place data in `./Data/train` and `./Data/test`
3. Set hyperparameters in Section 0
4. Run all cells top to bottom

## Outputs
- TensorBoard logs: train/test loss per epoch, confusion matrix, MCC score
- Launch TensorBoard: `tensorboard --logdir=logs/`

## Metric
- **Matthews Correlation Coefficient (MCC)** is used over accuracy to account for class imbalance. 
    - Range: -1 to 1
        - 1.0 is perfect classification.
- **Confusion Matrix (CM)** is used to view fine grained misclassifications between gestures.
    - 7x7 matrix
        - 100.0 across diagonal is perfect classification

### Example Confusion Matrix
<img src="./Example_Confusion_Matrix.png">
