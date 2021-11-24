# Using Self-Supervised Pretext Tasks for Active Learning - Official Pytorch Implementation

## Experiment Setting:
- CIFAR10 (downloaded and saved in ```./DATA```
- Rotation prediction for pretext task

## Prerequisites:
Python >= 3.7

CUDA = 11.0

PyTorch = 1.7.1

numpy >= 1.16.0

## Running the Code

To train the rotation predition task on the unlabeled set:
```
python rotation.py
```

To extract pretext task losses and create batches:
```
python make_batches.py
```

To evaluate on active learning task:
```
python main.py
```
