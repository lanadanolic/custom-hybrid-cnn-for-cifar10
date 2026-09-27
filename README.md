# Custom Hybrid CNN for CIFAR-10

This project presents a custom Hybrid CNN architecture that combines standard, depthwise, and dilated convolutions within a single model.

The proposed Hybrid CNN is compared with three baseline architectures:
- Standard CNN
- Depthwise CNN
- Dilated CNN

The goal is to evaluate whether combining different convolution types can improve image classification performance on the CIFAR-10 dataset.

## Installation

Install the required dependencies using:

```bash
python -m pip install -r requirements.txt
```

If TensorFlow is not installed, install it separately using:

```bash
python -m pip install tensorflow
```