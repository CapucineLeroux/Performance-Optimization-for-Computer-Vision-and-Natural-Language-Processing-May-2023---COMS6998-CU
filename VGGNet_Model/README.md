# Optimizing VGG model with PyTorch on CIFAR-10

Code adapted from https://github.com/kuangliu/pytorch-cifar

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Optimize
```
# Run optimization script wth: 
python main.py
```

## Results
Optimal number of workers is 4. Quantization reduced model size from 37 MB to 9 MB. Torchscripting sped up inference by 2.3x



