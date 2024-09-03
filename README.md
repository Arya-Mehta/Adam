# Custom Adam Optimizer in PyTorch

## Overview

This repository contains a custom implementation of the Adam optimizer from scratch in PyTorch. Adam (Adaptive Moment Estimation) is one of the most widely used optimization algorithms in deep learning due to its adaptive learning rate capabilities. The implementation closely follows the concepts introduced in the original research paper while integrating it into the PyTorch framework for seamless compatibility.

## Features

- **Custom Adam Optimizer**: Implemented entirely from scratch, following the mathematical foundations of the Adam optimizer algorithm.
  <img width="847" alt="image" src="https://github.com/user-attachments/assets/7fe9341c-358e-42bf-a215-28ada6c134eb">
- **Datasets Used**: Tested on the following dataset from SciKit:
  - Digit Dataset
- **Performance Comparison**: Benchmarked against PyTorch's built-in optimizers:
  - Stochastic Gradient Descent (SGD)
  - Inbuilt Adam Optimizer
- **Results**:
  - Learning Rate (Step Size) is set at = 0.09
  - Number of epochs are set at = 200
  - SGD: 
    <img width="424" alt="image" src="https://github.com/user-attachments/assets/89db27e9-118e-4d62-a9a3-5c235ff9e479">
  - Inbuilt Adam: 
    <img width="415" alt="image" src="https://github.com/user-attachments/assets/75b1431c-3157-4311-9c6c-1aea45f961a7">
  - Custom Implemented Adam: 
    <img width="474" alt="image" src="https://github.com/user-attachments/assets/9e2cf12a-09d9-496f-b82f-9ad6f1d1a4fa">
- **Conclusion**
    - Even after 200 epochs, SGD still concurs a loss of 0.2377
    - Meanwhile PyTorch's Adam and Custom Adam both almost converge to the minimum weights after around only 50/60 epochs.
    - The results also showcase the accuracy of the Custom Adam comapred to the inbuilt Adam, with both almost having around the same accuracy of about 95%.




