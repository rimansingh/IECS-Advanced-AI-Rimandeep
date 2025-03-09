# Deep Learning Notebooks

This repository contains a series of Jupyter notebooks designed to introduce and explore various deep learning concepts using PyTorch. The notebooks follow a structured approach, covering fundamental topics such as basic neural networks, convolutions, recurrent networks, and hyperparameter tuning.

| Folder | Description     |
|--------|-----------------|
| **1_pytorch_intro**      | This folder contains exercises that introduce basic concepts in PyTorch, including the creation of custom datasets, implementation of datastreamers, and model tuning techniques. |
| **2_convolutions**       | Focuses on convolutional neural networks, demonstrating how to add dropout and normalization layers to improve model performance. |
| **3_recurrent_networks** | Covers sequential models such as RNNs, LSTMs, and GRUs, along with their applications in text and time-series data. |
| **4_tuning_networks**    | Discusses hyperparameter tuning techniques, including experiments with batch size, learning rate, and various optimization methods. |

### **1. Prerequisites**
Ensure you have the following installed:
- Python (>=3.8)
- Jupyter Notebook
- PyTorch
- NumPy, Matplotlib
- Other dependencies: mads_datasets, mltrainer, gin-config, torch.optim, etc.

You can install the required packages using:

```sh
pip install torch torchvision torchaudio jupyter numpy matplotlib gin-config
```

### **2. Running the Notebooks**
To run the notebooks locally:
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
3. Open the notebook you want to work on and execute the cells.


### **Important Note**
Make sure you have the `mltrainer` library version `0.1.128` installed, as any other version did not work on my machine.

---

Happy coding! 😊
