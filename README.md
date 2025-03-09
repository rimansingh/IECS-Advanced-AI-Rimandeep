# Deep Learning Notebooks

This repository contains a series of Jupyter notebooks designed to introduce and explore various deep learning concepts using PyTorch. The notebooks follow a structured approach, covering fundamental topics such as basic neural networks, convolutions, recurrent networks, and hyperparameter tuning.


| Folder | Description     |
|--------|-----------------|
| **1_pytorch_intro**      |
| **2_convolutions**       |
| **3_recurrent_networks** |
| **4_tuning_networks**    |
|--------------------------|


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


### **1-PyTorch-Intro**
#### Summary of Exercises

    ##### 1. 3D Tensor Dataset
    - Created a custom dataset class named `Random3DTensorDataset` that extends PyTorch's `Dataset`.
    - This dataset generates random 3D tensors and corresponding random binary labels.
    - Demonstrated the creation of a dataset and DataLoader, and fetched one batch of tensors and labels.

    ##### 2. Datastreamers
    - Implemented a base class `BaseDatastreamer` for streaming data in batches.
    - Created a datastreamer with a limit of 5 batches, showcasing the use of a batch processor.

    ##### 3. Network Tuning
    - Included imports for model training and configuration using `gin`.
    - Demonstrated the setup for training a model, including logging and validation accuracy outputs.

---

### **2_convolutions**
#### Summary of Exercises

    ##### 1: Adding Dropout and Normalization Layers in PyTorch

### **3_recurrent_networks**
- Introduction to sequential models like RNNs, LSTMs, and GRUs.
- Example applications for text and time-series data.

### **4_tuning_networks**
- Covers techniques for hyperparameter tuning.
- Experiments with batch size, learning rate, and optimization methods.


---

Happy coding! �

