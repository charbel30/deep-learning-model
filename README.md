<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br></h1>
<h3>◦ Code together, conquer forever!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy" />
</p>
</div>

---

##  Table of Contents
- [ Table of Contents](#-table-of-contents)
- [ Overview](#-overview)
- [ Features](#-features)
- [ repository Structure](#-repository-structure)
- [ Modules](#modules)
- [ Getting Started](#-getting-started)
    - [ Installation](#-installation)
    - [ Running ](#-running-)
    - [ Tests](#-tests)
- [ Roadmap](#-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---


##  Overview

Exception: 

---

##  Features

Exception: 

---


##  Repository Structure

```sh
└── /
    └── Single-visit/
        └── deep learning model/
            ├── fast_pipe.ipynb
            ├── models/
            ├── pipeline.ipynb
            ├── predict_model.py
            ├── requirements.txt
            └── utils/

```

---


##  Modules

<details closed><summary>Root</summary>

| File                                 | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                  | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [fast_pipe.ipynb]({file})            | The code above is a Jupyter Notebook file called `fast_pipe.ipynb` located in the directory `Single-visit/deep learning model/`. It imports various libraries and modules for data preprocessing, model training, and evaluation. It reads a CSV file (`dukecathr.csv`), sets the device to either CUDA or CPU for training, and initializes variables for storing metrics. It also defines a function `train_epoch` for training the model using a given dataloader, criterion, and optimizer. The function trains the model, computes the loss and accuracy, updates the optimizer, and returns the running loss.                                   |
| [pipeline.ipynb]({file})             | The code in the "pipeline.ipynb" notebook trains a deep learning model using the Duke-Catheter data. It imports necessary libraries and modules, reads the data, and initializes variables. It also defines a function called "train_epoch" that trains the model for one epoch. The function uses a provided dataloader, loss function, optimizer, learning rate scheduler, and warm-up scheduler to train the model and calculate the training loss and accuracy. The goal is to train the model on the data and evaluate its performance.                                                                                                          |
| [predict_model.py]({file})           | The code is a function called `test_model` that evaluates the performance of a pre-trained deep learning model on a test dataset. It loads the saved model and test data, sets the input size of the model, creates a DataLoader for the test data, makes predictions on the test set, and calculates metrics such as accuracy, precision, recall, and balanced accuracy. The function returns these metrics along with the target variable name.                                                                                                                                                                                                     |
| [requirements.txt]({file})           | The code presents a requirements.txt file that specifies the dependencies for a deep learning model. These dependencies include imbalanced_learn, imblearn, matplotlib, numpy, pandas, pytorch_warmup, scikit_learn, seaborn, torch, and tqdm. These dependencies are necessary for running the pipeline and predicting the model accurately.                                                                                                                                                                                                                                                                                                         |
| [high_complexity_model.py]({file})   | This code defines a deep learning model called `Net` that consists of several fully connected layers. The purpose of this model is to prevent underfitting by using dropout and batch normalization techniques. It takes an input of `input_size` and outputs a single value. The model architecture includes a series of linear transformations, relu activation functions, dropout layers, and batch normalization layers. The final output is obtained by using a sigmoid activation function on the last layer.                                                                                                                                   |
| [medium_complexity_model.py]({file}) | This code defines a feedforward neural network model using the PyTorch library. The model consists of five fully connected (linear) layers with batch normalization and dropout layers. The network applies the ReLU activation function to the hidden layers and the sigmoid activation function to the output layer for binary classification. This particular implementation is best suited for medium complexity classification tasks.                                                                                                                                                                                                            |
| [simple_model.py]({file})            | The code defines a simple feed-forward neural network model, specifically a class called "Net". It includes two fully connected layers and a dropout layer for regularization. The model is designed to address overfitting and aims to produce accurate predictions on a test set. The code also utilizes the torch library for deep learning operations.                                                                                                                                                                                                                                                                                            |
| [EarlyStopping.py]({file})           | The code is an implementation of the Early Stopping technique in deep learning. It defines the EarlyStopping class, which is used to check the validation loss of a model during training and determine whether to stop or continue training based on a set of parameters. The class saves the model if the validation loss decreases and provides an option to set a patience value for stopping criteria. It also includes a save_checkpoint method to save the model state.                                                                                                                                                                        |
| [heatmap_plots.py]({file})           | The code provides two functions for plotting metrics of a deep learning model. 1. The `plot_heat_map` function takes a DataFrame of results and plots a heatmap of scores for each target. It can plot all scores or specify a subset of scores. The function can plot each score in a separate graph or combine them into a single graph.2. The `plot_metrics` function takes a dictionary of metrics and a list of plots to generate. It can plot metrics such as accuracy, loss, balanced accuracy, precision, and recall. Each plot is generated on a separate subplot.Both functions use the matplotlib and seaborn libraries for visualization. |
| [Preprocessing_utils.py]({file})     | Exception:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

</details>

---

##  Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- ℹ️ Dependency 1`

`- ℹ️ Dependency 2`

`- ℹ️ ...`

###  Installation

1. Clone the  repository:
```sh
git clone ../
```

2. Change to the project directory:
```sh
cd 
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

###  Running 

```sh
python main.py
```

###  Tests
```sh
pytest
```

---


##  Project Roadmap

> - [X] `ℹ️  Task 1: Implement X`
> - [ ] `ℹ️  Task 2: Implement Y`
> - [ ] `ℹ️ ...`


---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/local//blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/local//discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/local//issues)**: Submit bugs found or log feature requests for LOCAL.

#### *Contributing Guidelines*

<details closed>
<summary>Click to expand</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone <your-forked-repo-url>
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear and concise message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License


This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#Top)

---

