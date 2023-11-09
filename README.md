<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br></h1>
<h3>◦ Code better, collaborate faster!</h3>
<h3>◦ Developed with the software and tools below.</h3>
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

---


##  Overview

The repository contains a deep learning model for single-visit analysis. It includes notebooks for data preprocessing, model building, and evaluation. The code enables fast execution without evaluation on each epoch. Other features include early stopping, data balancing, and learning rate scheduling. The model can be tested using the predict_model.py file, which calculates various metrics. The utils folder contains modules for preprocessing, plotting heatmaps, and implementing early stopping. Overall, the repository provides a streamlined pipeline for deep learning analysis on single-visit data.

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

| File                                 | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---                                  | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [fast_pipe.ipynb]({file})            | The code in the fast_pipe.ipynb file is a streamlined version of a deep learning model pipeline. It imports necessary libraries and modules, including pandas, numpy, torch, and sklearn. It also includes functions for preprocessing and plotting. The code does not run evaluation on each epoch, making it faster than the main pipeline. The code also includes a function to test the model using the predict_model.py file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [pipeline.ipynb]({file})             | The code in the'pipeline.ipynb' notebook imports various libraries and modules necessary for data preprocessing, model building, and evaluation. It includes functions for importing and preparing data, defining machine learning models, creating data loaders, calculating metrics, visualizing results, and testing the trained model. Additional utilities such as early stopping, data balancing techniques, and learning rate scheduling are also utilized.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [predict_model.py]({file})           | The code above defines a function called `test_model` that evaluates the performance of a deep learning model on a test dataset. The function loads a trained model and test data, creates a data loader for the test data, and makes predictions on the test set. It then calculates various metrics including accuracy, precision, recall, and balanced accuracy.The function returns these metrics along with the target variable used for evaluation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [requirements.txt]({file})           | The code above represents a directory tree structure containing a deep learning model. The "requirements.txt" file inside the "deep learning model" directory specifies the versions of various Python libraries necessary to run the model. These libraries include imbalanced_learn, imblearn, matplotlib, numpy, pandas, pytorch_warmup, scikit_learn, seaborn, torch, and tqdm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [high_complexity_model.py]({file})   | The code above defines a deep learning model called Net, which is a feedforward neural network. It consists of multiple fully connected layers with batch normalization and dropout layers in between. The model takes an input of specified size and passes it through the layers using the ReLU activation function. The final output is obtained by applying the sigmoid activation function to the last layer. The purpose of this model is to prevent underfitting and improve model performance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [medium_complexity_model.py]({file}) | The code represents a deep learning model called "Medium Complexity Model" implemented using PyTorch. The model consists of multiple fully connected (linear) layers with batch normalization and dropout layers for regularization. The model performs forward propagation using the ReLU activation function for the hidden layers and sigmoid activation for binary classification. The purpose of the model is to make predictions based on input data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [simple_model.py]({file})            | The code defines a simple feedforward neural network model, called Net, which is used to address overfitting issues. The model takes an input of size'input_size' and consists of two fully connected layers with 64 and 1 neurons respectively. It also includes a dropout layer and batch normalization. The forward function applies the necessary operations to the input and returns the output predictions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [EarlyStopping.py]({file})           | The code implements an "EarlyStopping" class that provides functionality for early stopping during model training. It tracks the validation loss and checks if it improves over time. If the validation loss does not improve for a specified number of consecutive iterations, it stops the training early. The class also saves the model with the lowest validation loss during training. The class is initialized with parameters such as patience (number of consecutive iterations without improvement), verbose (whether to print messages), delta (minimum improvement required), and model_path (where to save the model).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [heatmap_plots.py]({file})           | The code provides two functions for plotting metrics of a deep learning model. The'plot_heat_map' function takes a dataframe of results and plots heatmaps of specified scores (e.g., Balanced Accuracy, Precision, Recall) for each target. Users can choose to plot each score type separately or all in one graph.The'plot_metrics' function takes a dictionary of metrics (e.g., accuracy, loss, balanced accuracy) and a list of plots to generate. It plots the specified metrics over epochs, comparing the training and validation data. The available plots include accuracy, loss, balanced accuracy, precision, and recall.Both functions use the matplotlib and seaborn libraries for plotting the graphs.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [Preprocessing_utils.py]({file})     | The code defines the `output_selection_prepro` function, which preprocesses data by imputing missing values and transforming the target variable. The function takes a dataframe `df` and a target variable `target` as inputs. It first checks if the preprocessed data file already exists, and if so, loads the data from the file. It then checks for missing values and if none are found, proceeds with defining numerical and categorical columns, dropping the target variable, and returning the preprocessed dataframe, target variable, and column lists. If the file does not exist or has missing values, the function runs the imputation process. It defines the target variable and lists of categorical and numerical columns, drops rows with missing values in the target column, and performs imputation on the remaining data using the `IterativeImputer` and `SimpleImputer` from scikit-learn. It transforms the target variable based on predefined thresholds, drops the target columns from the dataframe, saves the imputed data to a CSV file, and returns the preprocessed dataframe, transformed target variable, and updated column lists. |

</details>

---

##  Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- ℹ️ Python 3.11.4`

`- ℹ️ Anaconda(Optional)`

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

