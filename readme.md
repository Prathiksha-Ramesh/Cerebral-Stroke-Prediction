# Cerebral Stroke Prediction

## About the Project

A stroke, also known as a cerebrovascular accident (CVA), occurs when part of the brain loses its blood supply, causing the affected brain cells to stop functioning. This loss of blood supply can be ischemic, due to a lack of blood flow, or hemorrhagic, due to bleeding into brain tissue. Strokes are medical emergencies as they can lead to death or permanent disability. While there are opportunities to treat ischemic strokes, treatment needs to start within the first few hours after the signs of a stroke begin.

This project focuses on predicting cerebral strokes using a machine learning approach based on an imbalanced medical dataset. The dataset consists of 12 features, including the target column, which indicates whether a stroke occurred.

## Dataset

The cerebral stroke dataset contains the following features:

1. ID
2. Gender
3. Age
4. Hypertension
5. Heart Disease
6. Ever Married
7. Work Type
8. Residence Type
9. Average Glucose Level
10. BMI
11. Smoking Status
12. Stroke (target column)

The dataset is imbalanced, with significantly more non-stroke cases than stroke cases.

### Source

Liu, Tianyu; Fan, Wenhui; Wu, Cheng (2019), “Data for A hybrid machine learning approach to cerebral stroke prediction based on imbalanced medical-datasets”, Mendeley Data, V1, [doi: 10.17632/x8ygrw87jw.1](https://doi.org/10.17632/x8ygrw87jw.1)

## Project Structure

- `notebook.ipynb`: Jupyter Notebook containing the project code and analysis.
- `dataset.csv`: CSV file containing the dataset.
- `requirements.txt`: List of Python packages required for the project.
- `LICENSE`: License file for the project.
- `readme.md`: Documentation for the project

## Installation

To run the project, you need to have Python installed on your system. Follow these steps to set up the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Jupyter Notebook

1. **Open the project in VS Code**:
    - Open Visual Studio Code.
    - Open the project folder by navigating to `File > Open Folder` and selecting the project directory.

2. **Open the Jupyter Notebook**:
    - Install the Python and Jupyter extensions for VS Code if you haven't already.
    - Open `notebook.ipynb` in VS Code.

3. **Run the Notebook**:
    - Run the cells in the Jupyter Notebook to load the dataset, preprocess the data, train the model, and evaluate the results.

### Running the Script

1. **Load the dataset**:
    ```python
    import pandas as pd

    df = pd.read_csv('dataset.csv')
    ```

2. **Preprocess the data**:
    - Handle missing values
    - Encode categorical features
    - Normalize/scale numerical features

3. **Train the model**:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Split the dataset
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```

4. **Evaluate the model**:
    ```python
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

This project uses the dataset provided by Liu, Tianyu; Fan, Wenhui; Wu, Cheng (2019).
