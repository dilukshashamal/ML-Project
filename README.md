# Student Maths Mark Predictor

This project predicts a student's math score based on various features such as gender, ethnicity, parental level of education, lunch type, and test preparation course completion. The application is built with a focus on creating clean pipelines for data ingestion, transformation, training, and prediction, while ensuring continuous integration and delivery (CI/CD) through GitHub Actions.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Screenshots](#screenshots)

## Project Overview

This project is built using a **data-driven approach** and follows best practices in terms of **file structure**, ensuring modularity and scalability. The project supports:
- **Data ingestion** and **transformation** pipelines.
- A **prediction pipeline** for forecasting student exam performance.
- A **training pipeline** to fine-tune models.
- Continuous integration and deployment using **GitHub Actions**.
- Unit testing using **pytest**.

## Features

- **Prediction and Training Pipelines**: Predicts math scores based on the provided student data and trains models.
- **Data Ingestion and Transformation**: Automates the process of importing and cleaning data.
- **CI/CD Pipeline**: Automatically builds, tests, and deploys using GitHub Actions.
- **Unit Testing**: Comprehensive unit testing implemented with `pytest`.
  
## Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/your-username/student-performance-indicator.git](https://github.com/dilukshashamal/ML-Project.git)
    ```
2. Navigate to the project directory:
    ```bash
    cd ML Project
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Start the application:
    ```bash
    python app.py
    ```

## Screenshots
![Homepage](./assets/homepage.png)
