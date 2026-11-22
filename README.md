# AAVAIL Revenue Forecasting - AI Enterprise Workflow Capstone

## Project Overview
This repository contains the end-to-end AI workflow for forecasting revenue for **AAVAIL** using historical transactional JSON records (`cs-train` and `cs-production`), covering data ingestion, exploratory data analysis (EDA), regression modeling, Flask API serving, automated unit testing, drift/novelty monitoring, and Docker containerization.

## Data Ingestion & Automation (`ingestion.py`)
- Automated data loading and aggregation from daily JSON files into a clean time-series dataset per country and across all countries combined.
- Handles missing values, date parsing, and feature summarization across days.

## Exploratory Data Analysis (EDA) & Baseline Comparison
- Daily time-series analysis of `revenue`, `purchases`, and `total_views` shows strong seasonal peaks around **November–December**.
- Compared multiple regression models against a simple baseline model using RMSE/MAE metrics, selecting the best-performing model for revenue forecasting.

## Flask API & Performance Monitoring (`app.py`, `monitoring.py`)
- Provides prediction endpoints for:
  - A specific country and target date (`/predict`)
  - All countries combined (`/predict_all`)
- Includes input validation, logging, and data drift/novelty detection to monitor model performance over time.

## Unit Tests & Isolation (`tests.py`)
- Dedicated unit tests covering:
  - API endpoints and input validation
  - Model training and prediction outputs
  - Logging functionality
- Isolated from production models and production log files so tests can be executed cleanly in a single test runner without polluting production data.

## Docker Deployment (`Dockerfile`)
- Containerizes the Flask application and dependencies for reproducible deployment.
