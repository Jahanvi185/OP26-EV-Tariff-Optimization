# EV Dynamic Tariff Optimization using Agentic AI

## Overview

This project implements an Agentic AI-based dynamic pricing system for EV charging stations. The system predicts charging demand, adjusts tariffs based on charger utilization, and continuously evaluates pricing decisions to improve revenue, utilization, and customer experience.

## Features

* Demand prediction using Machine Learning
* Dynamic tariff optimization based on utilization
* Revenue and pricing efficiency analysis
* Charger utilization improvement
* Monitoring and feedback-based learning agent
* Data visualization and KPI reporting

## Datasets Used

* **ACN-Data (Caltech EV Charging Network)** – Revenue and pricing analysis
* **UrbanEV / ST-EVCDP (Shenzhen EV Charging Dataset)** – Demand prediction and utilization optimization

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* OpenPyXL

## Results

| Metric               | Improvement |
| -------------------- | ----------- |
| Revenue Gain         | +13.81%     |
| Charger Utilization  | +6.74%      |
| Off-Peak Utilization | +19.09%     |
| Wait-Time Reduction  | 13.93%      |
| Pricing Efficiency   | +13.81%     |
| Demand Prediction R² | 0.9603      |

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python ev_tariff_optimization.py
```

## Verify Dependencies

```bash
python -c "import pandas, numpy, matplotlib, sklearn, openpyxl; print('All OK')"
```

Expected Output:

```text
All OK
```

## Project Structure

text
EV_Tariff_Optimisation/
│
├── data/
├── outputs/
├── visuals/
├── ev_tariff_optimization.py
├── requirements.txt
└── README.md



## Author

Jahanvi Gautam (24112057)
Open Project 2026 – Agentic AI Dynamic Tariff Optimization
