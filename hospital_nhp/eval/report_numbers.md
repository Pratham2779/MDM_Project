# Hospital Bed Capacity Forecasting - Evaluation Summary

## 1. Model Configuration
*   **Selected Model**: nhp_calendar
*   **Capacity Baseline**: 150 beds
*   **Test Period**: 2025-01-01 to 2025-12-31

## 2. Arrival Forecasting Accuracy
*   **1-Day Ahead (H1)**: MAE = 11.82 patients/day | sMAPE = 47.71%
*   **7-Day Ahead (H7)**: MAE = 11.82 patients/day | sMAPE = 47.72%
*   **Significance vs SMA Baseline**: p-value = 0.0000 (Paired T-test)

## 3. Capacity & Occupancy Simulation
*   **Average Estimated Utilisation**: 99.9%
*   **Average Predicted Breach Risk (H1)**: 49.1% days expected over capacity.
*   **Simulator Mechanism**: Monte Carlo (100 paths) applying fitted Length-of-Stay distributions to NHP Poisson arrival samples.
