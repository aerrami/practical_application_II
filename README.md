# What Drives the Price of a Car?

Practical Application 11.1: applying the CRISP-DM framework to identify which attributes drive used-car prices, with inventory recommendations for used car dealers.

**Notebook:** [prompt_II.ipynb](prompt_II.ipynb)

## Business Problem
A used car dealership wants to fine-tune its inventory by understanding what consumers value in a used car. I reframed this as a supervised regression problem: predict listing price from vehicle attributes, and use interpretable models to quantify how much each attribute raises or lowers price.

## Data
The source is a Kaggle dataset of 426,880 used-car listings. Cleaning removed $0 placeholder prices (about 33K listings), prices outside $1,000–$100,000, vehicles older than 1990, odometer readings over 400,000 miles, duplicates, and uninformative columns (ID, VIN, county, size, region, model). That left **254,278 listings** for modeling.

## Approach
- **Target:** log(price), so coefficients read as percentage effects on price
- **Features:** age, mileage, manufacturer, condition, cylinders, fuel, title status, transmission, drive, body type, paint color, state
- **Models:** mean baseline, linear regression, Ridge, Lasso, and Ridge with polynomial age/mileage terms, tuned with 5-fold cross-validated grid search
- **Metric:** RMSE on log price for tuning (it measures errors in relative terms, which suits prices ranging from $1K to $100K); MAE in dollars on the held-out test set for business interpretation

## Results
| Model | Test MAE ($) | Test R² (log price) |
|---|---|---|
| Baseline (mean) | 10,641 | 0.00 |
| Linear Regression | 4,671 | 0.75 |
| Ridge | 4,670 | 0.75 |
| Lasso | 4,670 | 0.75 |
| **Polynomial Ridge (degree 3)** | **4,569** | **0.77** |

The best model cuts the average pricing error by 57% compared with quoting the average price.

## Key Findings
1. **Age and mileage matter most.** Each year of age lowers the price about **7%**, and each 10,000 miles about **4%**.
2. **Trucks and 4WD earn a premium.** Pickups and trucks list about **40%** above comparable sedans, and 4WD about **30%** above front-wheel drive.
3. **Diesel and larger engines add value.** Diesel vehicles list far above comparable gas vehicles, and 8-cylinder engines about **49%** above 4-cylinder. This largely reflects the heavy-duty truck market.
4. **"Fair" condition and branded titles are costly.** "Fair" lists about **41% below** "good," while "excellent" adds only about 10%. Salvage titles cost about **28%** and rebuilt titles about **21%**.
5. **Brand matters, but less.** Lexus, Audi, Mercedes-Benz, and Toyota hold value best. Chrysler, Kia, Hyundai, and Nissan list lowest. Paint color barely matters.

## Recommendations for Dealers
- Prioritize late-model, lower-mileage inventory.
- Expand truck, 4WD, and diesel inventory where local demand supports it.
- Recondition cars to at least "good" condition, where the payoff is largest.
- Buy salvage and rebuilt titles only at a discount deeper than their 20–30% price penalty.
- Use the model as a pricing sanity check (typical error about $4,600), not a replacement for an appraiser.

## Next Steps
- Standardize and add the `model` field, since trim and model likely explain much of the remaining variation.
- Try tree-based models (random forest, gradient boosting) for better accuracy.
- Build region-specific analyses and retrain on recent data.

## Repository Structure
```
├── README.md
├── prompt_II.ipynb
├── data/
│   └── vehicles.csv
└── images/
    ├── crisp.png
    └── kurt.jpeg
```
