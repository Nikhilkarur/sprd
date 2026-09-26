# Walmart Demand Forecasting (M5), a 4-Week Build

This project forecasts daily unit sales for Walmart items across stores. It uses global LightGBM with probabilistic (P50/P90) forecasts and reconciles them across the item → store → state hierarchy. The P90 forecasts feed a safety-stock layer, and the model is served through a FastAPI + Docker endpoint.

**Dataset:** [M5 Forecasting – Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy): about 30k item-store daily sales series, plus calendar, price and event data.

**Rule #1:** Every model must beat the seasonal-naive and moving-average baselines on the same backtest.

---

## Plan

### Week 1: Data, baselines and features
- Melt the wide sales table into long format (`item, store, date, sales`), then join the calendar and price data.
- Start with **one state (CA)** for speed. Downcast dtypes (`float32`, `int16`).
- **Backtesting:** 3 rolling windows × 28 days, time-based splits only.
- **Baselines:** seasonal naive (same day last week) and a moving average.
- **Features** (use polars or vectorized pandas `groupby`, never Python loops over series):
  - Lags (7, 14, 28) and rolling mean/std, shifted by the forecast horizon so no future data leaks in
  - Price: price relative to the item's average, and price-change flags
  - Calendar: day of week, month, events, SNAP days
  - Intermittency: days since last sale, and share of zero-sales days

### Week 2: Modeling and evaluation
- One **global LightGBM** model across all series, with **Tweedie loss** to handle the many zeros.
- Compare **direct vs. recursive** multi-step forecasting and write up the trade-off.
- Evaluate with **WRMSSE** (the official M5 metric), plus **MAE by product category**.
- Scale from CA to all states once the pipeline is stable.

### Week 3: Uncertainty, hierarchy and business impact
- **Quantile models** for P50 and P90.
- **Hierarchical reconciliation** (bottom-up vs. MinT) with `hierarchicalforecast`, so that item, store and state forecasts add up.
- **Business layer:** turn P90 into safety stock, then simulate stockout vs. overstock cost against the baseline.

### Week 4: Productionize and write up
- Refactor the notebooks into `src/` modules, and track experiments with **MLflow**.
- **FastAPI** endpoint: `GET /forecast?item_id=...&store_id=...` returns P50/P90 forecasts.
- Containerize with **Docker**.
- Finalize this README with results vs. baseline and limitations.

---

## Planned structure
```
ml walmart/
├── data/               # raw M5 CSVs (not committed)
├── notebooks/          # EDA and experiments
├── src/
│   ├── data.py         # load, melt, join, downcast
│   ├── features.py     # lags, rolling, price, calendar, intermittency
│   ├── backtest.py     # rolling-window CV and baselines
│   ├── train.py        # LightGBM (Tweedie + quantile), MLflow logging
│   ├── evaluate.py     # WRMSSE, MAE by category
│   ├── reconcile.py    # bottom-up / MinT
│   ├── inventory.py    # safety stock and cost simulation
│   └── api.py          # FastAPI app
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech stack
Python · polars/pandas · LightGBM · hierarchicalforecast · MLflow · FastAPI · Docker

## Results
| Model | WRMSSE | Δ vs. seasonal naive |
|---|---|---|
| Seasonal naive | TBD | — |
| Moving average | TBD | TBD |
| LightGBM (Tweedie) | TBD | TBD |
| LightGBM + MinT | TBD | TBD |

**Inventory simulation:** stockouts reduced by **Y%** with P90 safety stock (TBD).

## Limitations
_To be filled in: e.g., no external data (weather, promos), CA-first scope, and assumptions in the cost simulation._

## Resume bullet (target)
> Built a hierarchical demand forecasting system on Walmart M5 (30k series) using global LightGBM with Tweedie loss and quantile forecasts; improved WRMSSE by X% over seasonal-naive and reduced simulated stockouts by Y% via P90-based safety stock; deployed with FastAPI and Docker, tracked with MLflow.
