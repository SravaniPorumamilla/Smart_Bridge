# Rising Waters — ML Flood Risk Prediction

A complete, runnable implementation of the "Rising Waters" project: a Flask web app that
predicts flood risk from weather readings (annual rainfall, seasonal rainfall, cloud
visibility, temperature, humidity) using an XGBoost model chosen after comparing
Decision Tree, Random Forest, KNN, and XGBoost classifiers.

```
rising_waters/
├── data/
│   ├── generate_dataset.py      # builds the synthetic flood dataset (see note below)
│   └── flood_dataset.csv        # generated dataset (already included)
├── model/
│   ├── train_model.py           # EDA + preprocessing + model comparison + save
│   ├── floods.save              # trained XGBoost model (already included)
│   ├── transform.save           # fitted StandardScaler (already included)
│   ├── model_comparison.txt     # accuracy/report for all 4 models
│   └── eda_*.png                # distribution / box / heat-map plots
├── app/
│   ├── app.py                   # Flask application
│   ├── templates/               # home.html, index.html, chance.html, no_chance.html
│   └── static/                  # main.css, main.js
├── requirements.txt
└── README.md
```

The trained model and scaler are already included in `model/`, and the Flask app is
ready to run immediately — steps 1–3 below get you a working environment, step 4 runs
the app. Steps 5–6 are only needed if you want to regenerate the dataset or retrain
the model yourself.

> **A note on the dataset.** The original project spec references a Kaggle dataset
> (`kaggle.com/arbethi/rainfall-dataset`). Kaggle requires an authenticated,
> browser-based download, so this build ships `data/generate_dataset.py`, which
> creates a statistically realistic synthetic dataset with the same columns
> (annual rainfall, seasonal rainfall, cloud visibility, temperature, humidity →
> flood class) and a genuine, noisy relationship between them. If you have a Kaggle
> account, download the real dataset and drop it in as `data/flood_dataset.csv` with
> the same column names, then re-run `train_model.py` — nothing else needs to change.

---

## 1. Update the system and install Python build tools (Ubuntu 20.04)

Ubuntu 20.04 ships Python 3.8 by default. That still works with the pinned
`requirements.txt`, but current ML libraries are increasingly dropping 3.8 support,
so this guide also shows how to add Python 3.10 via the `deadsnakes` PPA — use
whichever track you prefer.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-pip python3-venv python3-dev git curl
```

### Option A — use the default Python 3.8 (simplest)

```bash
python3 --version      # should print Python 3.8.x
```

Skip to step 2.

### Option B — install Python 3.10 (recommended for smoother installs)

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 --version
```

If you use Option B, replace `python3` with `python3.10` in every command below.

---

## 2. Get the project onto your machine

Unzip the delivered archive (or clone your own copy of it):

```bash
unzip rising_waters.zip -d ~/rising_waters
cd ~/rising_waters
```

---

## 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # your prompt should now show (venv)

pip install --upgrade pip
pip install -r requirements.txt
```

This installs: NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Flask, and
Joblib — everything the project spec calls for.

---

## 4. Run the web application

The trained model (`model/floods.save`) and scaler (`model/transform.save`) are
already included, so you can run the app immediately:

```bash
cd app
python3 app.py
```

You should see:

```
 * Running on http://127.0.0.1:5000
```

Open that URL in a browser. Flow:

1. **Home** (`/`) — landing page.
2. **Predict** (`/Predict`) — enter annual rainfall, seasonal rainfall, cloud
   visibility, temperature, and humidity.
3. Submitting the form (`POST /predict`) scales your inputs with the saved
   `StandardScaler` and runs them through the saved XGBoost model, then redirects to:
   - **Flood risk detected** (`/chance`), or
   - **No flood risk detected** (`/no_chance`)

Stop the server with `Ctrl+C`.

To make it reachable from other devices on your network, run instead:

```bash
python3 -c "from app import app; app.run(host='0.0.0.0', port=5000)"
```

then visit `http://<your-machine-ip>:5000` from another device (make sure port 5000
is allowed through `ufw` if you have it enabled: `sudo ufw allow 5000/tcp`).

---

## 5. (Optional) Regenerate the dataset

```bash
cd data
python3 generate_dataset.py
```

This overwrites `data/flood_dataset.csv` with a freshly sampled synthetic dataset
(1,500 rows, ~1.5% missing values sprinkled in, a handful of injected outliers — so
the preprocessing steps in `train_model.py` have real work to do).

---

## 6. (Optional) Re-run the full ML pipeline

This performs Epics 2–4 from the project spec: EDA, preprocessing (missing values,
IQR outlier capping, train/test split, feature scaling), training and comparing four
classifiers, and saving the best one.

```bash
cd model
python3 train_model.py
```

Console output includes `df.head()`, `df.info()`, `df.describe()`, missing-value
counts, and per-model accuracy/confusion-matrix/classification-report. It also writes:

- `eda_distribution.png` — univariate distribution plots
- `eda_boxplots.png` — box plots (outlier view)
- `eda_heatmap.png` — multivariate correlation heat map
- `model_comparison.txt` — full text log of all four models
- `floods.save` — the selected model (XGBoost, unless another model clearly wins)
- `transform.save` — the fitted `StandardScaler`

The Flask app in `app/app.py` loads these two `.save` files at startup, so retraining
takes effect the next time you start (or restart) the app — no code changes needed.

---

## Troubleshooting

- **`ModuleNotFoundError`** — make sure the virtual environment is activated
  (`source venv/bin/activate`) before running any `python3` command.
- **`xgboost` fails to build** — it ships prebuilt wheels for Linux x86_64, so a
  plain `pip install` should not need a compiler; if it still fails, re-run
  `sudo apt install -y build-essential` and retry.
- **Port 5000 already in use** — another process (or macOS AirPlay Receiver, if
  you're testing over SSH from a Mac) is bound to it; run Flask on another port:
  `python3 -c "from app import app; app.run(port=5001)"`.
- **Model/scaler not found** — `app.py` expects `model/floods.save` and
  `model/transform.save` to exist relative to the `app/` folder; run
  `python3 model/train_model.py` from the project root if they're missing.
