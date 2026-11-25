# Distributed Predictive Maintenance with Apache Spark

A scalable predictive maintenance pipeline built on **Apache Spark (PySpark / MLlib)** that analyzes the **NASA C-MAPSS** turbofan engine degradation dataset. Engine sensor streams are labeled with Remaining Useful Life (RUL), classified into health states, and used to benchmark **distributed Spark MLlib training against a sequential scikit-learn baseline**.

## Highlights

- Distributed ETL pipeline in PySpark for multivariate time-series sensor data
- RUL computed per engine with a distributed `groupBy` + `join`, then mapped to health states
- Random Forest and Gradient-Boosted Trees trained with Spark MLlib on a Spark standalone cluster
- Sequential scikit-learn baseline for latency and scalability comparison
- Batch inference on the held-out test set, checked against the ground-truth RUL file
- Fully containerized with Docker Compose (1 master, 1 worker, 1 driver app)

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (Commercial Modular Aero-Propulsion System Simulation). This project uses the **FD001** subset:

| File | Purpose |
|------|---------|
| `train_FD001.txt` | Run-to-failure trajectories used for training |
| `test_FD001.txt` | Truncated trajectories used for batch inference |
| `RUL_FD001.txt` | True RUL of each test engine at its last recorded cycle |

Each row has 26 columns: `engine_id`, `cycle`, 3 operational settings (`setting_1..3`), and 21 sensor readings (`s1..s21`).

The dataset is **not included** in this repo (it is git-ignored). Download it and place the files in a `CMAPSSData/` folder at the project root.

## Pipeline

```
 CMAPSSData/*.txt
        │
        ▼
 data_processor.py   Parse columns → compute RUL per engine → assign health label
        │             (repartition + cache)
        ▼
 model_trainer.py    VectorAssembler (s1–s10) → StringIndexer → 80/20 split
        │             → RandomForestClassifier + GBTClassifier
        ▼
 evaluator.py        Accuracy for both models + top feature importance
        │
        ▼
 main.py             Orchestrates the run, times training, saves models,
                     then runs batch inference on test_FD001.txt
```

### Health labels

RUL is calculated as `max_cycle − cycle` for each engine, then bucketed:

| RUL | Status |
|-----|--------|
| > 100 | Healthy |
| 31 – 100 | Warning |
| ≤ 30 | Critical |

## Project structure

```
.
├── main.py                 # End-to-end distributed pipeline (entry point)
├── data_processor.py       # Distributed ETL and RUL labeling
├── model_trainer.py        # Spark MLlib RF + GBT training
├── evaluator.py            # Accuracy and feature importance
├── batch_test.py           # Batch inference with a saved model vs. RUL_FD001 ground truth
├── sequential_baseline.py  # scikit-learn single-node baseline for benchmarking
├── Dockerfile              # Python 3.10 + JDK image for the driver app
├── docker-compose.yml      # Spark master, worker, and ml-app services
└── requirements.txt
```

## Getting started

### Prerequisites

- Docker and Docker Compose
- The C-MAPSS dataset (see above)

### 1. Add the data

```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

### 2. Update the volume mounts

`docker-compose.yml` currently mounts a machine-specific path (`/Users/vanshkeserwani/bdat`) to `/app`. Change it to your own project directory, or to `.`:

```yaml
volumes:
  - .:/app
```

This must be updated for both the `spark-worker` and `ml-app` services.

### 3. Run the distributed pipeline

```bash
docker compose up --build
```

This starts the Spark master, a worker (2 cores, 1.5 GB), and the `ml-app` driver, which runs `main.py`. It will:

1. Load and label the training data
2. Train the Random Forest and GBT models on the cluster
3. Print accuracy, feature importance, and training latency
4. Save the models to `rf_model_spark/` and `gbt_model_spark/`
5. Run batch predictions on `test_FD001.txt`

Useful UIs while it runs:

- Spark master: http://localhost:8080
- Spark application UI: http://localhost:4040

### 4. Batch-test a saved model (optional)

After the models are saved, change the `command` of `ml-app` in `docker-compose.yml` to `python batch_test.py` (the line is already there, commented out) and rerun. It loads `rf_model_spark`, predicts a health status for every test cycle, and compares the last-cycle prediction per engine to the true status derived from `RUL_FD001.txt`.

### 5. Run the sequential baseline

For the comparison run, execute the baseline on a single node:

```bash
pip install -r requirements.txt
python sequential_baseline.py
```

It trains scikit-learn Random Forest and Gradient Boosting classifiers on the same FD001 data and prints the training latency and Random Forest accuracy.

## Benchmark: distributed vs. sequential

Both implementations log training latency so the two approaches can be compared directly.

| | Spark MLlib (`main.py`) | Sequential (`sequential_baseline.py`) |
|---|---|---|
| Engine | Spark standalone cluster | Single process, scikit-learn |
| Features | Sensors `s1`–`s10` | Sensors `s1`–`s21`, 5-cycle rolling mean |
| Split | Random 80/20 (seed 42) | First 16,000 rows train, remainder test |
| Models | RF (10 trees), GBT (5 iterations) | RF (10 trees), GBT (early stopping) |
| Reported | Training latency, RF/GBT accuracy, top feature importance | Training latency, RF accuracy |

<!-- SPECULATED-RESULTS-START: delete everything between these two markers once you have real numbers -->
**Estimated results (speculated, not measured):**

| Approach | Training latency | RF accuracy |
|---|---|---|
| Spark MLlib | ~40 s | ~0.80 |
| Sequential | ~15 s | ~0.78 |

On a dataset as small as FD001, the single-node baseline is expected to train faster because cluster scheduling overhead dominates. The distributed approach should pull ahead as data volume and worker count grow.


Note that the two scripts use different feature sets and splits, so accuracy is not strictly like-for-like. At FD001 scale, cluster overhead can dominate; the distributed approach pays off as data volume and cluster size grow.

## Tech stack

Python 3.10 · Apache Spark 3.5.0 (PySpark, MLlib) · scikit-learn · pandas · Docker · Docker Compose

## Notes and limitations

- The task is framed as **health-state classification** (Healthy / Warning / Critical) derived from RUL, not direct RUL regression.
- Input paths are read through Spark's reader, so the data location can be pointed at a distributed file system such as HDFS by changing the path strings in `main.py`, `batch_test.py`, and `data_processor.py`; no HDFS service is bundled in the Compose file.
- Only the FD001 subset (single operating condition, single fault mode) is used.
- Trained models (`*.pkl`, Spark model folders) and the dataset are not tracked in git.

## Acknowledgements

Dataset: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.
