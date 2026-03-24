import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load Data
columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv("CMAPSSData/train_FD001.txt", sep=" ", header=None).iloc[:, :26]
df.columns = columns

# Preprocessing: RUL and Labeling [cite: 55-59]
max_cycle = df.groupby('engine_id')['cycle'].transform('max')
df['RUL'] = max_cycle - df['cycle']
df['label'] = df['RUL'].apply(lambda x: 2 if x <= 30 else (1 if x <= 100 else 0))

# Feature Engineering: Rolling Mean
features = [f"s{i}" for i in range(1, 22)]
df[features] = df.groupby('engine_id')[features].transform(lambda x: x.rolling(window=5).mean()).fillna(0)

# Split and Train [cite: 62-64]
X = df[features]
y = df['label']
X_train, X_test = X[:16000], X[16000:]
y_train, y_test = y[:16000], y[16000:]

start = time.time()
rf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
gbt = GradientBoostingClassifier(n_iter_no_change=2).fit(X_train, y_train)
latency = time.time() - start

print(f"✅ Sequential Latency: {latency:.2f}s")
print(f"✅ RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")