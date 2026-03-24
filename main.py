import time
from pyspark.sql import SparkSession
import data_processor, model_trainer, evaluator

spark = SparkSession.builder \
    .appName("CMAPSS_Final_Distributed") \
    .master("spark://spark-master:7077") \
    .config("spark.driver.host", "ml-app") \
    .getOrCreate()

print("🚀 Starting Distributed Pipeline...")
overall_start = time.time()

df = data_processor.process_data(spark, "/app/CMAPSSData/train_FD001.txt")

print(f"📊 Materializing Windowed Data... Count: {df.count()}")

train_start = time.time()
rf, gbt, test = model_trainer.train_models(df)
train_latency = time.time() - train_start

evaluator.run_evaluation(rf, gbt, test)
print("💾 Saving Models to Disk...")
rf.save("/app/rf_model_spark")
gbt.save("/app/gbt_model_spark")

print("✅ Models saved in /Users/vanshkeserwani/bdat/rf_model_spark")
print("🏁 Pipeline Finished Successfully.")

print(f"\n⏱️ Training Latency: {train_latency:.2f} seconds")
print(f"🏁 Total Execution Time: {time.time() - overall_start:.2f} seconds")
# ==============================
# 🚀 NEW: RESEARCH BATCH TESTING
# ==============================
print("\n📂 Loading test_FD001.txt for Research Validation...")
# Load the actual test file [cite: 18, 45-46]
test_raw = spark.read.option("sep", " ").option("inferSchema", True).csv("/app/CMAPSSData/test_FD001.txt")

# Standardize columns to match the training set [cite: 48-50]
columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
test_df = test_raw.select([test_raw.columns[i] for i in range(26)]).toDF(*columns).dropna()

print("🔮 Running Batch Predictions on Test Data...")
# Use the trained RF model to predict on the new batch [cite: 62-64]
# Note: model_trainer.py must provide the assembler logic or use the raw sensors
from pyspark.ml.feature import VectorAssembler
feature_cols = [f"s{i}" for i in range(1, 22)] # Match your trainer's features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
test_data = assembler.transform(test_df)

test_results = rf.transform(test_data)

# Show a sample of predictions for Objective #4 [cite: 41, 70]
print("📋 BATCH INFERENCE SAMPLES:")
test_results.select("engine_id", "cycle", "prediction").show(10)