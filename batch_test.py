import time
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("CMAPSS_Research_Validation").getOrCreate()

# 1. Load Model and Data
rf_model = RandomForestClassificationModel.load("/app/rf_model_spark")
cols = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
test_df = spark.read.option("sep", " ").option("inferSchema", True).csv("/app/CMAPSSData/test_FD001.txt")
test_df = test_df.select([test_df.columns[i] for i in range(26)]).toDF(*cols).dropna()

# 2. Ground Truth Integration (RUL_FD001.txt) [cite: 42, 65]
# This file contains the true remaining life for each engine
truth_df = spark.read.csv("/app/CMAPSSData/RUL_FD001.txt", inferSchema=True).toDF("true_rul")
truth_df = truth_df.withColumn("engine_id", F.monotonically_increasing_id() + 1)

# 3. Simulate "Streaming" Batch Processing
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=[f"s{i}" for i in range(1, 22)], outputCol="features")
test_data = assembler.transform(test_df)

print("🚀 Starting Batch Simulation...")
results = rf_model.transform(test_data)

# 4. Map Predictions to Status Strings
results = results.withColumn("Status",
    F.when(F.col("prediction") == 0.0, "Healthy")
    .when(F.col("prediction") == 1.0, "Warning")
    .otherwise("Critical"))

# Show the results clearly
print("\n" + "="*50)
print("📊 BATCH SIMULATION: ENGINE HEALTH STATUS")
print("="*50)
results.select("engine_id", "cycle", "Status").show(20)

# 5. Final Accuracy Calculation (Objective #4) [cite: 41, 68]
# Note: Since test_FD001 only has the 'last' cycle as ground truth, we filter for max cycle
final_cycles = results.groupBy("engine_id").agg(F.max("cycle").alias("cycle"))
final_preds = results.join(final_cycles, on=["engine_id", "cycle"])
comparison = final_preds.join(truth_df, on="engine_id")

# Research Logic: If true_rul > 100 Healthy, 30-100 Warning, <30 Critical
comparison = comparison.withColumn("True_Status",
    F.when(F.col("true_rul") > 100, "Healthy")
    .when((F.col("true_rul") <= 100) & (F.col("true_rul") > 30), "Warning")
    .otherwise("Critical"))

accuracy = comparison.filter(F.col("Status") == F.col("True_Status")).count() / 100.0
print(f"✅ FINAL RESEARCH TEST ACCURACY: {accuracy:.4f}")
print("="*50)