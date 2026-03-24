from pyspark.sql import functions as F

def process_data(spark, file_path):
    columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
    df = spark.read.option("sep", " ").option("inferSchema", True).csv(file_path)
    df = df.select([df.columns[i] for i in range(26)]).toDF(*columns).dropna()

    # Proper Distributed Labeling (Objective #3)
    max_cycle = df.groupBy("engine_id").agg(F.max("cycle").alias("max_cycle"))
    df = df.join(max_cycle, on="engine_id")
    df = df.withColumn("RUL", F.col("max_cycle") - F.col("cycle"))

    # Map RUL to Research Status
    df = df.withColumn("label",
        F.when(F.col("RUL") > 100, "Healthy")
        .when((F.col("RUL") <= 100) & (F.col("RUL") > 30), "Warning")
        .otherwise("Critical"))

    return df.repartition(3).cache()