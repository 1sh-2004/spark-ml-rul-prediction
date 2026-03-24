from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier

def train_models(df):
    # Use 10 key sensors instead of just 3 for better accuracy [cite: 42]
    feature_cols = [f"s{i}" for i in range(1, 11)]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    label_indexer = StringIndexer(inputCol="label", outputCol="label_index").fit(df)
    df = label_indexer.transform(df)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Increase trees to 10 for better research results [cite: 63]
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=10)
    gbt = GBTClassifier(featuresCol="features", labelCol="label_index", maxIter=5)

    print("🚀 Training High-Quality Models...")
    return rf.fit(train), gbt.fit(train), test