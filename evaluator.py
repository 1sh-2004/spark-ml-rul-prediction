from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def run_evaluation(rf_model, gbt_model, test_data):
    evaluator = MulticlassClassificationEvaluator(labelCol="label_index", metricName="accuracy")

    print("\n" + "=" * 40)
    print("📊 FINAL MODEL RESULTS (DISTRIBUTED)")
    print("=" * 40)
    print(f"Random Forest Accuracy: {evaluator.evaluate(rf_model.transform(test_data)):.4f}")
    print(f"GBT Accuracy:           {evaluator.evaluate(gbt_model.transform(test_data)):.4f}")

    # Objective #5: Feature Importance [cite: 42]
    importances = gbt_model.featureImportances.toArray()
    print(f"Top Sensor Influence:   {max(importances):.4f}")
    print("=" * 40)