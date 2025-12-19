from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re
from pyspark.sql.functions import udf, length
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("TwitterSentimentSpark") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Spark Session initialized successfully!")

names = ['id', 'company', 'label', 'sentence']

print("\nLoading training data...")
train_df = (
    spark.read
    .option("header", "false")
    .option("inferSchema", "true")
    .csv("/app/data/twitter_training.csv")
    .toDF(*names)
    .select("label", "sentence")
)

print("Loading validation data...")
test_df = (
    spark.read
    .option("header", "false")
    .option("inferSchema", "true")
    .csv("/app/data/twitter_validation.csv")
    .toDF(*names)
    .select("label", "sentence")
)

print(f"Training samples: {train_df.count()}")
print(f"Validation samples: {test_df.count()}")

def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_text_udf = udf(clean_text, StringType())

print("\nCleaning text data...")
train_df = train_df.withColumn("clean_text", clean_text_udf(train_df.sentence))
test_df = test_df.withColumn("clean_text", clean_text_udf(test_df.sentence))

train_df = train_df.filter(length("clean_text") > 0)
test_df = test_df.filter(length("clean_text") > 0)

print("Text cleaning completed!")

print("\nBuilding ML Pipeline...")

tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=20000)
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="label_idx")
model = LogisticRegression(featuresCol="features", labelCol="label_idx", maxIter=20, regParam=0.01)

pipeline = Pipeline(stages=[tokenizer, stopwords_remover, tf, idf, label_indexer, model])

print("\nSplitting training data...")
train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=42)

print("\n" + "="*50)
print("TRAINING MODEL...")
print("="*50)
spark_model = pipeline.fit(train_df)
print("Model training completed!")

print("\nEvaluating on training set...")
train_preds = spark_model.transform(train_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label_idx",
    predictionCol="prediction",
    metricName="accuracy"
)

train_acc = evaluator.evaluate(train_preds)
print(f"Train Accuracy: {train_acc:.4f}")

print("\nEvaluating on validation set...")
val_preds = spark_model.transform(val_df)
val_acc = evaluator.evaluate(val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")

print("\nValidation Confusion Matrix:")
val_preds.groupBy("label_idx", "prediction").count().orderBy("label_idx").show()

print("\n" + "="*50)
print("TESTING ON NEW EXAMPLES")
print("="*50)

new_data = [
    ("The app keeps crashing after the update",),
    ("Totally irrelevant discussion here",),
    ("Great customer support from Microsoft",),
    ("This product is terrible and broken",),
    ("Amazing experience with the service",)
]

new_df = spark.createDataFrame(new_data, ["sentence"])
new_df = new_df.withColumn("clean_text", clean_text_udf(new_df.sentence))
new_df = new_df.filter(length("clean_text") > 0)

preds = spark_model.transform(new_df)

label_mapping = spark_model.stages[4].labels

def decode_label(idx):
    return label_mapping[int(idx)]

decode_udf = udf(decode_label, StringType())
preds = preds.withColumn("predicted_label", decode_udf(preds.prediction))

print("\nPredictions on new data:")
preds.select("sentence", "predicted_label", "probability").show(truncate=False)

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)
model_path = "/app/output/sentiment_model"
spark_model.write().overwrite().save(model_path)
print(f"Model saved to: {model_path}")

print("\nSaving validation predictions...")
val_preds.select("label", "sentence", "prediction", "label_idx") \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv("/app/output/validation_predictions")

print("\n" + "="*50)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

spark.stop()
print("\nSpark session stopped. Goodbye!")