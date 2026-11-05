import sys
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, StringIndexer, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# STEP 1: CREATE SPARK SESSION

spark = SparkSession.builder \
    .appName("Loan Approval Prediction") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# STEP 2: LOAD DATASET

df = spark.read.csv(
    "loan_approval_dataset.csv",
    header=True,
    inferSchema=True
)

# Remove extra spaces from column names and values
for c in df.columns:
    df = df.withColumnRenamed(c, c.strip())

for c, t in df.dtypes:
    if t == "string":
        df = df.withColumn(c, F.trim(F.col(c)))

# Remove unnecessary ID column
df = df.drop("loan_id")

print("Total Records:", df.count())
print("Dataset:")
df.show(5)


# STEP 3: HANDLE MISSING VALUES

numeric_cols = [
    "no_of_dependents", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value",
    "bank_asset_value"
]

categorical_cols = ["education", "self_employed"]

# Divide data into training and testing
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Replace missing numerical values with median
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imputed" for c in numeric_cols],
    strategy="median"
)


# STEP 4: STANDARDIZE CREDIT SCORE

credit_assembler = VectorAssembler(
    inputCols=["cibil_score_imputed"],
    outputCol="credit_vector"
)

scaler = StandardScaler(
    inputCol="credit_vector",
    outputCol="credit_scaled",
    withMean=True,
    withStd=True
)


# STEP 5: CONVERT TEXT INTO NUMBERS

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

# Rejected = 0, Approved = 1
label_indexer = StringIndexer(
    inputCol="loan_status",
    outputCol="label",
    stringOrderType="alphabetDesc"
)


# STEP 6: PREPARE FEATURES

features = (
    [c + "_imputed" for c in numeric_cols if c != "cibil_score"]
    + ["credit_scaled"]
    + [c + "_index" for c in categorical_cols]
)

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)


# STEP 7: TRAIN DECISION TREE MODEL

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(stages=[
    imputer,
    credit_assembler,
    scaler,
    *indexers,
    label_indexer,
    assembler,
    dt
])

model = pipeline.fit(train)

# Make predictions
predictions = model.transform(test)

print("Loan Approval Predictions:")
predictions.select("loan_status", "prediction").show(10)


# STEP 8: EVALUATE MODEL

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# STEP 9: CONFUSION MATRIX

# 0 = Rejected, 1 = Approved

result = predictions.agg(
    F.sum(F.when(
        (F.col("label") == 0) & (F.col("prediction") == 0), 1
    ).otherwise(0)).alias("TN"),

    F.sum(F.when(
        (F.col("label") == 0) & (F.col("prediction") == 1), 1
    ).otherwise(0)).alias("FP"),

    F.sum(F.when(
        (F.col("label") == 1) & (F.col("prediction") == 0), 1
    ).otherwise(0)).alias("FN"),

    F.sum(F.when(
        (F.col("label") == 1) & (F.col("prediction") == 1), 1
    ).otherwise(0)).alias("TP")
).first()

tn, fp, fn, tp = [result[x] or 0 for x in ["TN", "FP", "FN", "TP"]]

print("\nCONFUSION MATRIX")
print("                 Predicted")
print("              Rejected  Approved")
print(f"Rejected       {tn:5}     {fp:5}")
print(f"Approved       {fn:5}     {tp:5}")

# Display confusion matrix as a graph
plt.figure(figsize=(5, 4))
plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")

for i, row in enumerate([[tn, fp], [fn, tp]]):
    for j, value in enumerate(row):
        plt.text(j, i, value, ha="center", va="center")

plt.xticks([0, 1], ["Rejected", "Approved"])
plt.yticks([0, 1], ["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Loan Approval Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# STEP 10: APPROVAL RATE BY INCOME GROUP

income_df = df.withColumn(
    "income_group",
    F.when(F.col("income_annum") < 2500000, "Low Income")
     .when(F.col("income_annum") < 5000000, "Middle Income")
     .otherwise("High Income")
)

approval = income_df.groupBy("income_group").agg(
    F.avg(
        (F.col("loan_status") == "Approved").cast("double")
    ).alias("approval_rate")
)

approval = approval.toPandas()

approval["approval_rate"] *= 100

# Plot the results
plt.figure(figsize=(7, 5))
plt.bar(approval["income_group"], approval["approval_rate"])

plt.xlabel("Income Group")
plt.ylabel("Approval Rate (%)")
plt.title("Loan Approval Rate by Income Group")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig("approval_by_income.png")
plt.close()


# STEP 11: FINISH

print("\nProject Completed Successfully!")
print("Generated: confusion_matrix.png")
print("Generated: approval_by_income.png")

spark.stop()
