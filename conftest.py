import os

# When using jax.experimental.enable_x64 in unit test, we want to keep the
# default dtype with 32 bits, aligning it with Keras's default.
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

try:
    # When using torch and tensorflow, torch needs to be imported first,
    # otherwise it will segfault upon import. This should force the torch
    # import to happen first for all tests.
    import torch  # noqa: F401
except ImportError:
    pass

import pytest  # noqa: E402
from keras.src.backend import backend  # noqa: E402
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, when

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )

def pytest_collection_modifyitems(config, items):
    openvino_skipped_tests = []
    if backend() == "openvino":
        with open(
            "keras/src/backend/openvino/excluded_concrete_tests.txt", "r"
        ) as file:
            openvino_skipped_tests = file.readlines()
            openvino_skipped_tests = [
                line.strip() for line in openvino_skipped_tests if line.strip()
            ]

    requires_trainable_backend = pytest.mark.skipif(
        backend() == "numpy" or backend() == "openvino",
        reason="Trainer not implemented for NumPy and OpenVINO backend.",
    )
    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        for skipped_test in openvino_skipped_tests:
            if skipped_test in item.nodeid:
                item.add_marker(
                    skip_if_backend(
                        "openvino",
                        "Not supported operation by openvino backend",
                    )
                )

def skip_if_backend(given_backend, reason):
    return pytest.mark.skipif(backend() == given_backend, reason=reason)

# Initialize Spark Session
def create_spark_session(app_name="KerasSparkTuning"):
    """
    Creates and returns a Spark session for distributed data processing.

    Args:
        app_name (str): Name of the Spark application.

    Returns:
        SparkSession: Configured Spark session.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    return spark

# Preprocess Data Using Spark
def preprocess_data_with_spark(spark, data_path):
    """
    Uses Spark to preprocess large datasets for deep learning training.

    Args:
        spark (SparkSession): Active Spark session.
        data_path (str): Path to the dataset file (CSV, Parquet, etc.).

    Returns:
        DataFrame: Cleaned and processed Spark DataFrame.
    """
    df = spark.read.format("csv").option("header", "true").load(data_path)

    # Convert numeric columns and drop null values
    processed_df = df.select(
        col("feature1").cast("float"),
        col("feature2").cast("float"),
        col("label").cast("int")
    ).dropna()

    print(f"Processed {processed_df.count()} rows using Spark.")
    return processed_df

# Normalize Data Using Standardization
def normalize_data(df):
    """
    Normalizes numeric features using Z-score normalization.

    Args:
        df (DataFrame): Spark DataFrame with numerical columns.

    Returns:
        DataFrame: Normalized Spark DataFrame.
    """
    stats = df.select(
        mean(col("feature1")).alias("mean_feature1"),
        stddev(col("feature1")).alias("std_feature1"),
        mean(col("feature2")).alias("mean_feature2"),
        stddev(col("feature2")).alias("std_feature2")
    ).collect()[0]

    normalized_df = df.withColumn("feature1", (col("feature1") - stats["mean_feature1"]) / stats["std_feature1"]) \
                      .withColumn("feature2", (col("feature2") - stats["mean_feature2"]) / stats["std_feature2"])

    return normalized_df

# Handle Missing Values
def handle_missing_values(df):
    """
    Handles missing values by filling them with the mean of the respective column.

    Args:
        df (DataFrame): Spark DataFrame with missing values.

    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    mean_values = df.select(
        mean(col("feature1")).alias("mean_feature1"),
        mean(col("feature2")).alias("mean_feature2")
    ).collect()[0]

    filled_df = df.withColumn("feature1", when(col("feature1").isNull(), mean_values["mean_feature1"]).otherwise(col("feature1"))) \
                  .withColumn("feature2", when(col("feature2").isNull(), mean_values["mean_feature2"]).otherwise(col("feature2")))

    return filled_df

# Split Data into Training and Testing Sets
def split_data(df, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets.

    Args:
        df (DataFrame): Spark DataFrame to be split.
        train_ratio (float): Proportion of the dataset to use for training.

    Returns:
        Tuple[DataFrame, DataFrame]: Training and testing DataFrames.
    """
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=42)
    return train_df, test_df

# Save Processed Data
def save_dataframe(df, output_path, format="parquet"):
    """
    Saves a Spark DataFrame to the specified output path.

    Args:
        df (DataFrame): The Spark DataFrame to be saved.
        output_path (str): Path to save the processed dataset.
        format (str): File format (default is "parquet").

    Returns:
        None
    """
    df.write.format(format).mode("overwrite").save(output_path)
    print(f"Data saved to {output_path} in {format} format.")

