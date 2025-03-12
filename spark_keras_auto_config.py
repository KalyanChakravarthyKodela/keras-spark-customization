from pyspark.sql import SparkSession
import tensorflow as tf
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import numpy as np

# Automatically configure Spark session for optimization
spark = (SparkSession.builder
         .appName("SparkKerasAutoConfig")
         .master("local[*]")
         .config("spark.executor.memory", "8g")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .getOrCreate())

# Function to pick the best algorithm based on Spark use cases
def pick_algorithm(use_case):
    algorithms = {
        "classification": {"hidden_units": [64, 32, 16], "activation": "relu", "output_activation": "sigmoid"},
        "regression": {"hidden_units": [128, 64, 32], "activation": "relu", "output_activation": "linear"},
        "anomaly_detection": {"hidden_units": [256, 128, 64], "activation": "tanh", "output_activation": "sigmoid"},
        "recommendation": {"hidden_units": [512, 256, 128], "activation": "relu", "output_activation": "softmax"}
    }
    return algorithms.get(use_case, algorithms["classification"])

# Define an optimized neural network model using a Deep Feedforward Neural Network
def create_model(input_dim=5, use_case="classification"):
    params = pick_algorithm(use_case)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    
    for units in params["hidden_units"]:
        model.add(tf.keras.layers.Dense(units, activation=params["activation"]))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(1, activation=params["output_activation"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy' if use_case == "classification" else 'mse', 
                  metrics=['accuracy' if use_case == "classification" else 'mae'])
    return model

# Create optimized model based on use case
model = create_model(use_case="classification")

# Convert model inference to UDF with optimized batch prediction
def predict_udf(*features):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data, batch_size=32, verbose=0)[0][0]
    return float(prediction)

predict = udf(predict_udf, FloatType())

# Example DataFrame usage
data = [(0.1, 0.2, 0.3, 0.4, 0.5), (0.5, 0.4, 0.3, 0.2, 0.1)]
df = spark.createDataFrame(data, ["f1", "f2", "f3", "f4", "f5"])
df = df.withColumn("prediction", predict("f1", "f2", "f3", "f4", "f5"))
df.show()

# Stop Spark session
spark.stop()
