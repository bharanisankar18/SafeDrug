import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, trim

# Initialize Spark with increased memory and timeouts
spark = SparkSession.builder \
    .appName("DrugSafety-Clean") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# Updated paths matching your terminal's D:\SafeDose\data\src structure
input_path = "../raw/*q1*.json" 
output_path = "../processed/clean_reports.parquet"

print(f"Reading raw data from {input_path}...")
raw_df = spark.read.json(input_path)

# OpenFDA data is wrapped in a 'results' array
if "results" in raw_df.columns:
    df = raw_df.select(explode(col("results")).alias("report")).select("report.*")
else:
    df = raw_df

# Remove duplicate reports
df = df.dropDuplicates(["safetyreportid"])

# Extract Patient data
df = df.withColumn("patient", col("patient"))

# Split drugs into rows
df_drugs = df.withColumn("drug", explode(col("patient.drug")))

# Keep suspect drugs (1 = Primary Suspect in FDA data)
df_drugs = df_drugs.filter(col("drug.drugcharacterization") == "1")

# Standardize drug names
df_drugs = df_drugs.withColumn("drug_name", lower(trim(col("drug.medicinalproduct"))))

# Split reactions into rows
df_reactions = df_drugs.withColumn("reaction", explode(col("patient.reaction")))

# Standardize reaction names
df_reactions = df_reactions.withColumn("reaction_name", lower(trim(col("reaction.reactionmeddrapt"))))

# Select only the required fields
final_df = df_reactions.select(
    col("safetyreportid"),
    col("receivedate"),
    col("seriousness"),
    col("patient.patientsex").alias("sex"),
    col("patient.patientonsetage").alias("age"),
    col("drug_name"),
    col("reaction_name")
).dropna(subset=["safetyreportid", "drug_name", "reaction_name"])

# Calculate and print row counts for validation
row_count = final_df.count()
print(f"Total rows after cleaning: {row_count}")

# Save as Parquet
print(f"Saving clean data to {output_path}...")
final_df.write.mode("overwrite").parquet(output_path)

print(f"Files created: clean_reports.parquet in {output_path}")
spark.stop()