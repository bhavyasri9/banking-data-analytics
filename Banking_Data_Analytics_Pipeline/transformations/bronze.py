import dlt
from pyspark.sql.types import *
from pyspark.sql.functions import col, when, rand, upper, lower

# ---------------------------
# SCHEMAS (DEFINE EXPLICITLY)
# ---------------------------

account_schema = StructType([
    StructField("account_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("branch_id", StringType(), True),
    StructField("account_type", StringType(), True),
    StructField("balance", DoubleType(), True),
    StructField("created_date", StringType(), True)
])

branch_schema = StructType([
    StructField("branch_id", StringType(), True),
    StructField("branch_name", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True)
])

customer_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("city", StringType(), True)
])

loan_schema = StructType([
    StructField("loan_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("loan_amount", DoubleType(), True),
    StructField("loan_type", StringType(), True),
    StructField("loan_status", StringType(), True)
])

txn_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("transaction_date", StringType(), True)
])

# ---------------------------
# REUSABLE FUNCTIONS
# ---------------------------

def add_bronze_issues(df):
    # Add random NULLs
    for column in df.columns:
        df = df.withColumn(
            column,
            when(rand() > 0.9, None).otherwise(col(column))
        )

    # Add duplicates
    df = df.union(df.sample(fraction=0.05))

    # Replace NULL with empty string for string columns
    for column, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(
                column,
                when(col(column).isNull(), "").otherwise(col(column))
            )

    return df


def add_inconsistency(df):
    for column, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(
                column,
                when(rand() > 0.7, upper(col(column)))
                .when(rand() > 0.5, lower(col(column)))
                .otherwise(col(column))
            )
    return df

# ---------------------------
# BRONZE TABLES
# ---------------------------

@dlt.table(name="bronze_accounts")
def bronze_accounts():
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(account_schema) \
        .option("pathGlobFilter", "Bank_Account*.csv") \
        .load("/Volumes/workspace/default/bankingpro/")
    
    df = add_bronze_issues(df)
    df = add_inconsistency(df)
    
    return df


@dlt.table(name="bronze_branch")
def bronze_branch():
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(branch_schema) \
        .option("pathGlobFilter", "Bank_Branch*.csv") \
        .load("/Volumes/workspace/default/bankingpro/")
    
    df = add_bronze_issues(df)
    
    return df


@dlt.table(name="bronze_customer")
def bronze_customer():
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(customer_schema) \
        .option("pathGlobFilter", "Bank_Customer*.csv") \
        .load("/Volumes/workspace/default/bankingpro/")
    
    df = add_bronze_issues(df)
    df = add_inconsistency(df)
    
    return df


@dlt.table(name="bronze_loan")
def bronze_loan():
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(loan_schema) \
        .option("pathGlobFilter", "Bank_Loan*.csv") \
        .load("/Volumes/workspace/default/bankingpro/")
    
    df = add_bronze_issues(df)
    
    return df


@dlt.table(name="bronze_transactions")
def bronze_transactions():
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(txn_schema) \
        .option("pathGlobFilter", "Bank_Transacation*.csv") \
        .load("/Volumes/workspace/default/bankingpro/")
    
    df = add_bronze_issues(df)
    
    return df