import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------
# SILVER CUSTOMERS
# ---------------------------

@dlt.table(name="silver_customers")
def silver_customers():

    df = dlt.read("bronze_customer")

    df = df \
        .withColumnRenamed("customer_id", "CUSTOMER_ID") \
        .withColumnRenamed("first_name", "First_Name") \
        .withColumnRenamed("last_name", "Last_Name") \
        .withColumnRenamed("phone", "Phone_Number") \
        .withColumnRenamed("city", "City")

    clean_phone = F.expr("try_cast(regexp_replace(Phone_Number, '[^0-9]', '') as BIGINT)")

    median_list = df \
        .withColumn("Phone_Number", clean_phone) \
        .filter(F.col("Phone_Number").isNotNull()) \
        .approxQuantile("Phone_Number", [0.5], 0.01)

    median_phone = median_list[0] if median_list else 0

    return (
        df
        .filter(F.col("CUSTOMER_ID").isNotNull())
        .withColumn("CUSTOMER_ID", F.upper(F.trim("CUSTOMER_ID")))
        .withColumn("First_Name", F.coalesce(F.initcap(F.trim("First_Name")), F.lit("Unknown")))
        .withColumn("Last_Name", F.coalesce(F.initcap(F.trim("Last_Name")), F.lit("Unknown")))
        .withColumn("City", F.coalesce(F.initcap(F.trim("City")), F.lit("Unknown")))
        .withColumn("Phone_Number", clean_phone)
        .withColumn("Phone_Number", F.coalesce(F.col("Phone_Number"), F.lit(median_phone)))
        .dropDuplicates(["CUSTOMER_ID"])
        .withColumn("_silver_loaded_at", F.current_timestamp())
    )

# ---------------------------
# SILVER BRANCHES
# ---------------------------

@dlt.table(name="silver_branches")
def silver_branches():

    df = dlt.read("bronze_branch")

    df = df \
        .withColumnRenamed("branch_id", "BRANCH_ID") \
        .withColumnRenamed("branch_name", "BRANCH_NAME") \
        .withColumnRenamed("state", "BRANCH_STATE")

    return (
        df
        .filter(F.col("BRANCH_ID").isNotNull())
        .withColumn("BRANCH_ID", F.upper(F.trim("BRANCH_ID")))
        .withColumn("BRANCH_NAME", F.coalesce(F.trim("BRANCH_NAME"), F.lit("Unknown Branch")))
        .withColumn("BRANCH_STATE", F.coalesce(F.initcap(F.trim("BRANCH_STATE")), F.lit("Unknown")))
        .dropDuplicates(["BRANCH_ID"])
        .withColumn("_silver_loaded_at", F.current_timestamp())
    )

# ---------------------------
# SILVER ACCOUNTS
# ---------------------------

@dlt.table(name="silver_accounts")
def silver_accounts():

    df = dlt.read("bronze_accounts")

    df = df \
        .withColumnRenamed("account_id", "ACCOUNT_ID") \
        .withColumnRenamed("customer_id", "CUSTOMER_ID") \
        .withColumnRenamed("branch_id", "BRANCH_ID") \
        .withColumnRenamed("balance", "OPENING_BALANCE") \
        .withColumnRenamed("created_date", "ACCOUNT_OPEN_DATE")

    clean_balance = F.expr("try_cast(regexp_replace(OPENING_BALANCE, '[^0-9.]', '') as DOUBLE)")

    median_list = df \
        .withColumn("OPENING_BALANCE", clean_balance) \
        .filter(F.col("OPENING_BALANCE").isNotNull()) \
        .approxQuantile("OPENING_BALANCE", [0.5], 0.01)

    median_balance = median_list[0] if median_list else 0.0

    return (
        df
        .filter(F.col("ACCOUNT_ID").isNotNull())
        .withColumn("ACCOUNT_ID", F.upper(F.trim("ACCOUNT_ID")))
        .withColumn("CUSTOMER_ID", F.upper(F.trim("CUSTOMER_ID")))
        .withColumn("BRANCH_ID", F.upper(F.trim("BRANCH_ID")))
        .withColumn("ACCOUNT_TYPE",
            F.coalesce(F.initcap(F.lower(F.trim("ACCOUNT_TYPE"))), F.lit("Unknown"))
        )
        .withColumn("ACCOUNT_OPEN_DATE",
            F.coalesce(
                F.expr("try_to_date(ACCOUNT_OPEN_DATE, 'yyyy-MM-dd')"),
                F.lit("1900-01-01").cast("date")
            )
        )
        .withColumn("OPENING_BALANCE", clean_balance)
        .withColumn("OPENING_BALANCE",
            F.coalesce(F.col("OPENING_BALANCE"), F.lit(median_balance))
        )
        .dropDuplicates(["ACCOUNT_ID"])
        .withColumn("_silver_loaded_at", F.current_timestamp())
    )

# ---------------------------
# SILVER LOANS
# ---------------------------

@dlt.table(name="silver_loans")
def silver_loans():

    df = dlt.read("bronze_loan")

    df = df \
        .withColumnRenamed("loan_id", "LOAN_ID") \
        .withColumnRenamed("customer_id", "CUSTOMER_ID")

    median_list = df \
        .withColumn("LOAN_AMOUNT", F.expr("try_cast(loan_amount as DOUBLE)")) \
        .filter(F.col("LOAN_AMOUNT").isNotNull()) \
        .approxQuantile("LOAN_AMOUNT", [0.5], 0.01)

    median_loan = median_list[0] if median_list else 0.0

    return (
        df
        .filter(F.col("LOAN_ID").isNotNull())
        .withColumn("LOAN_ID", F.upper(F.trim("LOAN_ID")))
        .withColumn("CUSTOMER_ID", F.upper(F.trim("CUSTOMER_ID")))
        .withColumn("LOAN_AMOUNT",
            F.coalesce(
                F.expr("try_cast(loan_amount as DOUBLE)"),
                F.lit(median_loan)
            )
        )
        .dropDuplicates(["LOAN_ID"])
        .withColumn("_silver_loaded_at", F.current_timestamp())
    )

# ---------------------------
# SILVER TRANSACTIONS
# ---------------------------

@dlt.table(name="silver_transactions")
def silver_transactions():

    df = dlt.read("bronze_transactions")

    df = df \
        .withColumnRenamed("transaction_id", "TRANSACTION_ID") \
        .withColumnRenamed("account_id", "ACCOUNT_ID") \
        .withColumnRenamed("amount", "TRANSACTION_AMOUNT") \
        .withColumnRenamed("transaction_date", "TRANSACTION_DATE") \
        .withColumnRenamed("transaction_type", "TRANSACTION_TYPE")

    median_list = df \
        .withColumn("TRANSACTION_AMOUNT", F.expr("try_cast(TRANSACTION_AMOUNT as DOUBLE)")) \
        .filter(F.col("TRANSACTION_AMOUNT").isNotNull()) \
        .approxQuantile("TRANSACTION_AMOUNT", [0.5], 0.01)

    median_txn = median_list[0] if median_list else 0.0

    window_spec = Window.partitionBy("TRANSACTION_ID").orderBy(F.col("TRANSACTION_DATE").desc())

    return (
        df
        .filter(F.col("TRANSACTION_ID").isNotNull())
        .withColumn("TRANSACTION_ID", F.upper(F.trim("TRANSACTION_ID")))
        .withColumn("ACCOUNT_ID", F.upper(F.trim("ACCOUNT_ID")))
        .withColumn("TRANSACTION_DATE",
            F.coalesce(
                F.expr("try_to_date(TRANSACTION_DATE, 'yyyy-MM-dd')"),
                F.lit("1900-01-01").cast("date")
            )
        )
        .withColumn("TRANSACTION_TYPE",
            F.coalesce(F.initcap(F.trim("TRANSACTION_TYPE")), F.lit("Unknown"))
        )
        .withColumn("TRANSACTION_AMOUNT",
            F.coalesce(
                F.expr("try_cast(TRANSACTION_AMOUNT as DOUBLE)"),
                F.lit(median_txn)
            )
        )
        .withColumn("row_num", F.row_number().over(window_spec))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
        .withColumn("_silver_loaded_at", F.current_timestamp())
    )