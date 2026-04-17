import dlt
from pyspark.sql import functions as F

# =========================================================
# DIMENSION TABLES
# =========================================================

# ---------------------------
# DIM CUSTOMERS
# ---------------------------
@dlt.table(name="dim_customers")
def dim_customers():
    return dlt.read("silver_customers").select(
        "CUSTOMER_ID",
        "First_Name",
        "Last_Name",
        "City"
    )

# ---------------------------
# DIM BRANCHES
# ---------------------------
@dlt.table(name="dim_branches")
def dim_branches():
    return dlt.read("silver_branches").select(
        "BRANCH_ID",
        "BRANCH_NAME",
        "BRANCH_STATE"
    )

# ---------------------------
# DIM ACCOUNTS
# ---------------------------
@dlt.table(name="dim_accounts")
def dim_accounts():
    return dlt.read("silver_accounts").select(
        "ACCOUNT_ID",
        "CUSTOMER_ID",
        "BRANCH_ID",
        "ACCOUNT_TYPE",
        "OPENING_BALANCE",
        "ACCOUNT_OPEN_DATE"
    )

# =========================================================
# FACT TABLE
# =========================================================

# ---------------------------
# FACT TRANSACTIONS
# ---------------------------
@dlt.table(name="fact_transactions")
def fact_transactions():

    txn = dlt.read("silver_transactions")
    acc = dlt.read("silver_accounts")

    return txn.join(acc, "ACCOUNT_ID", "inner") \
        .select(
            txn["TRANSACTION_ID"],
            txn["ACCOUNT_ID"],
            acc["CUSTOMER_ID"],
            acc["BRANCH_ID"],
            txn["TRANSACTION_AMOUNT"],
            txn["TRANSACTION_TYPE"],
            txn["TRANSACTION_DATE"]
        )

# =========================================================
# KPI TABLES (BUSINESS INSIGHTS)
# =========================================================

# ---------------------------
# KPI 1: OVERALL BUSINESS
# ---------------------------
@dlt.table(name="kpi_total_business")
def kpi_total_business():

    df = dlt.read("fact_transactions")

    return df.agg(
        F.count("TRANSACTION_ID").alias("total_transactions"),
        F.sum("TRANSACTION_AMOUNT").alias("total_revenue"),
        F.avg("TRANSACTION_AMOUNT").alias("avg_transaction_value")
    )

# ---------------------------
# KPI 2: TRANSACTION TYPE
# ---------------------------
@dlt.table(name="kpi_transaction_type")
def kpi_transaction_type():

    df = dlt.read("fact_transactions")

    return df.groupBy("TRANSACTION_TYPE") \
        .agg(
            F.count("*").alias("txn_count"),
            F.sum("TRANSACTION_AMOUNT").alias("total_amount")
        )

# ---------------------------
# KPI 3: TOP CUSTOMERS
# ---------------------------
@dlt.table(name="kpi_top_customers")
def kpi_top_customers():

    df = dlt.read("fact_transactions")

    return df.groupBy("CUSTOMER_ID") \
        .agg(
            F.sum("TRANSACTION_AMOUNT").alias("total_spent"),
            F.count("*").alias("txn_count")
        ) \
        .orderBy(F.desc("total_spent"))

# ---------------------------
# KPI 4: BRANCH PERFORMANCE
# ---------------------------
@dlt.table(name="kpi_branch_performance")
def kpi_branch_performance():

    df = dlt.read("fact_transactions")

    return df.groupBy("BRANCH_ID") \
        .agg(
            F.count("*").alias("total_transactions"),
            F.sum("TRANSACTION_AMOUNT").alias("branch_revenue")
        ) \
        .orderBy(F.desc("branch_revenue"))

# ---------------------------
# KPI 5: MONTHLY TREND
# ---------------------------
@dlt.table(name="kpi_monthly_trend")
def kpi_monthly_trend():

    df = dlt.read("fact_transactions")

    return df.withColumn("month", F.date_format("TRANSACTION_DATE", "yyyy-MM")) \
        .groupBy("month") \
        .agg(
            F.sum("TRANSACTION_AMOUNT").alias("monthly_revenue"),
            F.count("*").alias("txn_count")
        ) \
        .orderBy("month")

# ---------------------------
# KPI 6: ACCOUNT TYPE ANALYSIS
# ---------------------------
@dlt.table(name="kpi_account_type")
def kpi_account_type():

    txn = dlt.read("fact_transactions")
    acc = dlt.read("dim_accounts")

    return txn.join(acc, "ACCOUNT_ID") \
        .groupBy("ACCOUNT_TYPE") \
        .agg(
            F.sum("TRANSACTION_AMOUNT").alias("total_amount"),
            F.count("*").alias("txn_count")
        )

# ---------------------------
# KPI 7: TOP BRANCH + CUSTOMER COMBINED
# ---------------------------
@dlt.table(name="kpi_customer_branch")
def kpi_customer_branch():

    df = dlt.read("fact_transactions")

    return df.groupBy("CUSTOMER_ID", "BRANCH_ID") \
        .agg(
            F.sum("TRANSACTION_AMOUNT").alias("total_amount")
        ) \
        .orderBy(F.desc("total_amount"))

# ---------------------------
# KPI 8: HIGH VALUE TRANSACTIONS
# ---------------------------
@dlt.table(name="kpi_high_value_txn")
def kpi_high_value_txn():

    df = dlt.read("fact_transactions")

    threshold = df.approxQuantile("TRANSACTION_AMOUNT", [0.9], 0.01)
    high_value = threshold[0] if threshold else 0

    return df.filter(F.col("TRANSACTION_AMOUNT") >= high_value)