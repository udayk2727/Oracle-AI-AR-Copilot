import os
import oracledb
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Oracle Database Connection
# ---------------------------------------------------------
def get_connection():
    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER", "AR_ADMIN"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
    )

    return connection


# ---------------------------------------------------------
# Read Accounts Receivable data from Oracle
# ---------------------------------------------------------
def load_customer_data(connection):

    sql = """
    SELECT
        customer_id,

        SUM(NVL(outstanding_amount, 0))
            AS total_outstanding,

        SUM(
            CASE
                WHEN due_date < SYSDATE
                     AND NVL(outstanding_amount, 0) > 0
                THEN 1
                ELSE 0
            END
        ) AS overdue_invoices,

        NVL(
    AVG(
        CASE
            WHEN due_date < SYSDATE
                 AND NVL(outstanding_amount, 0) > 0
            THEN TRUNC(SYSDATE - due_date)
        END
    ),
    0
) AS avg_days_overdue

    FROM invoices

    GROUP BY customer_id
"""
    cursor = connection.cursor()

    cursor.execute(sql)

    rows = cursor.fetchall()

    columns = [
        "CUSTOMER_ID",
        "TOTAL_OUTSTANDING",
        "OVERDUE_INVOICES",
        "AVG_DAYS_OVERDUE"
    ]

    cursor.close()

    dataframe = pd.DataFrame(
        rows,
        columns=columns
    )

    return dataframe


# ---------------------------------------------------------
# AI / Machine Learning Risk Classification
# ---------------------------------------------------------
def calculate_ai_risk(dataframe):

    if dataframe.empty:
        print("No customer invoice data found.")
        return dataframe

    print("\nPreparing customer financial features...")

    features = dataframe[
        [
            "TOTAL_OUTSTANDING",
            "OVERDUE_INVOICES",
            "AVG_DAYS_OVERDUE"
        ]
    ].fillna(0)

    # Standardize financial data
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    number_of_customers = len(dataframe)

    number_of_clusters = min(
        3,
        number_of_customers
    )

    # If database only has one customer
    if number_of_clusters == 1:

        dataframe["RISK_CLUSTER"] = 0
        dataframe["RISK_LEVEL"] = "LOW"

        return dataframe

    # -----------------------------------------------------
    # K-Means Machine Learning Model
    # -----------------------------------------------------
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    dataframe["RISK_CLUSTER"] = (
        model.fit_predict(
            scaled_features
        )
    )

    # -----------------------------------------------------
    # Understand which cluster has greater payment risk
    # -----------------------------------------------------
    cluster_analysis = (
        dataframe
        .groupby("RISK_CLUSTER")
        [
            [
                "TOTAL_OUTSTANDING",
                "OVERDUE_INVOICES",
                "AVG_DAYS_OVERDUE"
            ]
        ]
        .mean()
    )

    # Internal score used to rank ML clusters
    cluster_analysis["RISK_SCORE"] = (
        cluster_analysis["TOTAL_OUTSTANDING"]
        +
        (
            cluster_analysis["OVERDUE_INVOICES"]
            * 100
        )
        +
        (
            cluster_analysis["AVG_DAYS_OVERDUE"]
            * 10
        )
    )

    ordered_clusters = (
        cluster_analysis["RISK_SCORE"]
        .sort_values()
        .index
        .tolist()
    )

    # -----------------------------------------------------
    # Convert ML clusters to business-friendly risk labels
    # -----------------------------------------------------
    if number_of_clusters == 3:

        risk_mapping = {
            ordered_clusters[0]: "LOW",
            ordered_clusters[1]: "MEDIUM",
            ordered_clusters[2]: "HIGH"
        }

    else:

        risk_mapping = {
            ordered_clusters[0]: "LOW",
            ordered_clusters[1]: "HIGH"
        }

    dataframe["RISK_LEVEL"] = (
        dataframe["RISK_CLUSTER"]
        .map(risk_mapping)
    )

    return dataframe


# ---------------------------------------------------------
# Save AI prediction back into Oracle
# ---------------------------------------------------------
def save_predictions(connection, dataframe):

    merge_sql = """
        MERGE INTO customer_ai_risk target

        USING (
            SELECT
                :customer_id AS customer_id
            FROM dual
        ) source

        ON (
            target.customer_id =
            source.customer_id
        )

        WHEN MATCHED THEN

            UPDATE SET

                target.risk_cluster =
                    :risk_cluster,

                target.risk_level =
                    :risk_level,

                target.total_outstanding =
                    :total_outstanding,

                target.overdue_invoices =
                    :overdue_invoices,

                target.avg_days_overdue =
                    :avg_days_overdue,

                target.updated_at =
                    SYSTIMESTAMP

        WHEN NOT MATCHED THEN

            INSERT (
                customer_id,
                risk_cluster,
                risk_level,
                total_outstanding,
                overdue_invoices,
                avg_days_overdue,
                updated_at
            )

            VALUES (
                :customer_id,
                :risk_cluster,
                :risk_level,
                :total_outstanding,
                :overdue_invoices,
                :avg_days_overdue,
                SYSTIMESTAMP
            )
    """

    cursor = connection.cursor()

    for _, row in dataframe.iterrows():

        cursor.execute(
            merge_sql,
            {
                "customer_id":
                    int(row["CUSTOMER_ID"]),

                "risk_cluster":
                    int(row["RISK_CLUSTER"]),

                "risk_level":
                    str(row["RISK_LEVEL"]),

                "total_outstanding":
                    float(
                        row["TOTAL_OUTSTANDING"]
                        or 0
                    ),

                "overdue_invoices":
                    int(
                        row["OVERDUE_INVOICES"]
                        or 0
                    ),

                "avg_days_overdue":
                    float(
                        row["AVG_DAYS_OVERDUE"]
                        or 0
                    )
            }
        )

    connection.commit()

    cursor.close()


# ---------------------------------------------------------
# Display high-risk customers
# ---------------------------------------------------------
def display_high_risk_customers(dataframe):

    high_risk = dataframe[
        dataframe["RISK_LEVEL"] == "HIGH"
    ]

    print("\n====================================")
    print("HIGH RISK CUSTOMERS")
    print("====================================")

    if high_risk.empty:

        print("No high-risk customers identified.")

        return

    for _, customer in high_risk.iterrows():

        print(
            "\nCustomer:",
            customer["CUSTOMER_ID"]
        )

        print(
            "Outstanding Amount: $",
            round(
                customer["TOTAL_OUTSTANDING"],
                2
            )
        )

        print(
            "Overdue Invoices:",
            customer["OVERDUE_INVOICES"]
        )

        print(
            "Average Days Overdue:",
            round(
                customer["AVG_DAYS_OVERDUE"],
                2
            )
        )

        print(
            "AI Risk Level:",
            customer["RISK_LEVEL"]
        )


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    connection = None

    try:

        print("\n====================================")
        print(" Oracle AI Customer Risk Agent")
        print("====================================")

        print(
            "\nConnecting to Oracle Database..."
        )

        connection = get_connection()

        print(
            "Oracle connection successful."
        )

        # Load customer AR information
        dataframe = load_customer_data(
            connection
        )

        print(
            f"\nCustomers found: {len(dataframe)}"
        )

        if dataframe.empty:
            return

        # Run AI model
        print(
            "\nRunning machine learning model..."
        )

        dataframe = calculate_ai_risk(
            dataframe
        )

        print(
            "\nAI Customer Risk Results"
        )

        print(
            "------------------------------------"
        )

        print(
            dataframe.to_string(
                index=False
            )
        )

        # Write AI results to Oracle
        print(
            "\nSaving AI predictions to Oracle..."
        )

        save_predictions(
            connection,
            dataframe
        )

        print(
            "AI predictions saved successfully."
        )

        display_high_risk_customers(
            dataframe
        )

        print(
            "\n===================================="
        )

        print(
            "AI risk analysis completed."
        )

        print(
            "===================================="
        )

    except oracledb.DatabaseError as error:

        print(
            "\nOracle Database Error:"
        )

        print(error)

    except Exception as error:

        print(
            "\nApplication Error:"
        )

        print(error)

    finally:

        if connection:

            connection.close()

            print(
                "\nOracle connection closed."
            )


if __name__ == "__main__":
    main()