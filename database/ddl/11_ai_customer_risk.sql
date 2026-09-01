CREATE TABLE customer_ai_risk (
    customer_id         NUMBER PRIMARY KEY,
    risk_cluster        NUMBER,
    risk_level          VARCHAR2(20),
    total_outstanding   NUMBER(15,2),
    overdue_invoices    NUMBER,
    avg_days_overdue    NUMBER(10,2),
    updated_at          TIMESTAMP DEFAULT SYSTIMESTAMP
);

COMMIT;