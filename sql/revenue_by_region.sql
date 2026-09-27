-- Active: 1790523346434@@127.0.0.1@5432@nexora@raw
SELECT 
c.region, 
COUNT(*) AS transactions_count, --counts the matvhing transation rows and names the output column
ROUND(SUM(t.net_amount), 2) AS revenue   --adds the transaction net amounts and rounds the resuly to 2 d.pp
FROM raw.transactions AS t
JOIN raw.customers AS C
    ON c.customer_id = t.customer_id --matches each transaction to its customer using the customer id
WHERE t.transaction_status = 'Completed'
GROUP BY c.region
ORDER BY revenue DESC;