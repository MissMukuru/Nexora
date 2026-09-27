SELECT
    c.customer_id,
    c.company_name,
    c.region,
    COUNT(*) AS transaction_count,
    ROUND(SUM(t.net_amount), 2) AS total_revenue
FROM raw.customers AS c
JOIN raw.transactions AS t
    ON t.customer_id = c.customer_id
WHERE t.transaction_status = 'Completed'
GROUP BY c.customer_id, c.company_name, c.region
ORDER BY total_revenue DESC
LIMIT 10;