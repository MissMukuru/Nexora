SELECT DATE_TRUNC('month', transaction_ts):: DATE AS MONTH, --groups eacamh transaction into its seperate month
COUNT (*) AS transaction_count,
ROUND(SUM(net_amount), 2) AS revenue
FROM raw.transactions
WHERE transaction_status = 'Completed'
GROUP BY DATE_TRUNC('month', transaction_ts) --creates one group per month
ORDER BY month