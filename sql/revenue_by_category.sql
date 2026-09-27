SELECT p.category, 
COUNT(*) AS transaction_count
SUM(t.quantity) AS units_sold
ROUND(SUM(t.net_amount), 2) AS revenue
FROM raw.transactions as t
join raw.product_services as p
on p.product_id = t.product_id
where t.transaction_status = 'Completed'
group by p.category
order by revenue desc