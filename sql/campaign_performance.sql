SELECT
    campaign_id,
    campaign_name,
    channel,
    spend,
    leads,
    qualified_leads,
    conversions,
    revenue,
    ROUND(roi::numeric, 2) AS roi
FROM raw.marketing_campaigns
ORDER BY revenue DESC
LIMIT 20;