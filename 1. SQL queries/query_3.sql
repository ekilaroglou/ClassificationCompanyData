-- Create the index. We choose country as first parameter since we partition by country.
-- We choose "total employee estimate" as the second parameter since we compute the ranking based on this.
-- We choose CompanyName as third parameter since it's the least important parameter
CREATE INDEX 
    idx_company_country_total 
ON 
    CompanyDataset(country, "total employee estimate", CompanyName) 
WHERE 
    Country IS NOT NULL;

-- Create a Common Table Expression (CTE) to rank companies between each country
WITH RankedCompaniesByCountry AS (
    SELECT 
        CompanyName, -- We need the CompanyName
        country, -- We also need the country in the result
        -- Rank companies within each country based on the total employee estimate in descending order
        RANK() OVER (PARTITION BY country ORDER BY "total employee estimate" DESC) AS RankInCountry
    FROM 
        CompanyDataset
    WHERE 
        -- Only include records where the country field is not null
        country IS NOT NULL
)
-- Create the query
SELECT 
    CompanyName, -- Return the company name
    country,  -- Return the country
    RankInCountry -- Return the rank within the country
FROM 
    RankedCompaniesByCountry
WHERE 
    RankInCountry <= 5;  -- Filter to only include the top 5 companies per country

