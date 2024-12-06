-- Create the index
CREATE INDEX 
    idx_industry_year_employee 
ON 
    CompanyDataset(industry, "year founded", "current employee estimate") 
WHERE 
    "year founded" > 2000 AND "current employee estimate" > 10;

-- Create the query
SELECT 
    industry -- Return the industry
FROM
    CompanyDataset -- Dataset of interest
WHERE
    -- Companies founded after 2000 that have more than 10 employees
    "year founded" > 2000 AND "current employee estimate" > 10 
GROUP BY
    industry
ORDER BY
    -- Sort by average number of employyes in descending order
    AVG("current employee estimate") DESC
LIMIT 10; -- Take the top 10 industries
