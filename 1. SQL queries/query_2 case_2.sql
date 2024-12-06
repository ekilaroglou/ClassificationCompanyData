-- Create index for CompanyDataset
CREATE INDEX 
    idx_company_employee 
ON 
    CompanyDataset(CompanyName, "current employee estimate") 
WHERE 
    "current employee estimate" < 100;

-- Create index for CompanyClassification
CREATE INDEX 
    idx_company_homepage_category 
ON 
    CompanyClassification(CompanyName, homepage_text, Category) 
WHERE 
    homepage_text IS NULL 
    AND Category = 'Information Technology';

-- Create the query
SELECT 
    cc.CompanyName -- Return the company name
FROM 
    CompanyClassification cc -- Inner join of CompanyClassification
JOIN
    CompanyDataset cd -- and CompanyDataset
ON 
    cc.CompanyName = cd.CompanyName -- CompanyName is the common column between the two tables
WHERE
    cc.homepage_text IS NULL -- Filter to have an effective homepage
    AND cc.Category = 'Information Technology' -- Filter for 'Technology' like industry
    AND cd."current employee estimate" < 100; -- Filter for companies that have fewer than 100 employees
