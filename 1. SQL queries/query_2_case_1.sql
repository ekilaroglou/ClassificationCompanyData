-- Create index for CompanyDataset
CREATE INDEX 
    idx_company_employee_industry 
ON 
    CompanyDataset(CompanyName, "current employee estimate", industry) 
WHERE 
    "current employee estimate" < 100;

-- Create index for CompanyClassification
CREATE INDEX 
    idx_company_homepage 
ON 
    CompanyClassification(CompanyName, homepage_text) 
WHERE 
    homepage_text IS NULL;


-- Identify the technology industries
WITH technology_industries AS (
    SELECT 'accounting' AS industry UNION ALL
    SELECT 'animation' UNION ALL
    SELECT 'automotive' UNION ALL
    SELECT 'aviation & aerospace' UNION ALL
    SELECT 'banking' UNION ALL
    SELECT 'biotechnology' UNION ALL
    SELECT 'computer & network security' UNION ALL
    SELECT 'computer games' UNION ALL
    SELECT 'computer hardware' UNION ALL
    SELECT 'computer networking' UNION ALL
    SELECT 'computer software' UNION ALL
    SELECT 'consumer electronics' UNION ALL
    SELECT 'defense & space' UNION ALL
    SELECT 'e-learning' UNION ALL
    SELECT 'electrical/electronic manufacturing' UNION ALL
    SELECT 'financial services' UNION ALL
    SELECT 'human resources' UNION ALL
    SELECT 'industrial automation' UNION ALL
    SELECT 'information technology and services' UNION ALL
    SELECT 'internet' UNION ALL
    SELECT 'logistics and supply chain' UNION ALL
    SELECT 'machinery' UNION ALL
    SELECT 'management consulting' UNION ALL
    SELECT 'marketing and advertising' UNION ALL
    SELECT 'mechanical or industrial engineering' UNION ALL
    SELECT 'medical devices' UNION ALL
    SELECT 'nanotechnology' UNION ALL
    SELECT 'online media' UNION ALL
    SELECT 'program development' UNION ALL
    SELECT 'public safety' UNION ALL
    SELECT 'security and investigations' UNION ALL
    SELECT 'semiconductors' UNION ALL
    SELECT 'telecommunications' UNION ALL
    SELECT 'wireless' UNION ALL
    SELECT 'writing and editing'
)
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
    AND cd."current employee estimate" < 100 -- Filter for companies that have fewer than 100 employees
    AND cd.industry IN (SELECT industry FROM technology_industries); -- Filter for 'Technology' like industry
