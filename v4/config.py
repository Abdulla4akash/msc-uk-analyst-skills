"""
v4 configuration — frozen taxonomy and provenance.

This file is a FROZEN COPY of v3/base-model/config.py at tag v3-original-baseline,
with an explicit taxonomy version.  v4 does NOT import from v3 at runtime so that
future changes to v3 cannot silently affect v4.

Taxonomy version: v3-13cat-frozen
Source: v3/base-model/config.py  (git tag v3-original-baseline, commit d7ee030)
"""

TAXONOMY_VERSION = "v3-13cat-frozen"
TAXONOMY_SOURCE = "v3/base-model/config.py @ v3-original-baseline (d7ee030)"
# Any change to categories requires a new taxonomy_version.
# For publication we freeze this file before external test construction.

CATEGORIES = [
    "programming",
    "sql",
    "visualisation_bi",
    "reporting",
    "excel",
    "statistics",
    "machine_learning",
    "data_cleaning",
    "etl",
    "data_modelling",
    "cloud",
    "stakeholder_comm",
    "ethics_governance",
]

CATEGORY_LABELS = {
    "programming": "programming or scripting languages",
    "sql": "database querying with SQL",
    "visualisation_bi": "data visualisation and business intelligence tools",
    "reporting": "producing reports and management information",
    "excel": "spreadsheet software such as Excel",
    "statistics": "statistical analysis and forecasting",
    "machine_learning": "machine learning and predictive modelling",
    "data_cleaning": "data cleaning and data quality",
    "etl": "data engineering, ETL pipelines and data warehousing",
    "data_modelling": "data modelling and schema design",
    "cloud": "cloud computing platforms",
    "stakeholder_comm": "stakeholder communication and presenting findings",
    "ethics_governance": "data governance, privacy and GDPR compliance",
}

# Tier 2 lexicons — exact copy of v3 LEXICONS at freeze point.
# Do NOT edit after looking at held-out data.
LEXICONS = {
    "programming": [
        "python", "r programming", "scala", "java", "c#", "vba", "macros", "dax",
        "pyspark", "bash", "shell scripting", "powershell", "programming",
        "coding", "scripting", "programming languages",
    ],
    "sql": [
        "sql", "t-sql", "transact-sql", "pl/sql", "mysql", "postgresql", "postgres",
        "sql server", "oracle database", "mongodb", "nosql", "bigquery", "database",
        "databases", "relational database", "relational databases", "ms access",
        "microsoft access", "dynamodb", "elasticsearch", "hive", "cassandra", "db2",
    ],
    "visualisation_bi": [
        "power bi", "powerbi", "tableau", "looker", "looker studio", "qlik",
        "qlikview", "alteryx", "spotfire", "data visualisation", "data visualization",
        "visualisation", "visualization", "dashboard", "dashboards",
    ],
    "reporting": [
        "reporting", "reports", "report writing", "report production", "mi",
        "management information", "management reporting", "kpi reporting",
        "performance pack", "ssrs", "crystal reports", "sisense",
    ],
    "excel": [
        "excel", "microsoft excel", "ms excel", "google sheets", "spreadsheet",
        "spreadsheets", "pivot table", "pivot tables", "vlookup", "vlookups",
        "power query", "powerquery",
    ],
    "statistics": [
        "statistics", "statistical", "statistical analysis", "statistical modelling",
        "statistical modeling", "regression", "forecasting", "forecast", "forecasts",
        "time series", "hypothesis testing", "a/b testing", "a/b test",
        "econometrics", "econometric", "sas", "spss", "matlab", "minitab",
    ],
    "machine_learning": [
        "machine learning", "ml", "predictive modelling", "predictive modeling",
        "predictive analytics", "feature engineering", "deep learning", "nlp",
        "natural language processing", "llm", "tensorflow", "pytorch",
        "scikit-learn", "sklearn", "artificial intelligence", "model development",
    ],
    "data_cleaning": [
        "data cleaning", "data cleansing", "cleansing", "cleanse", "data preparation",
        "data prep", "data wrangling", "data quality", "data validation",
        "data integrity", "data profiling",
    ],
    "etl": [
        "etl", "elt", "data pipeline", "data pipelines", "pipelines", "ingestion",
        "data ingestion", "data warehouse", "data warehousing", "data lake",
        "ssis", "informatica", "airflow", "kafka", "spark", "snowflake", "redshift",
        "hadoop", "databricks", "big data", "data mart",
    ],
    "data_modelling": [
        "data model", "data models", "data modelling", "data modeling",
        "dimensional modelling", "dimensional modeling", "star schema", "kimball",
        "entity relationship", "entity-relationship", "erwin", "schema design",
    ],
    "cloud": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "cloud platform", "cloud platforms", "cloud environment", "databricks",
        "snowflake", "s3", "ec2", "sagemaker", "azure synapse", "azure data factory",
    ],
    "stakeholder_comm": [
        "stakeholder", "stakeholders", "stakeholder management",
        "stakeholder engagement", "presenting", "presentation", "presentations",
        "communication skills", "communicate", "communicating", "data storytelling",
        "storytelling", "non-technical", "non technical",
    ],
    "ethics_governance": [
        "gdpr", "data protection", "data governance", "data privacy", "data ethics",
        "information confidentiality",
    ],
}

NEGATIVE_PATTERNS = {
    "excel": [r"excellen\w*", r"excel(?:ling|led)? (?:in|at|within)"],
    "reporting": [r"report(?:s|ing)? (?:to|into|directly to)\b", r"direct reports?\b"],
    "etl": [r"demand pipeline", r"(?:drug|product|sales|candidate) pipeline"],
    "programming": [r"clinical coding", r"wellbeing programme", r"programme\b"],
    "cloud": [r"sales cloud", r"service cloud", r"oracle cloud services"],
    "sql": [r"oracle (?:fusion|epm|epbcs|pbcs)", r"hyperion"],
}

# Default paths relative to repo root (overridden by CLI args in experiments).
PATHS = {
    "corpus": "v3/manual_work/uk_analyst_corpus_v4_clean.csv",
    "gold_workbook": "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx",
}

# v4 evaluation defaults
RANDOM_SEED = 42
TAXONOMY_VERSION_STR = TAXONOMY_VERSION  # alias for code that expects this name
