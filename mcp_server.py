import re
import sys
import os
from sqlalchemy import create_engine, text
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv(override=True)

# Initialize FastMCP Server
mcp = FastMCP("mysql-mcp-server")

# Default database URL
DB_URL = None

if len(sys.argv) > 1:
    provided_arg = sys.argv[1]
    # Check if the CLI argument looks like a connection URL
    if provided_arg.startswith("mysql"):
        DB_URL = provided_arg

if not DB_URL:
    # Build connection URI programmatically from environment variables
    dialect = os.getenv("DB_DIALECT", "mysql")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "company_data")
    
    from sqlalchemy.engine import URL
    db_url_obj = URL.create(
        drivername=f"{dialect}+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=db_name
    )
    DB_URL = db_url_obj.render_as_string(hide_password=False)

# Initialize the SQLAlchemy Engine
engine = create_engine(DB_URL)

@mcp.tool()
def list_tables() -> str:
    """
    List all tables in the database.
    
    Returns:
        A comma-separated string containing the names of the tables in the database.
    """
    try:
        db_name = engine.url.database
        with engine.connect() as conn:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = :db_name 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            result = conn.execute(query, {"db_name": db_name})
            tables = [row[0] for row in result.fetchall()]
            
        if not tables:
            return "No tables found in the database."
        return f"Tables in the database: {', '.join(tables)}"
    except Exception as e:
        return f"Error listing tables: {str(e)}"

@mcp.tool()
def get_schema() -> str:
    """
    Retrieve the detailed database schema structure including column names, types, primary keys, and row counts.
    
    Returns:
        A formatted multi-line schema string containing the schema representation of all tables.
    """
    try:
        db_name = engine.url.database
        with engine.connect() as conn:
            query_cols = text("""
                SELECT table_name, column_name, data_type, column_key, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = :db_name
                ORDER BY table_name, ordinal_position
            """)
            cols_result = conn.execute(query_cols, {"db_name": db_name}).fetchall()
            
            if not cols_result:
                return "No tables or columns found in the database."
            
            # Group columns by table
            tables_dict = {}
            for row in cols_result:
                t_name, col_name, dtype, col_key, is_null, col_default = row
                if t_name not in tables_dict:
                    tables_dict[t_name] = []
                tables_dict[t_name].append({
                    "name": col_name,
                    "type": dtype,
                    "pk": col_key == "PRI",
                    "nullable": is_null == "YES",
                    "default": col_default
                })
            
            schema_parts = []
            for t_name, cols in tables_dict.items():
                col_lines = []
                for col in cols:
                    parts = [f"  {col['name']} {col['type'].upper()}"]
                    if col['pk']:
                        parts.append("PRIMARY KEY")
                    if not col['nullable']:
                        parts.append("NOT NULL")
                    if col['default'] is not None:
                        parts.append(f"DEFAULT {col['default']}")
                    col_lines.append(" ".join(parts))
                
                # Fetch row count safely for the table
                try:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM `{t_name}`"))
                    row_count = count_result.scalar()
                    count_hint = f"  -- {row_count} rows"
                except Exception:
                    count_hint = ""
                    
                schema_parts.append(
                    f"Table: {t_name}{count_hint}\nColumns:\n" + "\n".join(col_lines)
                )
                
        return "\n\n".join(schema_parts)
    except Exception as e:
        return f"Error extracting schema: {str(e)}"

@mcp.tool()
def query_db(sql_query: str) -> str:
    """
    Safely execute a read-only SELECT SQL query on the database.
    Only SELECT statements are permitted. Banned actions like INSERT, UPDATE, DELETE, etc., will be blocked.
    
    Args:
        sql_query: The raw SQL string (SELECT query) to execute.
        
    Returns:
        The result columns and matching database rows formatted as plain text, or an error message if invalid.
    """
    sql_upper = sql_query.upper().strip()
    
    # Layer 1 Safety check: Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are permitted on this database."
        
    # Layer 2 Safety check: Check for write keywords
    banned_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
        "BEGIN", "COMMIT", "ROLLBACK", "VACUUM", "REINDEX",
        "GRANT", "REVOKE", "SAVEPOINT", "RELEASE",
    ]
    for keyword in banned_keywords:
        if re.search(rf"\b{keyword}\b", sql_upper):
            return f"Error: Forbidden keyword detected: {keyword}"
            
    # Layer 3 Safety check: Semicolons and multiple statements
    stripped = sql_query.rstrip().rstrip(";")
    if ";" in stripped:
        return "Error: Multiple SQL statements are not allowed."
        
    # Layer 4: Execution via SQLAlchemy Engine
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = result.fetchmany(100)
            
            if not rows:
                return "No rows returned."
                
            result_str = f"Columns: {', '.join(columns)}\nRows (up to 100):\n"
            for row in rows:
                # Convert row tuple/mapping values to string
                row_values = tuple(row)
                result_str += f"- {row_values}\n"
            if len(rows) == 100:
                # Check if there are remaining rows
                try:
                    has_more = conn.execute(text(f"SELECT COUNT(*) FROM ({sql_query}) AS t")).scalar() > 100
                    if has_more:
                        result_str += f"... (and more rows exist)"
                except Exception:
                    pass
            return result_str
    except Exception as e:
        return f"Error executing query: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")

