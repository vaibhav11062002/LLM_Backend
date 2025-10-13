import sqlite3

# Get database path and SQL statement from the user
db_path = r"C:\Users\gajananda.mallidi\Downloads\NEWPUSH\NEWPUSH\LLM_Backend\db.sqlite3"
sql_code ="""
    select strftime('%w',) from t_75_Product_Basic_Data_mandatory
"""

# Connect to the database
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if it's a SELECT query to print results
    if sql_code.strip().lower().startswith('select'):
        cursor.execute(sql_code)
        # Fetch and print all results
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    else:
        cursor.execute(sql_code)
        conn.commit()
        print("SQL executed successfully.")

    cursor.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")