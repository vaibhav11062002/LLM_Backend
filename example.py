from dqtool.dqtool import DQTool
import pandas as pd
TABLE_NAME = "t_19_MATMASTERBASICENRZROHNVALCOMPRANDOM20250708_0955xlsxRandom_Samples_MATMASTERBASI"

def main():
    dq = DQTool(table_name=TABLE_NAME)
    
    # Example natural language query
    user_query = "get all the records where length of VALID_FROM_DATE_FOR_X_PLANT_MATL_STATUS is greater than 8"
    
    result = dq.process_query(user_query)
    
    if result['success']:
        print("✓ Query succeeded!")
        print("SQL Query:", result['sql_query'])
        print("Results:")
        print(len(result['results']))
        df = pd.DataFrame(result['results'])
        print(df.head()) 
        # for row in result['results']:
        #     print(row)
    else:
        print("❌ Query failed:", result.get('error', 'Unknown error'))
        if 'last_error' in result:
            print("Last error:", result['last_error'])

if __name__ == "__main__":
    print("came")
    main()