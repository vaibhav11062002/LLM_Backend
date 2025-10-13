import logging
import sqlite3
import re
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from DMtool.sqlite_utils import add_sqlite_functions
from DMtool.source_table_manager import handle_query_execution

load_dotenv()


logger = logging.getLogger(__name__)
class SQLExecutor:
    """Executes SQL queries against the database"""
    
    def __init__(self, db_path=os.environ.get('DB_PATH')):
        """Initialize the SQL executor
        
        Parameters:
        db_path (str): Path to the SQLite database
        """
        self.db_path = db_path

    def _extract_target_table_name(self, query: str) -> Optional[str]:
        """
        Extract target table name from SQL query
        Add this method to the SQLExecutor class
        
        Args:
            query (str): SQL query
            
        Returns:
            str or None: Target table name
        """
        import re
        
        query = query.strip()
        insert_match = re.search(r'INSERT\s+INTO\s+([^\s\(]+)', query,re.IGNORECASE)
        if insert_match:
            return insert_match.group(1).strip('[]"`')
        update_match = re.search(r'UPDATE\s+([^\s]+)', query, re.IGNORECASE)
        if update_match:
            return update_match.group(1).strip('[]"`')
        alter_match = re.search(r'ALTER\s+TABLE\s+([^\s]+)', query, re.IGNORECASE)
        if alter_match:
            return alter_match.group(1).strip('[]"`')
        
        return None
    
    def execute_query(self, 
                     query: str, 
                     params: Optional[Dict[str, Any]] = None, 
                     fetch_results: bool = True,
                     commit: bool = True,
                     object_id: Optional[int] = None,
                     segment_id: Optional[int] = None,
                     project_id: Optional[int] = None,
                     session_id: Optional[str] = None,
                     planner_info: Optional[Dict[str, Any]] = None,
                     is_selection_criteria = False
                     ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute an SQL query with parameter binding
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        fetch_results (bool): Whether to fetch and return results
        commit (bool): Whether to commit changes
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: 
            Query results as a list of dictionaries, or error information
        """
        conn = None
        execution_successful = False
        target_table = None
        
        try:
            target_table = self._extract_target_table_name(query)
            
            conn = sqlite3.connect(self.db_path)
            add_sqlite_functions(conn)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if commit:
                conn.commit()
                execution_successful = True
            
            if commit and query.upper().strip().startswith('ALTER TABLE'):
                if object_id and segment_id and project_id:
                    if 'ADD COLUMN' in query.upper():
                        column_name = self._extract_column_name_from_alter(query, 'ADD')
                        if column_name:
                            add_column_metadata(column_name, object_id, segment_id, project_id)
                            logger.info(f"Successfully added column metadata for '{column_name}'")
                            
                    elif 'DROP COLUMN' in query.upper():
                        column_name = self._extract_column_name_from_alter(query, 'DROP')
                        if column_name:
                            remove_column_metadata(column_name, object_id, segment_id, project_id)
                            logger.info(f"Successfully removed column metadata for '{column_name}'")

            if commit and query.upper().strip().startswith('ALTER TABLE'):
                if object_id and segment_id and project_id:
                    pass
            if fetch_results:
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
            else:
                result = {"rowcount": cursor.rowcount}

            if execution_successful and target_table:
                try:
                    enhanced_result = handle_query_execution(
                        query=query,
                        target_table=target_table,
                        planner_info=planner_info,
                        params=params,
                        main_execution_successful=True,
                        session_id=session_id,
                        is_selection_criteria=is_selection_criteria
                    )
                    
                    if enhanced_result.get("sync_attempted"):
                        if enhanced_result.get("sync_successful"):
                            logger.debug(f"Source sync successful for {target_table}: {enhanced_result.get('reason')}")
                        else:
                            logger.warning(f"Source sync failed for {target_table}: {enhanced_result.get('reason')}")
                    
                    if enhanced_result.get("lineage_captured"):
                        logger.debug(f"Lineage captured for {target_table}")
                            
                except Exception as sync_error:
                    logger.warning(f"Enhanced source table processing error (non-critical): {sync_error}")
                
                return result
                    
        except sqlite3.Error as e:
            if conn and commit:
                conn.rollback()                    
            logger.error(f"SQLite error: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:
            if conn and commit:
                conn.rollback()                
            logger.error(f"Error executing query: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:
            if conn:
                conn.close()

    def sync_src_to_target(self, target_table: str):
        """
        Sync source table to target table with column intersection handling
        Parameters:
        target_table (str): Name of the target table
        Returns:
        bool: True if sync was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        logger.info(f"Starting sync from {target_table}_src to {target_table}")
        try:
            # Get column names from source table
            cursor.execute(f"PRAGMA table_info({target_table}_src)")
            src_columns = [row[1] for row in cursor.fetchall()]
            
            # Get column names from target table
            cursor.execute(f"PRAGMA table_info({target_table})")
            target_columns = [row[1] for row in cursor.fetchall()]
            
            # Find intersection of columns (common columns)
            common_columns = list(set(src_columns) & set(target_columns))
            
            if not common_columns:
                logger.error(f"No common columns found between {target_table}_src and {target_table}")
                return False
            
            # Create SELECT clause for source table (only common columns)
            src_select = ", ".join(common_columns)
            
            # Create column list for INSERT (all target columns)
            target_column_list = ", ".join(target_columns)
            
            # Create SELECT clause that includes NULLs for non-common target columns
            select_values = []
            for col in target_columns:
                if col in common_columns:
                    select_values.append(col)
                else:
                    select_values.append("NULL")
            
            select_clause = ", ".join(select_values)

            cursor.execute(f"DELETE from {target_table}")
            conn.commit()

            sync_query = f"""
            INSERT INTO {target_table} ({target_column_list})
            SELECT {select_clause} FROM {target_table}_src
            """
            
            cursor.execute(sync_query)
            conn.commit()
            
            logger.info(f"Successfully synced data from {target_table}_src to {target_table}")
            logger.info(f"Common columns synced: {common_columns}")
            logger.info(f"Target-only columns filled with NULL: {set(target_columns) - set(common_columns)}")
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error during sync: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            conn.close()

    def execute_and_fetch_df(self, query: str, params: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a query and return results as a pandas DataFrame
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with results or error information
        """
        conn = None
        try:

            conn = sqlite3.connect(self.db_path)
            

            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
                
            return df
                
        except sqlite3.Error as e:

            logger.error(f"SQLite error in execute_and_fetch_df: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:

            logger.error(f"Error in execute_and_fetch_df: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:

            if conn:
                conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Parameters:
        table_name (str): Name of the table to check
        
        Returns:
        bool: True if the table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=? Limit 1
        """
        
        result = self.execute_query(query, {"table_name": table_name})
        
        if isinstance(result, list):
            return len(result) > 0
        
        return False
    
    def get_table_schema(self, table_name: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the schema information for a table
        
        Parameters:
        table_name (str): Name of the table
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: Schema information or error
        """
        query = f"PRAGMA table_info({table_name})"
        
        return self.execute_query(query)
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get a sample of rows from a table as a DataFrame
        
        Parameters:
        table_name (str): Name of the table
        limit (int): Maximum number of rows to return
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with sample rows or error
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        return self.execute_and_fetch_df(query)
    
    def backup_table(self, table_name: str) -> Tuple[bool, str]:
        """
        Create a backup of a table
        
        Parameters:
        table_name (str): Name of the table to backup
        
        Returns:
        Tuple[bool, str]: Success status and backup table name
        """

        backup_name = f"{table_name}_backup_{uuid.uuid4().hex[:8]}"
        
        query = f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}"
        
        result = self.execute_query(query, fetch_results=False)
        
        if isinstance(result, dict) and "error_type" in result:
            return False, ""
        
        return True, backup_name
    
    def execute_multi_statement_query(self, multi_sql, sql_params,context_manager=None, session_id=None,object_id=None, segment_id=None, project_id=None,planner_info=None):
        """Execute multiple SQL statements with recovery support"""
        try:            

            statements = self.split_sql_statements(multi_sql)
            

            if session_id:
                execution_state = context_manager.load_multi_query_state(session_id)
                if execution_state:
                    return self.resume_execution(execution_state, statements, sql_params, context_manager, session_id, object_id, segment_id, project_id,planner_info)
            

            return self.execute_with_recovery(statements, sql_params, session_id, context_manager, object_id, segment_id, project_id,planner_info)
            
        except Exception as e:
            logger.error(f"Error in execute_multi_statement_query: {e}")
            return {"error_type": "MultiQueryError", "error_message": str(e)}
        
    def resume_execution(self, execution_state, statements, sql_params, context_manager, session_id, object_id=None, segment_id=None, project_id=None,planner_info: Optional[Dict[str, Any]] = None):
        """Resume execution from a previously failed multi-query operation"""
        try:
            logger.info(f"Resuming multi-query execution from session {session_id}")
            

            completed_count = execution_state.get("completed_count", 0)
            failed_count = execution_state.get("failed_count", 0)
            previous_statements = execution_state.get("statements", [])
            

            if len(statements) != len(previous_statements):
                logger.warning("Statement count mismatch during resume, starting fresh execution")
                return self.execute_with_recovery(statements, sql_params, session_id, context_manager,object_id=object_id, segment_id=segment_id, project_id=project_id)
            

            results = []
            

            for prev_statement in previous_statements:
                if prev_statement.get("status") == "completed":
                    results.append(prev_statement.get("result", {}))
            

            resume_index = completed_count
            

            execution_state["resumed_at"] = datetime.now().isoformat() if 'datetime' in globals() else "resumed"
            execution_state["resume_attempt"] = execution_state.get("resume_attempt", 0) + 1
            
            logger.info(f"Resuming from statement {resume_index + 1}/{len(statements)}")
            

            for i in range(resume_index, len(statements)):
                statement = statements[i]
                
                try:
                    logger.info(f"Executing statement {i+1}/{len(statements)} (resumed): {statement[:100]}...")
                    

                    result = self.execute_query(statement, sql_params, fetch_results=False,object_id=object_id, segment_id=segment_id, project_id=project_id,session_id=session_id,planner_info=planner_info)
                    
                    if isinstance(result, dict) and "error_type" in result:

                        statement_result = {
                            "statement": statement,
                            "index": i,
                            "status": "failed", 
                            "error": result,
                            "can_retry": self._can_retry_statement(statement, result),
                            "resume_attempt": execution_state.get("resume_attempt", 1)
                        }
                        execution_state["failed_count"] += 1
                        

                        if i < len(execution_state["statements"]):
                            execution_state["statements"][i] = statement_result
                        else:
                            execution_state["statements"].append(statement_result)
                        

                        context_manager.save_multi_query_state(session_id, execution_state)
                        

                        return {
                            "multi_query_result": True,
                            "completed_statements": execution_state["completed_count"],
                            "failed_statement_index": i,
                            "failed_statement": statement,
                            "error": result,
                            "can_resume": True,
                            "session_id": session_id,
                            "all_results": results,
                            "is_resumed_execution": True,
                            "resume_attempt": execution_state.get("resume_attempt", 1)
                        }
                    else:

                        statement_result = {
                            "statement": statement,
                            "index": i,
                            "status": "completed",
                            "result": result,
                            "resumed_execution": True
                        }
                        execution_state["completed_count"] += 1
                        results.append(result)
                        

                        if i < len(execution_state["statements"]):
                            execution_state["statements"][i] = statement_result
                        else:
                            execution_state["statements"].append(statement_result)
                        

                        prev_statement = next((s for s in previous_statements if s.get("index") == i), None)
                        if prev_statement and prev_statement.get("status") == "failed":
                            execution_state["failed_count"] = max(0, execution_state["failed_count"] - 1)
                    
                except Exception as e:

                    logger.error(f"Unexpected error during resume at statement {i}: {e}")
                    return {
                        "error_type": "ResumeExecutionError",
                        "error_message": f"Resume failed at statement {i+1}: {str(e)}",
                        "completed_statements": execution_state["completed_count"],
                        "failed_statement": statement,
                        "session_id": session_id,
                        "is_resumed_execution": True
                    }
            

            context_manager.cleanup_multi_query_state(session_id)
            
            logger.info(f"Multi-query execution resumed and completed successfully. Total statements: {len(statements)}")
            
            return {
                "multi_query_result": True, 
                "completed_statements": len(statements),
                "all_results": results,
                "success": True,
                "is_resumed_execution": True,
                "resume_attempt": execution_state.get("resume_attempt", 1)
            }
            
        except Exception as e:
            logger.error(f"Error in resume_execution: {e}")
            return {
                "error_type": "ResumeExecutionError",
                "error_message": f"Failed to resume execution: {str(e)}",
                "session_id": session_id
            }

    def split_sql_statements(self, multi_sql):
        """Split multi-SQL into individual statements, handling edge cases"""
        statements = []
        current_statement = ""
        in_string = False
        string_char = None
        
        for char in multi_sql:
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif char == ';' and not in_string:
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
                continue
            
            current_statement += char
        

        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements

    def execute_with_recovery(self, statements, sql_params, session_id, context_manager, object_id=None, segment_id=None, project_id=None,planner_info: Optional[Dict[str, Any]] = None):
        """Execute statements one by one with failure tracking"""
        results = []
        execution_state = {
            "statements": [],
            "completed_count": 0,
            "failed_count": 0,
            "session_id": session_id,
            "original_sql": "; ".join(statements)
        }
        
        for i, statement in enumerate(statements):
            try:
                logger.info(f"Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                

                result = self.execute_query(statement, sql_params, fetch_results=False,object_id=object_id, segment_id=segment_id, project_id=project_id,session_id=session_id,planner_info=planner_info)
                
                if isinstance(result, dict) and "error_type" in result:

                    statement_result = {
                        "statement": statement,
                        "index": i,
                        "status": "failed", 
                        "error": result,
                        "can_retry": self._can_retry_statement(statement, result)
                    }
                    execution_state["failed_count"] += 1
                    execution_state["statements"].append(statement_result)
                    

                    if session_id:
                        context_manager.save_multi_query_state(session_id, execution_state)
                    

                    return {
                        "multi_query_result": True,
                        "completed_statements": execution_state["completed_count"],
                        "failed_statement_index": i,
                        "failed_statement": statement,
                        "error": result,
                        "can_resume": True,
                        "session_id": session_id,
                        "all_results": results
                    }
                else:

                    statement_result = {
                        "statement": statement,
                        "index": i,
                        "status": "completed",
                        "result": result
                    }
                    execution_state["completed_count"] += 1
                    results.append(result)
                    
                execution_state["statements"].append(statement_result)
                
            except Exception as e:

                logger.error(f"Unexpected error executing statement {i}: {e}")
                return {
                    "error_type": "StatementExecutionError",
                    "error_message": f"Statement {i+1} failed: {str(e)}",
                    "completed_statements": execution_state["completed_count"],
                    "failed_statement": statement
                }
        

        if session_id:
            context_manager.cleanup_multi_query_state(session_id)
        
        return {
            "multi_query_result": True, 
            "completed_statements": len(statements),
            "all_results": results,
            "success": True
        }

    def _can_retry_statement(self, statement, error):
        """Determine if a failed statement can be safely retried"""
        error_msg = error.get("error_message", "").lower()
        

        if "alter table" in statement.lower() and "add column" in statement.lower():
            if "already exists" in error_msg or "duplicate column" in error_msg:
                return False
        

        if any(keyword in statement.lower() for keyword in ["update", "insert", "select"]):
            return True
            
        return True
    
    def _extract_column_name_from_alter(self, query: str, operation: str) -> str:
        """Extract column name from ALTER TABLE query"""
        try:
            if operation == 'ADD':
                pattern = r'ADD\s+COLUMN\s+(\w+)'
            else:
                pattern = r'DROP\s+COLUMN\s+(\w+)'
            match = re.search(pattern, query, re.IGNORECASE)
            return match.group(1) if match else None
        except:
            return None

def add_column_metadata(column_name: str, object_id: int, segment_id: int, project_id: int) -> bool:
    """
    Add column metadata to connection_fields table after successful ALTER TABLE ADD COLUMN
    
    Parameters:
    column_name (str): Name of the column that was added
    object_id (int): Object ID for the field mapping
    segment_id (int): Segment ID for the field mapping  
    project_id (int): Project ID for the field mapping
    
    Returns:
    bool: True if successful, False otherwise
    """
    conn = None
    try:
        conn = sqlite3.connect(os.environ.get('DB_PATH'))
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(field_id) FROM connection_fields")
        max_field_id = cursor.fetchone()[0]
        next_field_id = (max_field_id + 1) if max_field_id is not None else 1
        cursor.execute("""
            INSERT INTO connection_fields 
            (field_id, fields, description, isMandatory, obj_id_id, project_id_id, segement_id_id, sap_structure, isKey)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            next_field_id,
            column_name,
            "",
            "False",
            object_id,
            project_id,
            segment_id,
            "",
            "False"
        ))
        
        conn.commit()
        
        logger.info(f"Successfully added column metadata for '{column_name}' with field_id {next_field_id}")
        return True
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error adding column metadata for '{column_name}': {e}")
        return False
    finally:
        if conn:
            conn.close()


def remove_column_metadata(column_name: str, object_id: int, segment_id: int, project_id: int) -> bool:
    """
    Remove column metadata from connection_fields table after successful ALTER TABLE DROP COLUMN
    
    Parameters:
    column_name (str): Name of the column that was dropped
    object_id (int): Object ID for the field mapping
    segment_id (int): Segment ID for the field mapping
    project_id (int): Project ID for the field mapping
    
    Returns:
    bool: True if successful, False otherwise
    """
    conn = None
    try:
        conn = sqlite3.connect(os.environ.get('DB_PATH'))
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM connection_fields 
            WHERE fields = ? 
            AND obj_id_id = ? 
            AND segement_id_id = ? 
            AND project_id_id = ?
        """, (column_name, object_id, segment_id, project_id))
        
        rows_deleted = cursor.rowcount
        conn.commit()
        
        if rows_deleted > 0:
            logger.info(f"Successfully removed column metadata for '{column_name}' ({rows_deleted} rows deleted)")
            return True
        else:
            logger.warning(f"No metadata found to remove for column '{column_name}'")
            return False
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error removing column metadata for '{column_name}': {e}")
        return False
    finally:
        if conn:
            conn.close()
