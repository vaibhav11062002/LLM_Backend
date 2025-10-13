"""
DQTool - Data Query Tool: Natural Language to SQL Query Converter

A tool that converts natural language queries to SQL queries with intelligent
value matching and error correction.
"""

import os
import re
import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher
import yaml
from dotenv import load_dotenv

from dqtool.llm_config import create_gemini_llm
from dqtool.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# same logic as in DMtool
def find_closest_match(query, word_list, threshold=0.6):
    """
    Find the closest matching word from a list using fuzzy string matching.
    
    Args:
        query (str): The search term (possibly with typos)
        word_list (list): List of valid words to match against
        threshold (float): Minimum similarity score (0.0 to 1.0)
        
    Returns:
        dict: Contains 'match' (best matching word), 'score' (similarity score), 
              and 'all_matches' (list of all matches above threshold)
    """
    if not query or not word_list:
        return {"match": None, "score": 0.0, "all_matches": []}
    
    clean_query = re.sub(r'\s+', ' ', query.strip().lower())
    
    matches = []
    for word in word_list:
        clean_word = word.lower()
        
        overall_sim = SequenceMatcher(None, clean_query, clean_word).ratio()
        
        substring_bonus = 0
        if clean_query in clean_word or clean_word in clean_query:
            substring_bonus = 0.2
        
        query_words = clean_query.split()
        word_words = clean_word.split()
        word_level_sim = 0
        
        if len(query_words) > 1 or len(word_words) > 1:
            word_matches = 0
            total_words = max(len(query_words), len(word_words))
            
            for q_word in query_words:
                best_word_match = 0
                for w_word in word_words:
                    word_sim = SequenceMatcher(None, q_word, w_word).ratio()
                    best_word_match = max(best_word_match, word_sim)
                word_matches += best_word_match
            
            word_level_sim = word_matches / total_words if total_words > 0 else 0
        
        final_score = (overall_sim * 0.4) + (substring_bonus) + (word_level_sim * 0.5)
        final_score = min(final_score, 1.0)
        
        if final_score >= threshold:
            matches.append({
                'word': word,
                'score': final_score,
                'original_word': word
            })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    result = {
        'match': matches[0]['word'] if matches else None,
        'score': matches[0]['score'] if matches else 0.0,
        'all_matches': [(m['word'], round(m['score'], 3)) for m in matches]
    }
    
    return result


class DQTool:
    """
    Data Query Tool - Converts natural language queries to SQL with fuzzy matching and validation
    """
    
    def __init__(self, table_name: str, db_path: Optional[str] = None):
        """
        Initialize the DQTool
        
        Args:
            table_name: Name of the table to query
            db_path: Path to SQLite database file (defaults to DB_PATH from env)
        """
        self.db_path = db_path or os.getenv("DB_PATH")
        
        if not self.db_path:
            raise ValueError("Database path must be provided or set in DB_PATH environment variable")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        self.table_name = table_name
        self.llm = create_gemini_llm()
        self.prompts = self._load_prompts()
        self.table_metadata = self._fetch_table_metadata(table_name)
        self.max_retries = 3
        
        logger.info(f"Initialized DQTool for table: {table_name} in database: {self.db_path}")
    # extracts all the file names from connection_tablemeta
    def _get_all_file_names(self) -> List[str]:
        """
        Get all file names from connection_tablemeta
        
        Returns:
            List of file names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_name FROM connection_tablemeta")
            file_names = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return file_names
        except Exception as e:
            logger.error(f"Error fetching file names: {e}")
            return []
    
    def get_available_files(self) -> List[Dict[str, str]]:
        """
        Get all available files with their table mappings
        Useful for showing users what files they can join with
        
        Returns:
            List of dicts with file_name, table_name, and primary_keys
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_name, table_name, primary_keys FROM connection_tablemeta")
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'file_name': row[0],
                    'table_name': row[1],
                    'primary_keys': row[2]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error fetching available files: {e}")
            return []
    
    def _get_table_info_from_file(self, file_name: str) -> Optional[Dict[str, str]]:
        """
        Get table name and primary key from file name using connection_tablemeta
        Uses fuzzy matching if exact match not found
        Enhanced to handle missing file extensions
        
        Args:
            file_name: Name of the file (may have typos)
            
        Returns:
            Dict with 'table_name', 'primary_key', 'file_name' (matched), 
            'match_type', and 'confidence', or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute(
                "SELECT file_name, table_name, primary_keys FROM connection_tablemeta WHERE file_name = ?",
                (file_name,)
            )
            result = cursor.fetchone()
            
            if result:
                conn.close()
                logger.info(f"Exact match found for file: {file_name}")
                return {
                    'file_name': result[0],
                    'table_name': result[1],
                    'primary_keys': result[2],
                    'match_type': 'exact',
                    'confidence': 1.0
                }
            
            # Try with common extensions if no extension provided
            if '.' not in file_name:
                for ext in ['.xlsx', '.csv', '.xls', '.xlsm']:
                    cursor.execute(
                        "SELECT file_name, table_name, primary_keys FROM connection_tablemeta WHERE file_name = ?",
                        (file_name + ext,)
                    )
                    result = cursor.fetchone()
                    if result:
                        conn.close()
                        logger.info(f"Exact match found with extension: {file_name + ext}")
                        return {
                            'file_name': result[0],
                            'table_name': result[1],
                            'primary_key': result[2] if result[2] else None,
                            'match_type': 'exact_with_extension',
                            'confidence': 1.0
                        }
            
            # No exact match, try fuzzy matching
            cursor.execute("SELECT file_name, table_name, primary_keys FROM connection_tablemeta")
            all_files = cursor.fetchall()
            conn.close()
            
            if not all_files:
                logger.warning("No files found in connection_tablemeta")
                return None
            
            # Use fuzzy matching to find closest file name
            file_names = [f[0] for f in all_files]
            search_terms = [file_name]
            
            # Add versions with extensions if not present
            if '.' not in file_name:
                for ext in ['.xlsx', '.csv', '.xls', '.xlsm']:
                    search_terms.append(file_name + ext)
            
            best_match = None
            best_score = 0
            
            # Try fuzzy matching with all search terms
            for search_term in search_terms:
                match_result = find_closest_match(search_term, file_names, threshold=0.6)
                
                if match_result['match'] and match_result['score'] > best_score:
                    best_match = match_result
                    best_score = match_result['score']
            
            if best_match and best_match['match']:
                # Find the matching record
                for file_record in all_files:
                    if file_record[0] == best_match['match']:
                        logger.info(
                            f"Fuzzy match found: '{file_name}' -> '{best_match['match']}' "
                            f"(confidence: {best_match['score']:.3f})"
                        )
                        return {
                            'file_name': file_record[0],
                            'table_name': file_record[1],
                            'primary_key': file_record[2] if file_record[2] else None,
                            'match_type': 'fuzzy',
                            'confidence': best_match['score'],
                            'original_input': file_name,
                            'alternatives': best_match['all_matches'][:5]
                        }
            
            logger.warning(f"No match found for file: {file_name}")
            logger.info(f"Available files: {file_names[:10]}")
            return None
                
        except Exception as e:
            logger.error(f"Error fetching table info for file {file_name}: {e}")
            return None
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from YAML file with UTF-8 encoding"""
        prompts_path = "prompts.yml"
        
        if not os.path.exists(prompts_path):
            logger.warning("prompts.yml not found, creating default prompts")
            default_prompts = self._get_default_prompts()
            
            with open(prompts_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_prompts, f, default_flow_style=False, allow_unicode=True)
            
            return default_prompts
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for query processing"""
        return {
            'query_understanding': """Analyze this natural language query and extract structured information.

Primary Table: {table_name}
Available Tables and Columns: {all_tables_info}

User Query: {user_query}

Extract:
1. Which columns are being requested (SELECT) - specify table.column for JOIN queries
2. What filters/conditions are being applied (WHERE)
3. Any JOIN requirements (file names or explicit join columns)
4. Any sorting requirements (ORDER BY)
5. Any aggregations (COUNT, SUM, AVG, etc.)
6. Limit on results

CRITICAL RULES FOR UNDERSTANDING:

A. JOIN DETECTION:
   - "join with [file_name]" or "from [file_name]" → indicates JOIN requirement
   - "join on [column]" → user specifies explicit join column
   - If no join column specified, use automatic primary key join
   - Store file names for JOIN resolution

B. NULL/BLANK VALUE DETECTION:
   - Words like "blank", "empty", "null", "missing", "not filled" indicate NULL checks
   - "where X is blank" → check if X IS NULL
   - "where X is not blank" → check if X IS NOT NULL
   - "any column is blank" → check if ANY of the columns IS NULL (use OR logic)
   - When checking for blank/null values, DO NOT use the "values" field

C. LOGICAL OPERATORS:
   - "any of" / "either" / "or" → OR logic between conditions
   - "all of" / "both" / "and" → AND logic between conditions

D. COLUMN QUALIFICATION:
   - For JOIN queries, specify table.column format
   - For single table, column name alone is fine

Return ONLY a JSON object with this structure:
{{
    "intent": "select or update",
    "columns": ["table1.column1", "table2.column2"] or ["*"],
    "joins": [
        {{
            "file_name": "filename.csv",
            "join_column": "column_name",
            "join_type": "INNER"
        }}
    ],
    "filters": [
        {{
            "field": "column_name",
            "operator": "=",
            "values": ["value1"],
            "null_check": false,
            "logical_connector": "AND"
        }}
    ],
    "order_by": {{"column": "column_name", "direction": "ASC"}},
    "aggregations": [{{"function": "COUNT", "column": "*"}}],
    "limit": 10
}}

DO NOT include any explanation, ONLY return the JSON object.""",
            
            'sql_generation': """Generate a SQL query based on this analyzed plan.

Primary Table: {table_name}
All Tables Schema: {schema}
Join Metadata: {join_metadata}

Query Plan:
{query_plan}

Corrected Filter Values:
{corrected_values}

CRITICAL SQL GENERATION RULES:

1. JOIN HANDLING:
   - Use the join metadata to construct proper JOIN clauses
   - Format: FROM table1 INNER JOIN table2 ON table1.pk = table2.fk
   - Use table aliases for clarity when needed
   - Ensure all columns in SELECT are properly qualified with table names

2. NULL VALUE HANDLING:
   - If filter has "null_check": true, use IS NULL or IS NOT NULL
   - NEVER use IN () with empty array

3. LOGICAL CONNECTORS:
   - Pay attention to "logical_connector" field
   - Use parentheses for OR conditions: (col1 IS NULL OR col2 IS NULL)

4. COLUMN QUALIFICATION:
   - For JOINs, always qualify columns: table_name.column_name
   - Handle ambiguous columns by using proper table prefixes

5. SQLITE COMPATIBILITY:
   - Use single quotes for string values
   - Proper escaping for special characters
   - No nested subqueries unless necessary

Return ONLY the SQL query, no explanations or markdown formatting.""",
            
            'query_fixer': """The following SQL query failed with an error. Fix it.

Original Query:
{original_query}

Error Message:
{error_message}

Schema Information:
{schema}

Available Data Sample:
{sample_data}

COMMON ISSUES:
1. Ambiguous column names in JOINs - add table prefixes
2. Empty IN clause - use IS NULL instead
3. Incorrect NULL comparison - use IS NULL/IS NOT NULL
4. Column doesn't exist - check schema
5. JOIN condition errors - verify foreign/primary keys

Fix the query and return ONLY the corrected SQL query, no explanations."""
        }
    
    def _fetch_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Fetch metadata about a table structure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            if not columns:
                raise ValueError(f"Table '{table_name}' not found in database")
            
            metadata = {
                'table_name': table_name,
                'columns': [],
                'column_types': {},
                'sample_data': {}
            }
            
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                metadata['columns'].append(col_name)
                metadata['column_types'][col_name] = col_type
                
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10")
                samples = [row[0] for row in cursor.fetchall() if row[0] is not None]
                metadata['sample_data'][col_name] = samples
            
            conn.close()
            
            logger.info(f"Fetched metadata for table {table_name}: {len(metadata['columns'])} columns")
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching table metadata: {e}")
            raise
    
    def _get_actual_column_values(self, column: str, table_name: Optional[str] = None, limit: int = 50) -> List[str]:
        """Get distinct values from a column"""
        try:
            table = table_name or self.table_name
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}")
            values = []
            for row in cursor.fetchall():
                val = row[0]
                # Convert to string, handling floats specially
                if isinstance(val, float):
                    # Remove .0 from whole numbers for better matching
                    if val.is_integer():
                        values.append(str(int(val)))
                    else:
                        values.append(str(val))
                else:
                    values.append(str(val))
            
            conn.close()
            return values
            
        except Exception as e:
            logger.error(f"Error getting column values for {column}: {e}")
            return []
    
    def _find_closest_match(self, query: str, word_list: List[str], 
                           threshold: float = 0.6, max_matches: int = 3, 
                           similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Find closest matching words using fuzzy string matching"""
        if not query or not word_list:
            return {"matches": [], "scores": [], "all_matches": []}
        
        clean_query = re.sub(r'\s+', ' ', str(query).strip().lower())
        
        matches = []
        for word in word_list:
            clean_word = str(word).lower()
            
            overall_sim = SequenceMatcher(None, clean_query, clean_word).ratio()
            substring_bonus = 0.2 if (clean_query in clean_word or clean_word in clean_query) else 0
            
            query_words = clean_query.split()
            word_words = clean_word.split()
            
            if len(query_words) > 1 or len(word_words) > 1:
                word_level_sim = 0
                word_matches = 0
                total_words = max(len(query_words), len(word_words))
                
                for q_word in query_words:
                    best_word_match = max([
                        SequenceMatcher(None, q_word, w_word).ratio() 
                        for w_word in word_words
                    ] or [0])
                    word_matches += best_word_match
                
                word_level_sim = word_matches / total_words if total_words > 0 else 0
                final_score = (overall_sim * 0.4) + substring_bonus + (word_level_sim * 0.5)
            else:
                final_score = (overall_sim * 0.7) + substring_bonus
            
            final_score = min(final_score, 1.0)
            
            if final_score >= threshold:
                matches.append({
                    'word': word,
                      'score': final_score
                      })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        if not matches:
            return {"matches": [], "scores": [], "all_matches": []}
        
        best_score = matches[0]['score']
        
        if best_score == 1.0:
            return {
                "matches": [matches[0]['word']],
                "scores": [matches[0]['score']],
                "all_matches": [(m['word'], round(m['score'], 3)) for m in matches]
            }
        
        if best_score >= similarity_threshold:
            high_matches = [
                m for m in matches[:max_matches]
                if m['score'] >= max(similarity_threshold, best_score - 0.1)
            ]
            
            return {
                "matches": [m['word'] for m in high_matches],
                "scores": [m['score'] for m in high_matches],
                "all_matches": [(m['word'], round(m['score'], 3)) for m in matches]
            }
        
        return {
            "matches": [matches[0]['word']],
            "scores": [matches[0]['score']],
            "all_matches": [(m['word'], round(m['score'], 3)) for m in matches]
        }
    
    def _resolve_joins(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve JOIN requirements by fetching table metadata from connection_tablemeta
        Uses fuzzy matching to find correct table for file names
        
        Args:
            plan: Query plan with potential JOIN specifications
            
        Returns:
            Updated plan with resolved JOIN metadata and match information
        """
        if 'joins' not in plan or not plan['joins']:
            return plan
        
        join_metadata = []
        all_tables_metadata = {self.table_name: self.table_metadata}
        match_warnings = []
        join_errors = []
        
        for join_spec in plan['joins']:
            file_name = join_spec.get('file_name')
            user_join_column = join_spec.get('join_column')
            join_type = join_spec.get('join_type', 'INNER')
            
            if not file_name:
                error_msg = "JOIN specification missing file_name"
                logger.warning(error_msg)
                join_errors.append(error_msg)
                continue
            
            # Get table info from connection_tablemeta (with fuzzy matching)
            table_info = self._get_table_info_from_file(file_name)
            
            if not table_info:
                error_msg = f"Could not resolve table for file: {file_name}"
                logger.error(error_msg)
                join_errors.append(error_msg)
                
                # Get available files for suggestion
                available_files = self._get_all_file_names()
                if available_files:
                    join_errors.append(f"Available files: {', '.join(available_files[:10])}")
                continue
            
            # Log fuzzy match information
            if table_info.get('match_type') == 'fuzzy':
                warning = (
                    f"Fuzzy match: '{table_info.get('original_input', file_name)}' "
                    f"matched to '{table_info['file_name']}' "
                    f"(confidence: {table_info['confidence']:.3f})"
                )
                logger.info(warning)
                match_warnings.append(warning)
                
                # Add alternatives if available
                if 'alternatives' in table_info and len(table_info['alternatives']) > 1:
                    alts = [f"{f[0]} ({f[1]})" for f in table_info['alternatives'][:3]]
                    match_warnings.append(f"Alternative matches: {', '.join(alts)}")
            
            join_table = table_info['table_name']
            primary_keys = table_info.get('primary_keys')
            
            # Fetch metadata for the join table
            if join_table not in all_tables_metadata:
                all_tables_metadata[join_table] = self._fetch_table_metadata(join_table)
            
            # Determine join column based on priority:
            # 1. User provided join column
            # 2. Primary key from connection_tablemeta
            # 3. Error if neither available
            
            if user_join_column:
                # User specified explicit join column - verify it exists in both tables
                join_column = user_join_column
                
                # Verify column exists in primary table
                if join_column not in self.table_metadata['columns']:
                    error_msg = f"Join column '{join_column}' not found in primary table '{self.table_name}'. Available columns: {', '.join(self.table_metadata['columns'][:10])}"
                    logger.error(error_msg)
                    join_errors.append(error_msg)
                    continue
                
                # Verify column exists in join table
                if join_column not in all_tables_metadata[join_table]['columns']:
                    error_msg = f"Join column '{join_column}' not found in join table '{join_table}'. Available columns: {', '.join(all_tables_metadata[join_table]['columns'][:10])}"
                    logger.error(error_msg)
                    join_errors.append(error_msg)
                    continue
                
                logger.info(f"Using user-specified join column: {join_column}")
                join_source = 'user_specified'
                
            elif primary_keys:
                # Use primary key from connection_tablemeta
                join_column = primary_keys
                
                # Verify primary key exists in both tables
                if join_column not in self.table_metadata['columns']:
                    error_msg = f"Primary key '{join_column}' from connection_tablemeta not found in primary table '{self.table_name}'. Please specify join column explicitly."
                    logger.error(error_msg)
                    join_errors.append(error_msg)
                    continue
                
                if join_column not in all_tables_metadata[join_table]['columns']:
                    error_msg = f"Primary key '{join_column}' from connection_tablemeta not found in join table '{join_table}'. Please specify join column explicitly."
                    logger.error(error_msg)
                    join_errors.append(error_msg)
                    continue
                
                logger.info(f"Using primary key from connection_table_meta: {join_column}")
                join_source = 'primary_keys'
                
            else:
                # Neither user column nor primary key available
                error_msg = (
                    f"JOIN condition not provided for file '{file_name}'. "
                    f"No primary_keys found in connection_tablemeta and no explicit join column specified. "
                    f"Please either: 1) Specify join column in query (e.g., 'join on column_name'), "
                    f"or 2) Update connection_tablemeta table with primary_keys for this file."
                )
                logger.error(error_msg)
                join_errors.append(error_msg)
                continue
            
            join_metadata.append({
                'file_name': table_info['file_name'],  # Use matched file name
                'original_file_name': file_name,  # Keep original for reference
                'table_name': join_table,
                'join_column': join_column,
                'primary_table_column': join_column,  # Assuming same column name in primary table
                'join_type': join_type,
                'match_type': table_info.get('match_type', 'exact'),
                'match_confidence': table_info.get('confidence', 1.0),
                'join_source': join_source
            })
            
            logger.info(f"Resolved JOIN: {self.table_name}.{join_column} = {join_table}.{join_column}")
        
        plan['resolved_joins'] = join_metadata
        plan['all_tables_metadata'] = all_tables_metadata
        
        # Add match warnings to plan for user feedback
        if match_warnings:
            plan['join_match_warnings'] = match_warnings
        
        # Add join errors - these are critical and should stop execution
        if join_errors:
            plan['join_errors'] = join_errors
        
        return plan
    
    def _understand_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Step 1: Understand the user's natural language query including JOINs
        """
        logger.info(f"Understanding query: {user_query}")
        
        # Prepare info about all available tables
        all_tables_info = f"Primary Table: {self.table_name}\n"
        all_tables_info += f"Columns: {', '.join(self.table_metadata['columns'])}\n"
        
        prompt = self.prompts['query_understanding'].format(
            table_name=self.table_name,
            all_tables_info=all_tables_info,
            user_query=user_query
        )
        
        response = self.llm.generate(prompt)
        
        if not response:
            logger.error("Failed to get response from LLM")
            return None
        
        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            plan = json.loads(response_clean)
            
            # Resolve JOINs if present
            if 'joins' in plan and plan['joins']:
                plan = self._resolve_joins(plan)
            
            logger.info(f"Query plan: {json.dumps(plan, indent=2)}")
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse query plan JSON: {e}")
            logger.error(f"Response was: {response}")
            return None
    
    def _validate_and_fix_filter_values(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Validate filter values and apply fuzzy matching
        Handles both single table and JOIN scenarios
        """
        logger.info("Validating and fixing filter values")
        
        if 'filters' not in plan or not plan['filters']:
            return plan
        
        # Get all tables metadata
        all_tables_metadata = plan.get('all_tables_metadata', {self.table_name: self.table_metadata})
        
        for filter_item in plan['filters']:
            field = filter_item.get('field')
            values = filter_item.get('values', [])
            
            if not field or not values:
                continue
            
            # Handle qualified column names (table.column)
            if '.' in field:
                table_name, column_name = field.split('.', 1)
                table_metadata = all_tables_metadata.get(table_name)
            else:
                column_name = field
                table_name = self.table_name
                table_metadata = self.table_metadata
            
            if not table_metadata or column_name not in table_metadata['columns']:
                logger.warning(f"Field {field} not found in table metadata")
                continue
            
            actual_values = self._get_actual_column_values(column_name, table_name)
            
            if not actual_values:
                logger.warning(f"No actual values found for field {field}")
                continue
            
            corrected_values = []
            all_corrected_filter_values = []
            
            for value in values:
                # Normalize value for comparison
                value_str = str(value).strip()
                
                # Try exact match first (handles both string and numeric)
                if value_str in actual_values:
                    corrected_values.append({
                        'original': value,
                        'corrected': [value_str],
                        'match_type': 'exact',
                        'confidence': [1.0]
                    })
                    all_corrected_filter_values.append(value_str)
                    continue
                
                # For numeric values, try numeric comparison
                numeric_match_found = False
                try:
                    value_num = float(value_str)
                    for actual_val in actual_values:
                        try:
                            actual_num = float(actual_val)
                            if abs(value_num - actual_num) < 0.0001:  # Float comparison tolerance
                                corrected_values.append({
                                    'original': value,
                                    'corrected': [actual_val],
                                    'match_type': 'exact_numeric',
                                    'confidence': [1.0]
                                })
                                all_corrected_filter_values.append(actual_val)
                                numeric_match_found = True
                                logger.info(f"Numeric exact match: '{value}' -> '{actual_val}'")
                                break
                        except (ValueError, TypeError):
                            continue
                except (ValueError, TypeError):
                    pass  # Not a numeric value, proceed to fuzzy matching
                
                if numeric_match_found:
                    continue
                
                # Try fuzzy matching for text values
                match_result = self._find_closest_match(
                    value_str, 
                    actual_values,
                    threshold=0.6,
                    max_matches=3,
                    similarity_threshold=0.85
                )
                
                if match_result['matches']:
                    corrected_values.append({
                        'original': value,
                        'corrected': match_result['matches'],
                        'match_type': 'fuzzy',
                        'confidence': match_result['scores'],
                        'alternatives': match_result['all_matches'][:5]
                    })
                    all_corrected_filter_values.extend(match_result['matches'])
                    
                    logger.info(f"Fuzzy matches for '{value}' -> {match_result['matches']}")
                else:
                    corrected_values.append({
                        'original': value,
                        'corrected': [],
                        'match_type': 'none',
                        'confidence': [],
                        'suggestion': f"No match found. Available: {actual_values[:5]}"
                    })
                    logger.warning(f"No match found for '{value}' in field '{field}'")
            
            filter_item['validated_values'] = corrected_values
            filter_item['corrected_filter_values'] = list(set(all_corrected_filter_values))
            filter_item['needs_correction'] = any(v['match_type'] != 'exact' for v in corrected_values)
        
        return plan
    
    def _generate_sql(self, plan: Dict[str, Any]) -> Optional[str]:
        """
        Step 3: Generate SQL query from validated plan including JOINs
        """
        logger.info("Generating SQL query")
        
        # Check for join errors before generating SQL
        if 'join_errors' in plan and plan['join_errors']:
            logger.error(f"Cannot generate SQL due to join errors: {plan['join_errors']}")
            return None
        
        # Prepare schema information for all tables
        all_tables_metadata = plan.get('all_tables_metadata', {self.table_name: self.table_metadata})
        schema_info = {}
        for table, metadata in all_tables_metadata.items():
            schema_info[table] = {
                'columns': metadata['columns'],
                'types': metadata['column_types']
            }
        
        # Prepare join metadata
        join_metadata = plan.get('resolved_joins', [])
        
        corrected_values_info = ""
        if 'filters' in plan:
            for f in plan['filters']:
                if 'corrected_filter_values' in f:
                    corrected_values_info += f"\nField '{f['field']}': {f['corrected_filter_values']}"
        
        try:
            prompt = self.prompts['sql_generation'].format(
                table_name=self.table_name,
                schema=json.dumps(schema_info, indent=2),
                join_metadata=json.dumps(join_metadata, indent=2),
                query_plan=json.dumps(plan, indent=2),
                corrected_values=corrected_values_info
            )
        except KeyError as e:
            logger.error(f"Missing key in sql_generation prompt template: {e}")
            logger.error(f"Available keys: table_name, schema, join_metadata, query_plan, corrected_values")
            logger.error(f"Prompt template may have incorrect placeholders. Check prompts.yml")
            return None
        
        response = self.llm.generate(prompt)
        
        if not response:
            logger.error("Failed to generate SQL query")
            return None
        
        sql_query = response.strip()
        if sql_query.startswith('```sql'):
            sql_query = sql_query[6:]
        if sql_query.startswith('```'):
            sql_query = sql_query[3:]
        if sql_query.endswith('```'):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        logger.info(f"Generated SQL: {sql_query}")
        return sql_query
    
    def _execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            
            if sql_query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                formatted_results = [
                    dict(zip(columns, row)) for row in results
                ]
                
                conn.close()
                logger.info(f"Query executed successfully, returned {len(formatted_results)} rows")
                return True, formatted_results
            else:
                conn.commit()
                affected_rows = cursor.rowcount
                conn.close()
                logger.info(f"Update executed successfully, affected {affected_rows} rows")
                return True, {"affected_rows": affected_rows}
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return False, str(e)
    
    def _fix_query(self, original_query: str, error_message: str, attempt: int) -> Optional[str]:
        """Step 4: Fix failed SQL query"""
        logger.info(f"Attempting to fix query (attempt {attempt}/{self.max_retries})")
        
        sample_data = {}
        for col in self.table_metadata['columns'][:5]:
            sample_data[col] = self.table_metadata['sample_data'].get(col, [])
        
        prompt = self.prompts['query_fixer'].format(
            original_query=original_query,
            error_message=error_message,
            schema=json.dumps(self.table_metadata['column_types'], indent=2),
            sample_data=json.dumps(sample_data, indent=2)
        )
        
        response = self.llm.generate(prompt)
        
        if not response:
            logger.error("Failed to get fixed query from LLM")
            return None
        
        fixed_query = response.strip()
        if fixed_query.startswith('```sql'):
            fixed_query = fixed_query[6:]
        if fixed_query.startswith('```'):
            fixed_query = fixed_query[3:]
        if fixed_query.endswith('```'):
            fixed_query = fixed_query[:-3]
        fixed_query = fixed_query.strip()
        
        logger.info(f"Fixed SQL: {fixed_query}")
        return fixed_query
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main method to process natural language query and return results
        Supports both single table queries and JOIN queries
        """
        logger.info(f"Processing query: {user_query}")
        
        plan = self._understand_query(user_query)
        if not plan:
            return {
                'success': False,
                'error': 'Failed to understand query',
                'results': None
            }
        
        validated_plan = self._validate_and_fix_filter_values(plan)
        
        # Check for join errors before proceeding
        if 'join_errors' in validated_plan and validated_plan['join_errors']:
            return {
                'success': False,
                'error': 'JOIN configuration error',
                'join_errors': validated_plan['join_errors'],
                'plan': validated_plan,
                'results': None
            }
        
        sql_query = self._generate_sql(validated_plan)
        if not sql_query:
            return {
                'success': False,
                'error': 'Failed to generate SQL query',
                'results': None
            }
        
        success, results = self._execute_query(sql_query)
        
        if success:
            response_data = {
                'success': True,
                'sql_query': sql_query,
                'results': results,
                'plan': validated_plan,
                'joins_used': validated_plan.get('resolved_joins', [])
            }
            
            # Add join match warnings if any fuzzy matching occurred
            if 'join_match_warnings' in validated_plan:
                response_data['warnings'] = validated_plan['join_match_warnings']
            
            return response_data
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Query failed, attempting fix {attempt}/{self.max_retries}")
            
            fixed_query = self._fix_query(sql_query, results, attempt)
            if not fixed_query:
                continue
            
            success, results = self._execute_query(fixed_query)
            
            if success:
                return {
                    'success': True,
                    'sql_query': fixed_query,
                    'results': results,
                    'plan': validated_plan,
                    'joins_used': validated_plan.get('resolved_joins', []),
                    'fixed_after_attempts': attempt
                }
            
            sql_query = fixed_query
        
        return {
            'success': False,
            'error': f'Query failed after {self.max_retries} retry attempts',
            'last_error': results,
            'last_query': sql_query,
            'plan': validated_plan
        }