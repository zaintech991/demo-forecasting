"""
Database connector utility for Supabase.
"""
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from supabase import create_client
import requests
import json

class SupabaseConnector:
    """
    Connector class for Supabase operations.
    Uses environment variables for configuration.
    """
    
    def __init__(self):
        """
        Initialize the Supabase connector with credentials from environment variables.
        """
        load_dotenv()  # Load environment variables from .env file
        
        # Get credentials from environment
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in .env file.")
        
        # Initialize client
        self.client = create_client(self.supabase_url, self.supabase_key)
    
    def execute_sql_directly(self, query: str) -> bool:
        """
        Execute a SQL statement directly using the Supabase REST API with service role.
        This is specifically for DDL statements that can't be executed through RPC.
        
        Args:
            query: SQL query to execute (CREATE TABLE, etc.)
            
        Returns:
            Boolean indicating success
        """
        if not self.supabase_service_key:
            print("Warning: SUPABASE_SERVICE_KEY not set, using regular key which may have limited permissions")
            auth_token = self.supabase_key
        else:
            auth_token = self.supabase_service_key
            
        # Construct the SQL API endpoint
        endpoint = f"{self.supabase_url}/rest/v1/rpc/exec_sql"
        
        # Set up headers with the service role key for admin access
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        # Create the payload
        payload = {"query_text": query}
        
        try:
            # Execute the query
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            
            # Check if successful
            if response.status_code < 300:
                return True
            else:
                print(f"SQL execution error: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"Error executing SQL directly: {e}")
            return False
        
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query on Supabase database.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            List of query results as dictionaries
        """
        # For DDL statements like CREATE TABLE, try direct execution
        if query.strip().lower().startswith(("create ", "drop ", "alter ")):
            if self.execute_sql_directly(query):
                return []
        
        if params is None:
            params = {}
            
        try:
            # Use RPC to execute the query
            response = self.client.rpc("exec_sql", {"query_text": query}).execute()
            
            # Check for errors in the new API format
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Supabase query error: {response.error}")
            
            # Return the data
            return response.data if hasattr(response, 'data') else []
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def select(self, table: str, columns: str = "*", filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Select data from a table.
        
        Args:
            table: Table name
            columns: Columns to select
            filters: Optional filters as dict
            
        Returns:
            List of results as dictionaries
        """
        query = self.client.table(table).select(columns)
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        # Execute query
        try:
            response = query.execute()
            return response.data if hasattr(response, 'data') else []
        except Exception as e:
            print(f"Supabase select error: {e}")
            return []
    
    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert data into a table.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Inserted record
        """
        try:
            response = self.client.table(table).insert(data).execute()
            return response.data[0] if hasattr(response, 'data') and response.data else {}
        except Exception as e:
            print(f"Supabase insert error: {e}")
            return {}
    
    def update(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Update records in a table.
        
        Args:
            table: Table name
            data: Data to update
            filters: Filters to apply
            
        Returns:
            Updated records
        """
        query = self.client.table(table).update(data)
        
        # Apply filters
        for key, value in filters.items():
            query = query.eq(key, value)
        
        # Execute query
        try:
            response = query.execute()
            return response.data if hasattr(response, 'data') else []
        except Exception as e:
            print(f"Supabase update error: {e}")
            return []
    
    def delete(self, table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Delete records from a table.
        
        Args:
            table: Table name
            filters: Filters to apply
            
        Returns:
            Deleted records
        """
        query = self.client.table(table).delete()
        
        # Apply filters
        for key, value in filters.items():
            query = query.eq(key, value)
        
        # Execute query
        try:
            response = query.execute()
            return response.data if hasattr(response, 'data') else []
        except Exception as e:
            print(f"Supabase delete error: {e}")
            return [] 