"""
Database connection utilities for Supabase and PostgreSQL.
"""
import os
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from supabase.client import create_client, Client
from typing import Optional

# Load environment variables
load_dotenv()

def get_supabase_client() -> Optional[Client]:
    """
    Create and return a Supabase client.
    
    Returns:
        Optional[Client]: Supabase client or None if connection fails
    """
    # Check for environment variables, with fallback values
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("Warning: SUPABASE_URL or SUPABASE_KEY environment variables not set.")
        print("Using default Supabase credentials from configuration.")
        # Default fallback values if not in environment
        supabase_url = 'https://vcoshhbfaymqyqjsytcw.supabase.co'
        supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZjb3NoaGJmYXltcXlxanN5dGN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MTE2NTMsImV4cCI6MjA2NzQ4NzY1M30.PUqs39Cfd2hh2_Yu4-ReX5pI9em2jYlE3TrTlV_7IA0'
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Error creating Supabase client: {str(e)}")
        print("Please check your Supabase credentials and network connection.")
        return None

def get_db_engine(max_retries=3, retry_delay=2):
    """
    Create and return a SQLAlchemy engine for connecting to Supabase PostgreSQL.
    Includes retry logic for connection issues.
    
    Args:
        max_retries: Maximum number of connection retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Engine: SQLAlchemy engine or None if connection fails
    """
    # Database connection parameters with fallbacks
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'postgres')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD')
    
    if not db_host or not db_password:
        print("Warning: DB_HOST or DB_PASSWORD environment variables not set.")
        print("Using default database credentials from configuration.")
        # Default fallback values
        db_host = 'db.vcoshhbfaymqyqjsytcw.supabase.co'
        db_password = 'boolmind'
    
    # Create connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Retry logic for connection
    for attempt in range(max_retries):
        try:
            # Create engine with appropriate timeout
            engine = create_engine(
                connection_string,
                connect_args={"connect_timeout": 10},
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("Database engine created successfully")
            return engine
            
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("All connection attempts failed.")
                print("Please check your database credentials, network connection, and ensure the database is accessible.")
                return None  # Return None instead of raising to handle error at the calling point

def test_connection():
    """
    Test database connection.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    connection_success = False
    
    try:
        # Test Supabase connection
        supabase = get_supabase_client()
        if supabase:
            try:
                response = supabase.table('_dummy').select('*').limit(1).execute()
                print("Supabase connection successful")
            except Exception as e:
                print(f"Supabase query error: {str(e)}")
                print("Supabase connection established but query failed. This may be normal if the _dummy table doesn't exist.")
        
        # Test PostgreSQL connection
        engine = get_db_engine()
        if engine:  # Check if engine is not None
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                    print("PostgreSQL connection successful")
                connection_success = True
            except Exception as e:
                print(f"PostgreSQL connection error: {str(e)}")
        else:
            print("Could not create database engine")
            
    except Exception as e:
        print(f"Connection error: {str(e)}")
    
    return connection_success

if __name__ == "__main__":
    test_connection()
