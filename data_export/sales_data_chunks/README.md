# Sales Data CSV Chunks

This directory contains the sales_data_csv.csv file split into smaller chunks, each under 100MB, to comply with Supabase's dashboard upload limit.

## Files

1. **sales_data_part_001.csv**: ~79MB, contains rows 1-1,175,402
2. **sales_data_part_002.csv**: ~80MB, contains rows 1,175,403-2,350,804
3. **sales_data_part_003.csv**: ~80MB, contains rows 2,350,805-3,526,206
4. **sales_data_part_004.csv**: ~67MB, contains rows 3,526,207-4,500,000

## Import Options

### Option 1: Supabase Dashboard

Since each file is under 100MB, you can now import them individually through the Supabase dashboard:

1. Go to your Supabase project dashboard
2. Navigate to the "Table Editor" section
3. Select the `sales_data` table
4. Click "Import" and select each CSV file one by one
5. Follow the UI prompts to complete each import

### Option 2: Direct Database Import (Recommended for Large Files)

For more efficient importing, you can use the PostgreSQL COPY command directly on the database. 

1. Upload the CSV files to a location accessible by your Supabase PostgreSQL instance
2. Modify the `merge_csv_files.sql` script to point to the correct file paths
3. Run the SQL script to import all files

#### Example of Using the SQL Script:

```sql
-- Connect to your Supabase PostgreSQL database
-- Replace with your actual connection details
psql postgres://postgres:your-password@db.your-project-ref.supabase.co:5432/postgres

-- Then run the commands in merge_csv_files.sql
```

#### Accessing PostgreSQL on Supabase:

1. Go to your Supabase dashboard
2. Click on "Settings" (gear icon) in the sidebar
3. Select "Database"
4. Under "Connection Info", you'll find connection string information
5. You can also enable "Direct Database Access" for more connection options

## Notes

- All CSV files have the same header row and column structure
- Each file can be imported independently and will append to the table
- The SQL script provided handles importing all files in sequence
- Make sure to update the file paths in the SQL script to match your environment 