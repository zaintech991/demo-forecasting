# Supabase Database Setup Instructions

If you're encountering issues using the REST API to create the database schema and load data, follow these alternative steps to set up your database using the Supabase SQL Editor.

## 1. Generate the SQL Schema and Sample Data

First, generate the SQL schema and sample data by running:

```bash
python scripts/load_data.py --generate-sql-only --limit 20 > schema_with_sample_data.sql
```

This will create a file called `schema_with_sample_data.sql` with all the necessary SQL statements to:
- Create the database tables
- Create indexes
- Insert sample data

## 2. Execute the SQL in Supabase SQL Editor

1. Log in to your Supabase account and navigate to your project
2. Click on "SQL Editor" in the left sidebar
3. Create a new query
4. Open the generated `schema_with_sample_data.sql` file in a text editor
5. Copy the entire contents of the file
6. Paste the SQL into the Supabase SQL Editor
7. Click "Run" to execute the SQL

## 3. Verify the Tables and Data

After running the SQL, verify that the tables were created correctly:

1. Go to the "Table Editor" in the Supabase dashboard
2. Check that you can see the following tables:
   - `sales_data`
   - `promotions`
   - `holidays`
3. Click on each table to verify the structure and that sample data was loaded

## 4. Loading the Full Dataset

For loading the full dataset, you have a few options:

### Option 1: Load in batches using the script (if it works)

```bash
python scripts/load_data.py --limit 100
# Then gradually increase the limit
python scripts/load_data.py --limit 1000
python scripts/load_data.py --limit 10000
# etc.
```

### Option 2: Generate SQL and execute in batches via SQL Editor

Generate SQL for larger batches:

```bash
python scripts/load_data.py --generate-sql-only --limit 1000 > batch1.sql
python scripts/load_data.py --generate-sql-only --limit 1000 --offset 1000 > batch2.sql
# etc.
```

Then execute each batch file in the SQL Editor.

### Option 3: Use Supabase's native CSV import

1. Extract a portion of the dataset to CSV:
```bash
python scripts/export_dataset.py --limit 5000 --format csv --output data_export.csv
```

2. Import using Supabase Table Editor's "Import" function

## Troubleshooting

If you encounter any issues:

1. Check that the SQL statements are valid PostgreSQL syntax
2. Verify that you have sufficient permissions in your Supabase project
3. Try executing smaller batches of statements if you run into timeouts
4. For large datasets, consider using the Supabase CLI for direct database access

## Additional Resources

- [Supabase SQL Documentation](https://supabase.io/docs/guides/database/sql)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Supabase CLI Documentation](https://supabase.io/docs/reference/cli) 