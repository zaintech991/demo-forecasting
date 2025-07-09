# FreshRetailNet-50K Dataset CSV Exports

This directory contains CSV exports of the Hugging Face dataset "Dingdong-Inc/FreshRetailNet-50K" formatted for direct import to Supabase tables as defined in `supabase_schema.sql`.

## Files

1. **sales_data_csv.csv**: Main sales data with weather and other features
   - Contains 4,500,000 records from the original dataset
   - Columns match the `sales_data` table schema in Supabase
   - **Note:** This file is 306MB and exceeds Supabase's 100MB dashboard upload limit. See `sales_data_chunks/` directory for split files.

2. **promotion_csv.csv**: Promotion events data
   - Generated based on products with activity_flag=1 in the original dataset
   - Contains approximately 2,000 promotion records
   - Columns match the `promotions` table schema in Supabase

3. **holidays_csv.csv**: US holiday calendar data
   - Contains US holidays for the relevant dates in the dataset
   - Columns match the `holidays` table schema in Supabase

4. **sales_data_chunks/**: Directory containing split versions of sales_data_csv.csv
   - Each file is under 100MB to comply with Supabase's dashboard upload limit
   - Contains 4 files (sales_data_part_001.csv through sales_data_part_004.csv)
   - See sales_data_chunks/README.md for more information

## Importing to Supabase

### Option 1: Supabase UI

For files under 100MB (promotions_csv.csv, holidays_csv.csv, and individual files in sales_data_chunks/):

1. Go to your Supabase project dashboard
2. Navigate to the "Table Editor" section
3. Select the target table (e.g., sales_data, promotions, or holidays)
4. Click "Import" and select the corresponding CSV file
5. Follow the UI prompts to complete the import

### Option 2: PostgreSQL COPY Command

You can also use the PostgreSQL COPY command to import the data:

```sql
-- For sales_data table (use files from sales_data_chunks/ directory)
COPY sales_data (
    city_id, store_id, management_group_id, 
    first_category_id, second_category_id, third_category_id,
    product_id, dt, sale_amount, stock_hour6_22_cnt, 
    discount, holiday_flag, activity_flag, precpt, 
    avg_temperature, avg_humidity, avg_wind_level
) 
FROM '/path/to/sales_data_part_001.csv' 
DELIMITER ',' 
CSV HEADER;

-- For promotions table
COPY promotions (
    store_id, product_id, start_date, end_date,
    promotion_type, discount, location, campaign_id
) 
FROM '/path/to/promotion_csv.csv' 
DELIMITER ',' 
CSV HEADER;

-- For holidays table
COPY holidays (
    date, name, country
) 
FROM '/path/to/holidays_csv.csv' 
DELIMITER ',' 
CSV HEADER;
```

### Option 3: Using Python Scripts

We've provided two Python scripts to help with large file uploads:

1. **database/split_csv.py**: Splits large CSV files into smaller chunks
   - Already used to create the files in the `sales_data_chunks/` directory
   - Can be modified to change chunk size or split other large files

2. **database/upload_chunks.py**: Uploads split CSV files directly to Supabase using the API
   - Run this script and follow the prompts to upload all chunk files
   - Uses the same Supabase connection as other scripts in the project

To use the upload script:
```bash
python database/upload_chunks.py
```

### Option 4: Supabase REST API

You can also use the Supabase JavaScript/TypeScript client to upload the data:

```javascript
// Example using the Supabase JavaScript client
import { createClient } from '@supabase/supabase-js';
import { readFileSync } from 'fs';
import csv from 'csv-parser';
import { createReadStream } from 'fs';

const supabase = createClient(
  'https://your-project-url.supabase.co',
  'your-supabase-key'
);

// Function to import a CSV file to a Supabase table
async function importCsv(filePath, tableName) {
  const results = [];
  
  // Parse CSV file
  await new Promise((resolve) => {
    createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', resolve);
  });
  
  // Upload data in batches of 1000 records
  const batchSize = 1000;
  for (let i = 0; i < results.length; i += batchSize) {
    const batch = results.slice(i, i + batchSize);
    const { error } = await supabase.from(tableName).insert(batch);
    
    if (error) {
      console.error(`Error uploading batch ${i/batchSize + 1}:`, error);
    } else {
      console.log(`Uploaded batch ${i/batchSize + 1}/${Math.ceil(results.length/batchSize)}`);
    }
  }
}

// Import each CSV file
importCsv('./sales_data_csv.csv', 'sales_data');
importCsv('./promotion_csv.csv', 'promotions');
importCsv('./holidays_csv.csv', 'holidays');
```

## Data Structure

### sales_data_csv.csv

Column | Description
------ | -----------
city_id | City identifier
store_id | Store identifier
management_group_id | Management group identifier
first_category_id | First level category identifier
second_category_id | Second level category identifier
third_category_id | Third level category identifier
product_id | Product identifier
dt | Sale date (YYYY-MM-DD)
sale_amount | Amount of sales
stock_hour6_22_cnt | Stock count between 6:00-22:00
discount | Discount rate applied
holiday_flag | Whether it's a holiday (1=yes, 0=no)
activity_flag | Whether a promotion is active (1=yes, 0=no)
precpt | Precipitation level
avg_temperature | Average temperature
avg_humidity | Average humidity
avg_wind_level | Average wind level

### promotion_csv.csv

Column | Description
------ | -----------
store_id | Store identifier
product_id | Product identifier
start_date | Promotion start date (YYYY-MM-DD)
end_date | Promotion end date (YYYY-MM-DD)
promotion_type | Type of promotion (Discount, BOGO, etc.)
discount | Discount amount (0.0-1.0)
location | Display location in store
campaign_id | Campaign identifier

### holidays_csv.csv

Column | Description
------ | -----------
date | Holiday date (YYYY-MM-DD)
name | Holiday name
country | Country (USA) 