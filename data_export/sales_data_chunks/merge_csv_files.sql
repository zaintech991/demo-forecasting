-- SQL script to load all CSV chunks into the sales_data table

-- Load data from sales_data_part_001.csv
\copy sales_data (city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id, dt, sale_amount, stock_hour6_22_cnt, discount, holiday_flag, activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level) FROM '/Users/boolmind_mac/Downloads/boolmind-demo-forcasting-741311c2faf6/data_export/sales_data_chunks/sales_data_part_001.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Load data from sales_data_part_002.csv
\copy sales_data (city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id, dt, sale_amount, stock_hour6_22_cnt, discount, holiday_flag, activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level) FROM '/Users/boolmind_mac/Downloads/boolmind-demo-forcasting-741311c2faf6/data_export/sales_data_chunks/sales_data_part_002.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Load data from sales_data_part_003.csv
\copy sales_data (city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id, dt, sale_amount, stock_hour6_22_cnt, discount, holiday_flag, activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level) FROM '/Users/boolmind_mac/Downloads/boolmind-demo-forcasting-741311c2faf6/data_export/sales_data_chunks/sales_data_part_003.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Load data from sales_data_part_004.csv
\copy sales_data (city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id, dt, sale_amount, stock_hour6_22_cnt, discount, holiday_flag, activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level) FROM '/Users/boolmind_mac/Downloads/boolmind-demo-forcasting-741311c2faf6/data_export/sales_data_chunks/sales_data_part_004.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

