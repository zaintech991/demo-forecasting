-- SQL script to load all CSV chunks into the sales_data table

-- Load data from sales_data_part_001.csv
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

-- Load data from sales_data_part_002.csv
COPY sales_data (
    city_id, store_id, management_group_id,
    first_category_id, second_category_id, third_category_id,
    product_id, dt, sale_amount, stock_hour6_22_cnt,
    discount, holiday_flag, activity_flag, precpt,
    avg_temperature, avg_humidity, avg_wind_level
)
FROM '/path/to/sales_data_part_002.csv'
DELIMITER ','
CSV HEADER;

-- Load data from sales_data_part_003.csv
COPY sales_data (
    city_id, store_id, management_group_id,
    first_category_id, second_category_id, third_category_id,
    product_id, dt, sale_amount, stock_hour6_22_cnt,
    discount, holiday_flag, activity_flag, precpt,
    avg_temperature, avg_humidity, avg_wind_level
)
FROM '/path/to/sales_data_part_003.csv'
DELIMITER ','
CSV HEADER;

-- Load data from sales_data_part_004.csv
COPY sales_data (
    city_id, store_id, management_group_id,
    first_category_id, second_category_id, third_category_id,
    product_id, dt, sale_amount, stock_hour6_22_cnt,
    discount, holiday_flag, activity_flag, precpt,
    avg_temperature, avg_humidity, avg_wind_level
)
FROM '/path/to/sales_data_part_004.csv'
DELIMITER ','
CSV HEADER;

