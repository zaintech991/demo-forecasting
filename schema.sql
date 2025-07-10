-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables
DROP TABLE IF EXISTS sales_data CASCADE;
DROP TABLE IF EXISTS store_hierarchy CASCADE;
DROP TABLE IF EXISTS product_hierarchy CASCADE;
DROP TABLE IF EXISTS promotions CASCADE;
DROP TABLE IF EXISTS holidays CASCADE;

-- Create sales_data table with structure matching CSV files
CREATE TABLE sales_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    city_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    dt DATE NOT NULL,
    sale_amount FLOAT NOT NULL,
    stock_hour6_22_cnt INTEGER,
    discount FLOAT,
    holiday_flag BOOLEAN,
    activity_flag BOOLEAN,
    precpt FLOAT,
    avg_temperature FLOAT,
    avg_humidity FLOAT,
    avg_wind_level FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create store_hierarchy table
CREATE TABLE store_hierarchy (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    city_id INTEGER NOT NULL,
    format_type VARCHAR(50) NOT NULL,
    size_type VARCHAR(50) NOT NULL
);

-- Create product_hierarchy table
CREATE TABLE product_hierarchy (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    management_group_id INTEGER NOT NULL DEFAULT 0
);

-- Create promotions table
CREATE TABLE promotions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    store_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    promotion_type VARCHAR(50) NOT NULL,
    discount FLOAT NOT NULL,
    location VARCHAR(50),
    campaign_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create holidays table
CREATE TABLE holidays (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    date DATE NOT NULL,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(50) DEFAULT 'USA',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, name)
);

-- Create indexes
CREATE INDEX idx_sales_store_product ON sales_data (store_id, product_id);
CREATE INDEX idx_sales_date ON sales_data (dt);
CREATE INDEX idx_sales_store ON sales_data (store_id);
CREATE INDEX idx_sales_product ON sales_data (product_id);
CREATE INDEX idx_sales_city ON sales_data (city_id);
CREATE INDEX idx_sales_category ON sales_data (first_category_id, second_category_id, third_category_id);

-- Insert sample store data
INSERT INTO store_hierarchy (store_id, store_name, city_id, format_type, size_type) VALUES
(0, 'Downtown Market', 0, 'Supermarket', 'Large'),
(1, 'Neighborhood Fresh', 0, 'Convenience', 'Small'),
(2, 'Central Grocers', 0, 'Supermarket', 'Medium');

-- Insert sample product data
INSERT INTO product_hierarchy (product_id, product_name, first_category_id, second_category_id, third_category_id) VALUES
(21, 'Premium Coffee', 1, 2, 3),
(38, 'Fresh Tomatoes', 0, 0, 1),
(70, 'Fresh Lettuce', 0, 0, 1);

-- Insert sample holiday data
INSERT INTO holidays (date, name, country)
VALUES 
('2024-03-01', 'Employee Appreciation Day', 'USA'),
('2024-03-17', 'St. Patrick''s Day', 'USA'),
('2024-03-31', 'Easter', 'USA')
ON CONFLICT (date, name) DO NOTHING; 