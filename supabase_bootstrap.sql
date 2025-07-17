-- =============================================================================
-- SUPABASE RETAIL FORECASTING BOOTSTRAP SCRIPT
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Drop tables in dependency order
DROP TABLE IF EXISTS sales_data CASCADE;
DROP TABLE IF EXISTS hourly_sales_data CASCADE;
DROP TABLE IF EXISTS stockout_events CASCADE;
DROP TABLE IF EXISTS product_correlations CASCADE;
DROP TABLE IF EXISTS store_performance_metrics CASCADE;
DROP TABLE IF EXISTS weather_impact_analysis CASCADE;
DROP TABLE IF EXISTS promotion_effectiveness CASCADE;
DROP TABLE IF EXISTS category_performance CASCADE;
DROP TABLE IF EXISTS store_clusters CASCADE;
DROP TABLE IF EXISTS demand_forecasts CASCADE;
DROP TABLE IF EXISTS store_hierarchy CASCADE;
DROP TABLE IF EXISTS product_hierarchy CASCADE;
DROP TABLE IF EXISTS city_hierarchy CASCADE;
DROP TABLE IF EXISTS promotions CASCADE;
DROP TABLE IF EXISTS holidays CASCADE;

-- =============================================================================
-- CORE HIERARCHY TABLES
-- =============================================================================

-- City hierarchy - 18 major US cities
CREATE TABLE city_hierarchy (
    city_id INTEGER PRIMARY KEY,
    city_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    timezone VARCHAR(50),
    population INTEGER,
    economic_index DECIMAL(5,2),
    climate_zone VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Store hierarchy - 25 stores across 18 cities
CREATE TABLE store_hierarchy (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    city_id INTEGER NOT NULL REFERENCES city_hierarchy(city_id),
    management_group_id INTEGER NOT NULL,
    store_format VARCHAR(50) NOT NULL,
    store_size_sqm INTEGER,
    location_type VARCHAR(50),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    opening_hours JSONB,
    employee_count INTEGER,
    parking_spaces INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Product hierarchy - 250 real products
CREATE TABLE product_hierarchy (
    product_id INTEGER PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    is_perishable BOOLEAN DEFAULT TRUE,
    shelf_life_days INTEGER,
    storage_temperature_min DECIMAL(5,2),
    storage_temperature_max DECIMAL(5,2),
    unit_cost DECIMAL(10,2),
    standard_price DECIMAL(10,2),
    profit_margin DECIMAL(5,4),
    supplier_id INTEGER,
    seasonality_pattern VARCHAR(50),
    weather_sensitivity_score DECIMAL(3,2),
    promotion_elasticity DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Main sales data table
CREATE TABLE sales_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES city_hierarchy(city_id),
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    sale_date DATE NOT NULL,
    sale_amount DECIMAL(12,4) NOT NULL,
    hours_sale DECIMAL[] NOT NULL,
    stock_hour6_22_cnt INTEGER DEFAULT 0,
    hours_stock_status INTEGER[] NOT NULL,
    discount DECIMAL(5,4) DEFAULT 1.0,
    activity_flag INTEGER DEFAULT 0,
    holiday_flag INTEGER DEFAULT 0,
    precpt DECIMAL(8,2),
    avg_temperature DECIMAL(5,2),
    avg_humidity DECIMAL(5,2),
    avg_wind_level DECIMAL(5,2),
    units_sold INTEGER,
    average_unit_price DECIMAL(10,2),
    stockout_duration_minutes INTEGER DEFAULT 0,
    peak_hour INTEGER,
    demand_volatility DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- (Other tables, indexes, and views would follow here, omitted for brevity)

-- =============================================================================
-- REFERENCE DATA: 18 US Cities
-- =============================================================================
INSERT INTO city_hierarchy (city_id, city_name, region, timezone, population, economic_index, climate_zone) VALUES
(1, 'New York', 'Northeast', 'America/New_York', 8419000, 98.5, 'humid_subtropical'),
(2, 'Los Angeles', 'West', 'America/Los_Angeles', 3980000, 95.2, 'mediterranean'),
(3, 'Chicago', 'Midwest', 'America/Chicago', 2716000, 92.1, 'humid_continental'),
(4, 'Houston', 'South', 'America/Chicago', 2328000, 89.7, 'humid_subtropical'),
(5, 'Phoenix', 'West', 'America/Phoenix', 1690000, 87.3, 'desert'),
(6, 'Philadelphia', 'Northeast', 'America/New_York', 1584000, 85.9, 'humid_subtropical'),
(7, 'San Antonio', 'South', 'America/Chicago', 1547000, 84.2, 'humid_subtropical'),
(8, 'San Diego', 'West', 'America/Los_Angeles', 1424000, 83.7, 'mediterranean'),
(9, 'Dallas', 'South', 'America/Chicago', 1341000, 82.5, 'humid_subtropical'),
(10, 'San Jose', 'West', 'America/Los_Angeles', 1035000, 81.3, 'mediterranean'),
(11, 'Austin', 'South', 'America/Chicago', 995000, 80.1, 'humid_subtropical'),
(12, 'Jacksonville', 'South', 'America/New_York', 911000, 78.9, 'humid_subtropical'),
(13, 'Fort Worth', 'South', 'America/Chicago', 902000, 77.6, 'humid_subtropical'),
(14, 'Columbus', 'Midwest', 'America/New_York', 898000, 76.4, 'humid_continental'),
(15, 'Charlotte', 'South', 'America/New_York', 885000, 75.2, 'humid_subtropical'),
(16, 'San Francisco', 'West', 'America/Los_Angeles', 883000, 74.0, 'mediterranean'),
(17, 'Indianapolis', 'Midwest', 'America/Indiana/Indianapolis', 876000, 72.8, 'humid_continental'),
(18, 'Seattle', 'West', 'America/Los_Angeles', 744000, 71.6, 'oceanic');

-- =============================================================================
-- REFERENCE DATA: 25 US Stores (sample, real chains, distributed across cities)
-- =============================================================================
INSERT INTO store_hierarchy (store_id, store_name, city_id, management_group_id, store_format, store_size_sqm, location_type, latitude, longitude, opening_hours, employee_count, parking_spaces) VALUES
(1, 'Walmart Supercenter Manhattan', 1, 1, 'supermarket', 12000, 'urban', 40.753685, -73.999163, '{"mon-sun": "07:00-23:00"}', 200, 150),
(2, 'Target LA Downtown', 2, 2, 'supermarket', 9000, 'urban', 34.040713, -118.246769, '{"mon-sun": "08:00-22:00"}', 120, 100),
(3, 'Kroger Chicago Central', 3, 3, 'supermarket', 8000, 'urban', 41.878113, -87.629799, '{"mon-sun": "07:00-22:00"}', 100, 80),
(4, 'Costco Houston', 4, 4, 'hypermarket', 15000, 'suburban', 29.760427, -95.369803, '{"mon-sun": "09:00-21:00"}', 180, 200),
(5, 'Safeway Phoenix', 5, 5, 'supermarket', 7000, 'urban', 33.448376, -112.074036, '{"mon-sun": "07:00-23:00"}', 90, 60),
(6, 'Publix Philadelphia', 6, 6, 'supermarket', 6000, 'urban', 39.952583, -75.165222, '{"mon-sun": "07:00-22:00"}', 80, 50),
(7, 'H-E-B San Antonio', 7, 7, 'supermarket', 8500, 'urban', 29.424122, -98.493629, '{"mon-sun": "07:00-23:00"}', 110, 70),
(8, 'Albertsons San Diego', 8, 8, 'supermarket', 6500, 'urban', 32.715736, -117.161087, '{"mon-sun": "07:00-22:00"}', 70, 40),
(9, 'Tom Thumb Dallas', 9, 9, 'supermarket', 6000, 'urban', 32.776664, -96.796988, '{"mon-sun": "07:00-23:00"}', 60, 30),
(10, 'Whole Foods San Jose', 10, 10, 'supermarket', 5500, 'urban', 37.338208, -121.886329, '{"mon-sun": "08:00-22:00"}', 50, 25),
(11, 'Randalls Austin', 11, 11, 'supermarket', 5000, 'urban', 30.267153, -97.743057, '{"mon-sun": "07:00-23:00"}', 45, 20),
(12, 'Winn-Dixie Jacksonville', 12, 12, 'supermarket', 4800, 'urban', 30.332184, -81.655647, '{"mon-sun": "07:00-22:00"}', 40, 15),
(13, 'Albertsons Fort Worth', 13, 13, 'supermarket', 4700, 'urban', 32.755488, -97.330766, '{"mon-sun": "07:00-23:00"}', 38, 12),
(14, 'Meijer Columbus', 14, 14, 'supermarket', 4600, 'urban', 39.961176, -82.998794, '{"mon-sun": "07:00-22:00"}', 36, 10),
(15, 'Harris Teeter Charlotte', 15, 15, 'supermarket', 4500, 'urban', 35.227087, -80.843127, '{"mon-sun": "07:00-23:00"}', 34, 8),
(16, 'Trader Joe''s San Francisco', 16, 16, 'supermarket', 4400, 'urban', 37.774929, -122.419416, '{"mon-sun": "08:00-21:00"}', 32, 6),
(17, 'Kroger Indianapolis', 17, 17, 'supermarket', 4300, 'urban', 39.768403, -86.158068, '{"mon-sun": "07:00-23:00"}', 30, 5),
(18, 'QFC Seattle', 18, 18, 'supermarket', 4200, 'urban', 47.606209, -122.332071, '{"mon-sun": "07:00-23:00"}', 28, 4),
(19, 'Costco Brooklyn', 1, 19, 'hypermarket', 14000, 'urban', 40.678178, -73.944158, '{"mon-sun": "09:00-21:00"}', 170, 180),
(20, 'Target Chicago North', 3, 20, 'supermarket', 8500, 'urban', 41.881832, -87.623177, '{"mon-sun": "08:00-22:00"}', 95, 70),
(21, 'Walmart Houston West', 4, 21, 'supermarket', 11000, 'suburban', 29.749907, -95.358421, '{"mon-sun": "07:00-23:00"}', 160, 120),
(22, 'Safeway Phoenix North', 5, 22, 'supermarket', 7200, 'urban', 33.448377, -112.074037, '{"mon-sun": "07:00-23:00"}', 92, 62),
(23, 'Publix Philly North', 6, 23, 'supermarket', 6100, 'urban', 39.952584, -75.165223, '{"mon-sun": "07:00-22:00"}', 82, 52),
(24, 'H-E-B San Antonio North', 7, 24, 'supermarket', 8600, 'urban', 29.424123, -98.493630, '{"mon-sun": "07:00-23:00"}', 112, 72),
(25, 'Whole Foods Seattle', 18, 25, 'supermarket', 5300, 'urban', 47.609722, -122.333056, '{"mon-sun": "08:00-22:00"}', 52, 28);

-- =============================================================================
-- REFERENCE DATA: 250 Products (sample, real product names)
-- =============================================================================
-- (For brevity, only a few sample products are shown. The full script will include 250.)
INSERT INTO product_hierarchy (product_id, sku, product_name, management_group_id, first_category_id, second_category_id, third_category_id, is_perishable, shelf_life_days, storage_temperature_min, storage_temperature_max, unit_cost, standard_price, profit_margin, supplier_id, seasonality_pattern, weather_sensitivity_score, promotion_elasticity) VALUES
(1, '0001', 'Milk, Whole, 1 Gallon', 1, 1, 1, 1, TRUE, 7, 1.0, 4.0, 2.50, 3.99, 0.3750, 1, 'stable', 0.2, 0.15),
(2, '0002', 'Eggs, Large, Dozen', 1, 1, 1, 2, TRUE, 21, 2.0, 8.0, 1.20, 2.49, 0.5181, 2, 'stable', 0.1, 0.10),
(3, '0003', 'Bread, White, Loaf', 1, 1, 2, 1, TRUE, 5, 0.0, 25.0, 1.00, 1.99, 0.4975, 3, 'high_summer', 0.05, 0.20),
(4, '0004', 'Bananas, 1 lb', 1, 1, 3, 1, TRUE, 5, 12.0, 20.0, 0.40, 0.69, 0.4203, 4, 'high_summer', 0.3, 0.25),
(5, '0005', 'Chicken Breast, 1 lb', 1, 2, 1, 1, TRUE, 10, -2.0, 4.0, 2.00, 3.49, 0.4270, 5, 'stable', 0.15, 0.18),
-- ... (245 more rows for products 6-250)
(250, '0250', 'Coca-Cola, 12 Pack', 1, 3, 2, 5, FALSE, NULL, NULL, NULL, 3.00, 5.99, 0.4992, 10, 'summer_peak', 0.4, 0.30);

-- =============================================================================
-- SAMPLE SALES DATA (for one date, all products, all stores)
-- =============================================================================
-- (For brevity, only a few sample rows are shown. The full script will include all combinations.)
INSERT INTO sales_data (city_id, store_id, product_id, management_group_id, first_category_id, second_category_id, third_category_id, sale_date, sale_amount, hours_sale, stock_hour6_22_cnt, hours_stock_status, discount, activity_flag, holiday_flag, precpt, avg_temperature, avg_humidity, avg_wind_level, units_sold, average_unit_price, stockout_duration_minutes, peak_hour, demand_volatility)
VALUES
(1, 1, 1, 1, 1, 1, 1, '2024-06-01', 120.00, ARRAY[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 0, ARRAY[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 1.0, 0, 0, 0.0, 22.0, 60.0, 2.0, 30, 4.00, 0, 12, 0.05),
(1, 1, 2, 1, 1, 1, 2, '2024-06-01', 80.00, ARRAY[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], 0, ARRAY[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 1.0, 0, 0, 0.0, 22.0, 60.0, 2.0, 24, 3.33, 0, 10, 0.04),
-- ... (more rows for all products and stores)
(18, 25, 250, 1, 3, 2, 5, '2024-06-01', 200.00, ARRAY[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], 0, ARRAY[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 0.9, 1, 0, 0.0, 25.0, 55.0, 1.5, 40, 5.00, 0, 15, 0.06);
-- (Repeat for all 250 products x 25 stores x 18 cities as needed)

-- =============================================================================
-- (Add other table definitions, indexes, and views as needed from your schema)
-- ============================================================================= 