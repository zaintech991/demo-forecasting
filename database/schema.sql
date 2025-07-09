-- Drop existing objects to reset
DROP TABLE IF EXISTS sales_data CASCADE;
DROP TABLE IF EXISTS store_hierarchy CASCADE;
DROP TABLE IF EXISTS product_hierarchy CASCADE;
DROP TABLE IF EXISTS weather_data CASCADE;
DROP TABLE IF EXISTS holiday_calendar CASCADE;
DROP TABLE IF EXISTS promotion_events CASCADE;
DROP TABLE IF EXISTS stockout_periods CASCADE;
DROP TABLE IF EXISTS store_clusters CASCADE;
DROP TABLE IF EXISTS product_lifecycle CASCADE;

-- Create store hierarchy table
CREATE TABLE IF NOT EXISTS store_hierarchy (
    store_id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL,
    store_type VARCHAR(50),
    store_name VARCHAR(100),
    region VARCHAR(50),
    latitude FLOAT,
    longitude FLOAT,
    opening_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create product hierarchy table
CREATE TABLE IF NOT EXISTS product_hierarchy (
    product_id INTEGER PRIMARY KEY,
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    product_name VARCHAR(200),
    is_fresh BOOLEAN,
    shelf_life_days INTEGER,
    unit_cost FLOAT,
    unit_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create main sales data table with features
CREATE TABLE IF NOT EXISTS sales_data (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    sale_date DATE NOT NULL,
    sale_amount FLOAT NOT NULL,
    sale_qty INTEGER,
    discount FLOAT,
    original_price FLOAT,
    stock_hour6_22_cnt INTEGER,
    stock_hour6_14_cnt INTEGER,
    stock_hour14_22_cnt INTEGER,
    holiday_flag BOOLEAN,
    promo_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for hourly sales data
CREATE TABLE IF NOT EXISTS hourly_sales_data (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    sale_datetime TIMESTAMP NOT NULL,
    sale_amount FLOAT NOT NULL,
    sale_qty INTEGER,
    discount FLOAT,
    stock_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create weather data table
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    city_id INTEGER NOT NULL,
    date DATE NOT NULL,
    temp_min FLOAT,
    temp_max FLOAT,
    temp_avg FLOAT,
    precipitation FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    weather_condition VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create holiday calendar table
CREATE TABLE IF NOT EXISTS holiday_calendar (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    holiday_name VARCHAR(100),
    holiday_type VARCHAR(50),
    country VARCHAR(50),
    region VARCHAR(50),
    significance INTEGER CHECK (significance BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create promotion events table
CREATE TABLE IF NOT EXISTS promotion_events (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES store_hierarchy(store_id),
    product_id INTEGER REFERENCES product_hierarchy(product_id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    promotion_type VARCHAR(100),
    discount_percentage FLOAT,
    display_location VARCHAR(50),
    campaign_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create stockout periods table for stockout tracking
CREATE TABLE IF NOT EXISTS stockout_periods (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    start_datetime TIMESTAMP NOT NULL,
    end_datetime TIMESTAMP,
    estimated_lost_sales FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create store clusters table
CREATE TABLE IF NOT EXISTS store_clusters (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    cluster_id INTEGER NOT NULL,
    cluster_name VARCHAR(100),
    cluster_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create product lifecycle table
CREATE TABLE IF NOT EXISTS product_lifecycle (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    lifecycle_stage VARCHAR(50),
    introduction_date DATE,
    growth_start_date DATE,
    maturity_start_date DATE,
    decline_start_date DATE,
    discontinuation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization
CREATE INDEX idx_sales_data_store_product ON sales_data(store_id, product_id);
CREATE INDEX idx_sales_data_date ON sales_data(sale_date);
CREATE INDEX idx_hourly_sales_datetime ON hourly_sales_data(sale_datetime);
CREATE INDEX idx_weather_data_date ON weather_data(date);
CREATE INDEX idx_holiday_date ON holiday_calendar(date);
CREATE INDEX idx_promotion_dates ON promotion_events(start_date, end_date);
CREATE INDEX idx_stockout_dates ON stockout_periods(start_datetime, end_datetime);

-- Create materialized view for daily sales aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_sales AS
SELECT
    sd.sale_date,
    sd.store_id,
    sh.city_id,
    sd.product_id,
    ph.first_category_id,
    ph.second_category_id,
    ph.third_category_id,
    SUM(sd.sale_amount) as total_sales,
    SUM(sd.sale_qty) as total_quantity,
    AVG(sd.discount) as avg_discount,
    COUNT(*) as transaction_count,
    BOOL_OR(sd.holiday_flag) as had_holiday,
    BOOL_OR(sd.promo_flag) as had_promo
FROM sales_data sd
JOIN store_hierarchy sh ON sd.store_id = sh.store_id
JOIN product_hierarchy ph ON sd.product_id = ph.product_id
GROUP BY 
    sd.sale_date,
    sd.store_id,
    sh.city_id,
    sd.product_id,
    ph.first_category_id,
    ph.second_category_id,
    ph.third_category_id
WITH DATA;

-- Create index on the materialized view for performance
CREATE INDEX idx_mv_daily_sales_date ON mv_daily_sales(sale_date);
CREATE INDEX idx_mv_daily_sales_store ON mv_daily_sales(store_id);
CREATE INDEX idx_mv_daily_sales_product ON mv_daily_sales(product_id);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_all_mv()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_sales;
    -- Add more materialized views here as needed
END;
$$ LANGUAGE plpgsql;

-- Create stored procedure for stockout demand estimation
CREATE OR REPLACE FUNCTION estimate_stockout_demand(
    p_store_id INTEGER,
    p_product_id INTEGER,
    p_start_date DATE,
    p_end_date DATE
)
RETURNS TABLE (
    date DATE,
    estimated_demand FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH avg_sales AS (
        SELECT 
            EXTRACT(DOW FROM sale_date) as day_of_week,
            AVG(sale_amount) as avg_daily_sales
        FROM sales_data
        WHERE 
            store_id = p_store_id AND
            product_id = p_product_id AND
            stock_hour6_22_cnt > 0 AND
            sale_date BETWEEN (p_start_date - INTERVAL '28 days') AND (p_start_date - INTERVAL '1 day')
        GROUP BY EXTRACT(DOW FROM sale_date)
    ),
    stockout_days AS (
        SELECT 
            sale_date,
            stock_hour6_22_cnt,
            EXTRACT(DOW FROM sale_date) as day_of_week
        FROM sales_data
        WHERE 
            store_id = p_store_id AND
            product_id = p_product_id AND
            sale_date BETWEEN p_start_date AND p_end_date AND
            stock_hour6_22_cnt = 0
    )
    SELECT 
        sd.sale_date as date,
        COALESCE(avs.avg_daily_sales, 0) as estimated_demand
    FROM stockout_days sd
    LEFT JOIN avg_sales avs ON sd.day_of_week = avs.day_of_week;
END;
$$ LANGUAGE plpgsql;

-- Create function for holiday sales impact calculation
CREATE OR REPLACE FUNCTION calculate_holiday_impact(
    p_product_id INTEGER,
    p_holiday_name VARCHAR
)
RETURNS TABLE (
    holiday_date DATE,
    normal_avg_sales FLOAT,
    holiday_sales FLOAT,
    sales_lift_pct FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH holiday_sales AS (
        SELECT 
            h.date,
            SUM(sd.sale_amount) as total_sales
        FROM holiday_calendar h
        JOIN sales_data sd ON h.date = sd.sale_date
        WHERE 
            h.holiday_name = p_holiday_name AND
            sd.product_id = p_product_id
        GROUP BY h.date
    ),
    normal_sales AS (
        SELECT
            h.date,
            (
                SELECT AVG(sd2.sale_amount)
                FROM sales_data sd2
                WHERE 
                    sd2.product_id = p_product_id AND
                    sd2.sale_date BETWEEN (h.date - INTERVAL '14 days') AND (h.date - INTERVAL '1 day') AND
                    EXTRACT(DOW FROM sd2.sale_date) = EXTRACT(DOW FROM h.date)
            ) as avg_normal_sales
        FROM holiday_calendar h
        WHERE h.holiday_name = p_holiday_name
        GROUP BY h.date
    )
    SELECT 
        hs.date as holiday_date,
        ns.avg_normal_sales as normal_avg_sales,
        hs.total_sales as holiday_sales,
        ((hs.total_sales - ns.avg_normal_sales) / NULLIF(ns.avg_normal_sales, 0)) * 100 as sales_lift_pct
    FROM holiday_sales hs
    JOIN normal_sales ns ON hs.date = ns.date;
END;
$$ LANGUAGE plpgsql;
