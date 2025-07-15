-- =============================================================================
-- ENHANCED FRESHRETAILNET-50K COMPATIBLE SCHEMA
-- Supporting multi-dimensional forecasting across 6 core use cases
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Drop existing tables in correct order
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

-- City hierarchy - 18 major cities from FreshRetailNet-50K
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

-- Store hierarchy - 898 stores across 18 cities
CREATE TABLE store_hierarchy (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    city_id INTEGER NOT NULL REFERENCES city_hierarchy(city_id),
    management_group_id INTEGER NOT NULL,
    store_format VARCHAR(50) NOT NULL, -- 'supermarket', 'convenience', 'hypermarket'
    store_size_sqm INTEGER,
    location_type VARCHAR(50), -- 'urban', 'suburban', 'rural'
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    opening_hours JSONB, -- Store operating hours
    employee_count INTEGER,
    parking_spaces INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Product hierarchy - Enhanced for perishable goods focus
CREATE TABLE product_hierarchy (
    product_id INTEGER PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    
    -- Multi-level categorization
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    
    -- Product characteristics
    is_perishable BOOLEAN DEFAULT TRUE,
    shelf_life_days INTEGER,
    storage_temperature_min DECIMAL(5,2),
    storage_temperature_max DECIMAL(5,2),
    
    -- Business metrics
    unit_cost DECIMAL(10,2),
    standard_price DECIMAL(10,2),
    profit_margin DECIMAL(5,4),
    supplier_id INTEGER,
    
    -- Demand characteristics
    seasonality_pattern VARCHAR(50), -- 'high_summer', 'winter_peak', 'stable'
    weather_sensitivity_score DECIMAL(3,2), -- 0.0 to 1.0
    promotion_elasticity DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- CORE TRANSACTIONAL DATA (FreshRetailNet-50K Structure)
-- =============================================================================

-- Main sales data table - Daily aggregated with hourly breakdown
CREATE TABLE sales_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- Location and product identifiers
    city_id INTEGER NOT NULL REFERENCES city_hierarchy(city_id),
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    management_group_id INTEGER NOT NULL,
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER NOT NULL,
    third_category_id INTEGER NOT NULL,
    
    -- Temporal information
    sale_date DATE NOT NULL,
    
    -- Sales metrics (matching FreshRetailNet-50K)
    sale_amount DECIMAL(12,4) NOT NULL, -- Daily total sales amount
    hours_sale DECIMAL[] NOT NULL, -- Array of 24 hourly sales values
    
    -- Stock status (unique FreshRetailNet-50K feature)
    stock_hour6_22_cnt INTEGER DEFAULT 0, -- Out-of-stock hours between 6:00-22:00
    hours_stock_status INTEGER[] NOT NULL, -- Array of 24 hourly stock status (0=out, 1=in)
    
    -- Promotional information
    discount DECIMAL(5,4) DEFAULT 1.0, -- 1.0 = no discount, 0.9 = 10% off
    activity_flag INTEGER DEFAULT 0, -- Special activity indicator
    
    -- External factors
    holiday_flag INTEGER DEFAULT 0, -- Holiday indicator
    precpt DECIMAL(8,2), -- Precipitation
    avg_temperature DECIMAL(5,2),
    avg_humidity DECIMAL(5,2),
    avg_wind_level DECIMAL(5,2),
    
    -- Enhanced metrics for analysis
    units_sold INTEGER,
    average_unit_price DECIMAL(10,2),
    stockout_duration_minutes INTEGER DEFAULT 0,
    peak_hour INTEGER, -- Hour with highest sales
    demand_volatility DECIMAL(5,4), -- Coefficient of variation for hourly sales
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- ENHANCED ANALYTICAL TABLES
-- =============================================================================

-- Detailed hourly sales breakdown for granular analysis
CREATE TABLE hourly_sales_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- Foreign keys
    sale_date DATE NOT NULL,
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    
    -- Hourly breakdown
    hour_of_day INTEGER NOT NULL CHECK (hour_of_day >= 0 AND hour_of_day <= 23),
    sales_amount DECIMAL(10,4) NOT NULL,
    units_sold INTEGER NOT NULL,
    stock_available BOOLEAN NOT NULL,
    customers_served INTEGER,
    
    -- Context
    temperature DECIMAL(5,2),
    is_promotion_active BOOLEAN DEFAULT FALSE,
    staff_count INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(sale_date, store_id, product_id, hour_of_day)
);

-- Stockout events tracking for risk assessment
CREATE TABLE stockout_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    
    stockout_start TIMESTAMPTZ NOT NULL,
    stockout_end TIMESTAMPTZ,
    duration_hours DECIMAL(5,2),
    
    -- Impact analysis
    estimated_lost_sales DECIMAL(12,2),
    estimated_lost_customers INTEGER,
    restock_urgency_score DECIMAL(3,2), -- 0.0 to 1.0
    
    -- Root cause
    cause_category VARCHAR(50), -- 'demand_surge', 'supply_delay', 'forecasting_error'
    supplier_delay_hours INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Product correlation matrix for cross-selling analysis
CREATE TABLE product_correlations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    product_a_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    product_b_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    store_id INTEGER REFERENCES store_hierarchy(store_id), -- NULL for global correlation
    
    correlation_coefficient DECIMAL(5,4) NOT NULL,
    correlation_type VARCHAR(20) NOT NULL, -- 'complementary', 'substitute', 'neutral'
    confidence_level DECIMAL(3,2),
    
    -- Temporal patterns
    seasonal_strength DECIMAL(3,2),
    weekly_pattern JSONB, -- Day-of-week correlation patterns
    
    analysis_period_start DATE NOT NULL,
    analysis_period_end DATE NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(product_a_id, product_b_id, store_id)
);

-- Store performance metrics for clustering and comparison
CREATE TABLE store_performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    metric_date DATE NOT NULL,
    
    -- Sales performance
    total_revenue DECIMAL(12,2),
    total_units_sold INTEGER,
    average_transaction_value DECIMAL(8,2),
    customer_count INTEGER,
    
    -- Operational efficiency
    stockout_frequency DECIMAL(5,4), -- Percentage of time out of stock
    inventory_turnover DECIMAL(5,2),
    staff_productivity DECIMAL(8,2), -- Revenue per employee
    
    -- Customer experience
    customer_satisfaction_score DECIMAL(3,2),
    average_wait_time_minutes DECIMAL(5,2),
    return_rate DECIMAL(5,4),
    
    -- Comparative metrics
    performance_rank INTEGER, -- Within city ranking
    peer_group_average DECIMAL(12,2),
    performance_deviation DECIMAL(5,4),
    
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(store_id, metric_date)
);

-- Weather impact analysis for demand modeling
CREATE TABLE weather_impact_analysis (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    city_id INTEGER NOT NULL REFERENCES city_hierarchy(city_id),
    
    -- Weather sensitivity metrics
    temperature_elasticity DECIMAL(6,4), -- Sales change per degree
    humidity_impact DECIMAL(6,4),
    precipitation_impact DECIMAL(6,4),
    wind_impact DECIMAL(6,4),
    
    -- Seasonal patterns
    spring_multiplier DECIMAL(4,2),
    summer_multiplier DECIMAL(4,2),
    autumn_multiplier DECIMAL(4,2),
    winter_multiplier DECIMAL(4,2),
    
    -- Optimal conditions
    optimal_temperature_min DECIMAL(5,2),
    optimal_temperature_max DECIMAL(5,2),
    optimal_humidity_range DECIMAL(5,2),
    
    confidence_score DECIMAL(3,2),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(product_id, city_id)
);

-- Promotion effectiveness tracking
CREATE TABLE promotion_effectiveness (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    
    promotion_start_date DATE NOT NULL,
    promotion_end_date DATE NOT NULL,
    discount_percentage DECIMAL(5,2) NOT NULL,
    promotion_type VARCHAR(50), -- 'percentage_off', 'buy_one_get_one', 'bundle'
    
    -- Effectiveness metrics
    baseline_sales DECIMAL(12,2), -- Pre-promotion average
    promotion_sales DECIMAL(12,2),
    uplift_percentage DECIMAL(6,2),
    incremental_revenue DECIMAL(12,2),
    roi DECIMAL(6,2),
    
    -- Customer behavior
    new_customers_acquired INTEGER,
    customer_retention_rate DECIMAL(5,4),
    cross_sell_rate DECIMAL(5,4),
    
    -- Market response
    competitor_response BOOLEAN DEFAULT FALSE,
    cannibalization_effect DECIMAL(5,4), -- Impact on other products
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Category performance aggregation
CREATE TABLE category_performance (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER REFERENCES store_hierarchy(store_id), -- NULL for global
    first_category_id INTEGER NOT NULL,
    second_category_id INTEGER,
    third_category_id INTEGER,
    
    performance_date DATE NOT NULL,
    
    -- Performance metrics
    total_revenue DECIMAL(12,2),
    total_units_sold INTEGER,
    product_count INTEGER,
    active_skus INTEGER,
    
    -- Market share
    category_market_share DECIMAL(5,4), -- Within store/company
    growth_rate DECIMAL(6,4), -- YoY growth
    seasonality_index DECIMAL(4,2),
    
    -- Profitability
    gross_margin DECIMAL(5,4),
    inventory_turnover DECIMAL(5,2),
    
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(store_id, first_category_id, second_category_id, third_category_id, performance_date)
);

-- Store clustering results
CREATE TABLE store_clusters (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    
    -- Clustering results
    cluster_id INTEGER NOT NULL,
    cluster_name VARCHAR(100),
    clustering_method VARCHAR(50), -- 'performance', 'demographic', 'geographic'
    
    -- Cluster characteristics
    cluster_size INTEGER, -- Number of stores in cluster
    performance_tier VARCHAR(20), -- 'high', 'medium', 'low'
    similarity_score DECIMAL(3,2), -- How similar to cluster centroid
    
    -- Key differentiators
    key_strength VARCHAR(100),
    improvement_opportunity VARCHAR(100),
    peer_stores INTEGER[], -- Array of similar store IDs
    
    analysis_date DATE NOT NULL,
    confidence_score DECIMAL(3,2),
    
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Forecasting results storage
CREATE TABLE demand_forecasts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER NOT NULL REFERENCES store_hierarchy(store_id),
    product_id INTEGER NOT NULL REFERENCES product_hierarchy(product_id),
    
    forecast_date DATE NOT NULL,
    forecast_horizon_days INTEGER NOT NULL,
    
    -- Forecast values
    predicted_demand DECIMAL(10,2),
    confidence_lower DECIMAL(10,2),
    confidence_upper DECIMAL(10,2),
    
    -- Model information
    model_type VARCHAR(50), -- 'prophet', 'xgboost', 'ensemble'
    model_accuracy DECIMAL(5,4), -- Historical accuracy score
    feature_importance JSONB, -- Important features and weights
    
    -- Contextual factors
    weather_factor DECIMAL(4,2),
    seasonality_factor DECIMAL(4,2),
    promotion_factor DECIMAL(4,2),
    trend_factor DECIMAL(4,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(store_id, product_id, forecast_date, forecast_horizon_days, model_type)
);

-- =============================================================================
-- SUPPORTING TABLES
-- =============================================================================

-- Enhanced promotions table
CREATE TABLE promotions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    store_id INTEGER REFERENCES store_hierarchy(store_id), -- NULL for chain-wide
    product_id INTEGER REFERENCES product_hierarchy(product_id), -- NULL for store-wide
    
    promotion_name VARCHAR(200) NOT NULL,
    promotion_type VARCHAR(50) NOT NULL,
    
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    
    discount_percentage DECIMAL(5,2),
    minimum_quantity INTEGER,
    maximum_discount_amount DECIMAL(8,2),
    
    -- Targeting
    customer_segment VARCHAR(50),
    channel VARCHAR(50), -- 'in_store', 'online', 'mobile_app'
    
    -- Performance tracking
    budget DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    target_uplift_percentage DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enhanced holidays table
CREATE TABLE holidays (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    city_id INTEGER REFERENCES city_hierarchy(city_id), -- NULL for national
    
    holiday_name VARCHAR(100) NOT NULL,
    holiday_date DATE NOT NULL,
    holiday_type VARCHAR(50), -- 'national', 'regional', 'religious', 'cultural'
    
    -- Impact characteristics
    shopping_impact VARCHAR(20), -- 'increase', 'decrease', 'neutral'
    impact_strength DECIMAL(3,2), -- 0.0 to 1.0
    preparation_days INTEGER, -- Days before holiday affects shopping
    
    -- Store operations
    stores_closed BOOLEAN DEFAULT FALSE,
    modified_hours BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =============================================================================

-- Primary lookup indexes
CREATE INDEX idx_sales_data_store_product_date ON sales_data (store_id, product_id, sale_date);
CREATE INDEX idx_sales_data_date_range ON sales_data (sale_date);
CREATE INDEX idx_sales_data_city_date ON sales_data (city_id, sale_date);
CREATE INDEX idx_sales_data_category_date ON sales_data (first_category_id, second_category_id, third_category_id, sale_date);

-- Hourly data indexes
CREATE INDEX idx_hourly_sales_datetime ON hourly_sales_data (sale_date, hour_of_day);
CREATE INDEX idx_hourly_sales_store_product ON hourly_sales_data (store_id, product_id, sale_date);

-- Stockout analysis indexes
CREATE INDEX idx_stockout_events_store_product ON stockout_events (store_id, product_id);
CREATE INDEX idx_stockout_events_timerange ON stockout_events (stockout_start, stockout_end);
CREATE INDEX idx_sales_data_stockout ON sales_data (stock_hour6_22_cnt) WHERE stock_hour6_22_cnt > 0;

-- Performance analysis indexes
CREATE INDEX idx_store_performance_date ON store_performance_metrics (metric_date);
CREATE INDEX idx_store_performance_store ON store_performance_metrics (store_id, metric_date);

-- Forecasting indexes
CREATE INDEX idx_demand_forecasts_lookup ON demand_forecasts (store_id, product_id, forecast_date);
CREATE INDEX idx_demand_forecasts_horizon ON demand_forecasts (forecast_horizon_days, created_at);

-- Correlation analysis indexes
CREATE INDEX idx_product_correlations_lookup ON product_correlations (product_a_id, product_b_id);
CREATE INDEX idx_product_correlations_store ON product_correlations (store_id, correlation_coefficient DESC);

-- Weather impact indexes
CREATE INDEX idx_weather_impact_product_city ON weather_impact_analysis (product_id, city_id);
CREATE INDEX idx_sales_data_weather ON sales_data (avg_temperature, avg_humidity, precpt);

-- Promotion effectiveness indexes
CREATE INDEX idx_promotion_effectiveness_store_product ON promotion_effectiveness (store_id, product_id);
CREATE INDEX idx_promotion_effectiveness_timerange ON promotion_effectiveness (promotion_start_date, promotion_end_date);

-- Clustering indexes
CREATE INDEX idx_store_clusters_analysis_date ON store_clusters (analysis_date, cluster_id);
CREATE INDEX idx_store_clusters_performance ON store_clusters (performance_tier, similarity_score DESC);

-- Composite indexes for complex queries
CREATE INDEX idx_sales_data_composite ON sales_data (city_id, store_id, product_id, sale_date);
CREATE INDEX idx_sales_data_stockout_composite ON sales_data (sale_date, stock_hour6_22_cnt, sale_amount);

-- GIN indexes for JSON and array fields
CREATE INDEX idx_hours_sale_gin ON sales_data USING gin (hours_sale);
CREATE INDEX idx_hours_stock_status_gin ON sales_data USING gin (hours_stock_status);
CREATE INDEX idx_feature_importance_gin ON demand_forecasts USING gin (feature_importance);

-- Text search indexes
CREATE INDEX idx_product_hierarchy_text ON product_hierarchy USING gin (product_name gin_trgm_ops);
CREATE INDEX idx_store_hierarchy_text ON store_hierarchy USING gin (store_name gin_trgm_ops);

-- =============================================================================
-- VIEWS FOR COMMON ANALYSIS PATTERNS
-- =============================================================================

-- Daily aggregated view with key metrics
CREATE VIEW daily_sales_summary AS
SELECT 
    sale_date,
    city_id,
    store_id,
    product_id,
    sale_amount,
    array_length(hours_sale, 1) as hours_with_data,
    stock_hour6_22_cnt,
    CASE WHEN stock_hour6_22_cnt > 0 THEN true ELSE false END as had_stockout,
    discount,
    holiday_flag::boolean,
    avg_temperature,
    avg_humidity,
    precpt,
    -- Calculated metrics
    (SELECT SUM(unnest) FROM unnest(hours_sale)) as total_hourly_sales,
    (SELECT AVG(unnest) FROM unnest(hours_sale) WHERE unnest > 0) as avg_hourly_sales,
    (SELECT MAX(unnest) FROM unnest(hours_sale)) as peak_hourly_sales,
    (SELECT COUNT(*) FROM unnest(hours_stock_status) WHERE unnest = 0) as total_stockout_hours
FROM sales_data;

-- Store performance comparison view
CREATE VIEW store_performance_comparison AS
WITH store_metrics AS (
    SELECT 
        s.store_id,
        s.store_name,
        s.city_id,
        c.city_name,
        s.management_group_id,
        COUNT(DISTINCT sd.product_id) as active_products,
        SUM(sd.sale_amount) as total_revenue,
        AVG(sd.sale_amount) as avg_daily_revenue,
        SUM(sd.stock_hour6_22_cnt) as total_stockout_hours,
        AVG(CASE WHEN sd.stock_hour6_22_cnt > 0 THEN 1.0 ELSE 0.0 END) as stockout_frequency,
        COUNT(*) as days_of_data
    FROM store_hierarchy s
    JOIN city_hierarchy c ON s.city_id = c.city_id
    LEFT JOIN sales_data sd ON s.store_id = sd.store_id
    WHERE sd.sale_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY s.store_id, s.store_name, s.city_id, c.city_name, s.management_group_id
)
SELECT 
    *,
    RANK() OVER (PARTITION BY city_id ORDER BY total_revenue DESC) as city_revenue_rank,
    RANK() OVER (ORDER BY total_revenue DESC) as overall_revenue_rank,
    RANK() OVER (PARTITION BY city_id ORDER BY stockout_frequency ASC) as city_reliability_rank
FROM store_metrics;

-- Product performance with weather correlation
CREATE VIEW product_weather_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.is_perishable,
    p.weather_sensitivity_score,
    COUNT(DISTINCT sd.store_id) as stores_carrying,
    SUM(sd.sale_amount) as total_revenue,
    AVG(sd.sale_amount) as avg_daily_sales,
    CORR(sd.sale_amount, sd.avg_temperature) as temperature_correlation,
    CORR(sd.sale_amount, sd.avg_humidity) as humidity_correlation,
    CORR(sd.sale_amount, sd.precpt) as precipitation_correlation,
    AVG(CASE WHEN sd.stock_hour6_22_cnt > 0 THEN 1.0 ELSE 0.0 END) as stockout_rate
FROM product_hierarchy p
LEFT JOIN sales_data sd ON p.product_id = sd.product_id
WHERE sd.sale_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY p.product_id, p.product_name, p.is_perishable, p.weather_sensitivity_score
HAVING COUNT(*) >= 30; -- At least 30 days of data

-- =============================================================================
-- SAMPLE DATA POPULATION (Optional - for testing)
-- =============================================================================

-- This section would be populated with actual FreshRetailNet-50K data
-- For now, we'll create the structure to support the data loading process

COMMENT ON TABLE sales_data IS 'Main sales data table matching FreshRetailNet-50K structure with hourly breakdown and stockout tracking';
COMMENT ON TABLE hourly_sales_data IS 'Detailed hourly sales data for granular analysis and pattern detection';
COMMENT ON TABLE stockout_events IS 'Stockout event tracking for risk assessment and inventory optimization';
COMMENT ON TABLE product_correlations IS 'Product correlation matrix for cross-selling and substitution analysis';
COMMENT ON TABLE store_performance_metrics IS 'Store performance metrics for clustering and comparison analysis';
COMMENT ON TABLE weather_impact_analysis IS 'Weather sensitivity analysis for demand modeling and forecasting';
COMMENT ON TABLE promotion_effectiveness IS 'Promotion impact tracking for optimization and ROI analysis';
COMMENT ON TABLE category_performance IS 'Category-level performance aggregation for hierarchical forecasting';
COMMENT ON TABLE store_clusters IS 'Store clustering results for behavioral segmentation and best practices';
COMMENT ON TABLE demand_forecasts IS 'Forecasting results storage with confidence intervals and model metadata'; 