
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    

    -- Create sales_data table
    CREATE TABLE IF NOT EXISTS sales_data (
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
        holiday_flag INTEGER,
        activity_flag INTEGER,
        precpt FLOAT,
        avg_temperature FLOAT,
        avg_humidity FLOAT,
        avg_wind_level FLOAT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_sales_store_product ON sales_data (store_id, product_id);
    CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_data (dt);
    CREATE INDEX IF NOT EXISTS idx_sales_city ON sales_data (city_id);
    CREATE INDEX IF NOT EXISTS idx_sales_category ON sales_data (first_category_id, second_category_id, third_category_id);
    

    -- Create promotions table
    CREATE TABLE IF NOT EXISTS promotions (
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
    CREATE TABLE IF NOT EXISTS holidays (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        date DATE NOT NULL,
        name VARCHAR(100) NOT NULL,
        country VARCHAR(50) DEFAULT 'USA',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (date, name)
    );
    
Loading data from Hugging Face...
Converting to DataFrame...
Limited dataset to 10 records for testing
Processing data...
Generating INSERT statements...

            -- Insert batch 1/1
            INSERT INTO sales_data (
                city_id, store_id, management_group_id, 
                first_category_id, second_category_id, third_category_id,
                product_id, dt, sale_amount, stock_hour6_22_cnt, 
                discount, holiday_flag, activity_flag, precpt, 
                avg_temperature, avg_humidity, avg_wind_level
            ) 
            VALUES (0, 0, 0, 5, 6, 65, 38, '2022-01-01', 0.1, 0, 1.0, 0, 0, 1.6999, 15.48, 73.54, 1.97), (0, 0, 0, 5, 6, 65, 38, '2022-01-02', 0.1, 1, 1.0, 0, 0, 3.019, 15.08, 76.56, 1.71), (0, 0, 0, 5, 6, 65, 38, '2022-01-03', 0.0, 0, 1.0, 1, 0, 2.0942, 15.91, 76.47, 1.73), (0, 0, 0, 5, 6, 65, 38, '2022-01-04', 0.1, 11, 1.0, 1, 0, 1.5618, 16.13, 77.4, 1.76), (0, 0, 0, 5, 6, 65, 38, '2022-01-05', 0.2, 8, 1.0, 0, 0, 3.5386, 15.37, 78.26, 1.25), (0, 0, 0, 5, 6, 65, 38, '2022-01-06', 0.1, 0, 1.0, 0, 0, 3.1459, 15.69, 76.63, 2.13), (0, 0, 0, 5, 6, 65, 38, '2022-01-07', 0.1, 0, 1.0, 0, 0, 1.7165, 16.11, 76.31, 1.51), (0, 0, 0, 5, 6, 65, 38, '2022-01-08', 0.2, 0, 1.0, 1, 0, 1.3021, 16.08, 74.24, 1.47), (0, 0, 0, 5, 6, 65, 38, '2022-01-09', 0.0, 0, 1.0, 1, 0, 1.909, 16.24, 71.99, 1.46), (0, 0, 0, 5, 6, 65, 38, '2022-01-10', 0.2, 0, 1.0, 1, 0, 2.4001, 16.72, 78.42, 1.67);
            
Generating holiday INSERT statements...

        -- Insert holiday New Year's Day on 2022-01-03
        INSERT INTO holidays (date, name, country)
        VALUES ('2022-01-03', 'New Year's Day', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        

        -- Insert holiday Martin Luther King Jr. Day on 2022-01-04
        INSERT INTO holidays (date, name, country)
        VALUES ('2022-01-04', 'Martin Luther King Jr. Day', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        

        -- Insert holiday Presidents' Day on 2022-01-08
        INSERT INTO holidays (date, name, country)
        VALUES ('2022-01-08', 'Presidents' Day', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        

        -- Insert holiday Memorial Day on 2022-01-09
        INSERT INTO holidays (date, name, country)
        VALUES ('2022-01-09', 'Memorial Day', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        

        -- Insert holiday Independence Day on 2022-01-10
        INSERT INTO holidays (date, name, country)
        VALUES ('2022-01-10', 'Independence Day', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        
