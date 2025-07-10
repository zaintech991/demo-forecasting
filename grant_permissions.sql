-- Grant all privileges on all tables in the schema to retail_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO retail_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO retail_user;

-- Grant specific permissions on each table
GRANT SELECT, INSERT, UPDATE, DELETE ON sales_data TO retail_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON store_hierarchy TO retail_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON product_hierarchy TO retail_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON promotions TO retail_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON holidays TO retail_user;

-- Make sure retail_user owns the tables
ALTER TABLE sales_data OWNER TO retail_user;
ALTER TABLE store_hierarchy OWNER TO retail_user;
ALTER TABLE product_hierarchy OWNER TO retail_user;
ALTER TABLE promotions OWNER TO retail_user;
ALTER TABLE holidays OWNER TO retail_user; 