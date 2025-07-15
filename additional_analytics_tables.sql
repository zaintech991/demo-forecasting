-- =============================================================================
-- ADDITIONAL ANALYTICAL TABLES FOR ENHANCED MULTI-MODAL CAPABILITIES
-- Add these tables to your existing Supabase database for advanced features
-- =============================================================================

-- Enable required extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =============================================================================
-- 1. REAL-TIME ALERTS & NOTIFICATIONS SYSTEM
-- =============================================================================

CREATE TABLE real_time_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL, -- 'stockout', 'demand_spike', 'weather_impact', 'promotion_opportunity'
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    store_id INTEGER NOT NULL,
    product_id INTEGER,
    city_id INTEGER,
    alert_message TEXT NOT NULL,
    alert_data JSONB, -- Flexible data storage for alert details
    threshold_value DECIMAL(10,2),
    current_value DECIMAL(10,2),
    predicted_impact DECIMAL(10,2),
    recommended_action TEXT,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    resolution_status VARCHAR(50) DEFAULT 'open', -- 'open', 'in_progress', 'resolved', 'false_positive'
    business_impact_score DECIMAL(5,2), -- 0-100 score
    urgency_level INTEGER DEFAULT 1, -- 1-5 urgency scale
    affected_customers INTEGER,
    estimated_revenue_impact DECIMAL(10,2)
);

-- Indexes for real-time alerts
CREATE INDEX idx_real_time_alerts_type ON real_time_alerts (alert_type, severity, created_at);
CREATE INDEX idx_real_time_alerts_store ON real_time_alerts (store_id, is_acknowledged);
CREATE INDEX idx_real_time_alerts_status ON real_time_alerts (resolution_status, created_at);
CREATE INDEX idx_real_time_alerts_severity ON real_time_alerts (severity, urgency_level, created_at);
CREATE INDEX idx_real_time_alerts_impact ON real_time_alerts (business_impact_score DESC, created_at);

-- =============================================================================
-- 2. CROSS-STORE INVENTORY OPTIMIZATION
-- =============================================================================

CREATE TABLE cross_store_inventory (
    optimization_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_date DATE NOT NULL,
    product_id INTEGER NOT NULL,
    city_id INTEGER NOT NULL,
    total_demand DECIMAL(10,2),
    total_supply DECIMAL(10,2),
    surplus_stores INTEGER[], -- Array of store IDs with surplus
    deficit_stores INTEGER[], -- Array of store IDs with deficit
    recommended_transfers JSONB, -- Store-to-store transfer recommendations
    transfer_cost_estimate DECIMAL(10,2),
    potential_revenue_impact DECIMAL(10,2),
    optimization_score DECIMAL(5,2), -- 0-100 score
    weather_factor DECIMAL(5,2),
    seasonality_factor DECIMAL(5,2),
    promotion_factor DECIMAL(5,2),
    demand_volatility DECIMAL(5,2),
    implementation_priority VARCHAR(20), -- 'low', 'medium', 'high', 'urgent'
    estimated_implementation_time INTEGER, -- hours
    risk_assessment JSONB,
    success_probability DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for cross-store inventory
CREATE INDEX idx_cross_store_inventory_date ON cross_store_inventory (analysis_date, city_id);
CREATE INDEX idx_cross_store_inventory_product ON cross_store_inventory (product_id, optimization_score DESC);
CREATE INDEX idx_cross_store_inventory_priority ON cross_store_inventory (implementation_priority, potential_revenue_impact DESC);
CREATE INDEX idx_cross_store_inventory_city_date ON cross_store_inventory (city_id, analysis_date, optimization_score DESC);

-- =============================================================================
-- 3. MULTI-PRODUCT PORTFOLIO ANALYSIS
-- =============================================================================

CREATE TABLE portfolio_analysis (
    portfolio_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_date DATE NOT NULL,
    store_id INTEGER NOT NULL,
    city_id INTEGER NOT NULL,
    portfolio_type VARCHAR(50), -- 'complementary', 'substitute', 'seasonal_bundle', 'promotion_set'
    product_ids INTEGER[] NOT NULL,
    correlation_matrix JSONB,
    cross_elasticity JSONB, -- Price/demand elasticity between products
    bundle_performance JSONB,
    optimal_pricing JSONB,
    cannibalization_risk DECIMAL(5,2),
    synergy_score DECIMAL(5,2),
    revenue_opportunity DECIMAL(10,2),
    margin_impact DECIMAL(10,2),
    customer_adoption_rate DECIMAL(5,2),
    seasonal_performance JSONB,
    competitive_analysis JSONB,
    recommended_strategy TEXT,
    implementation_timeline VARCHAR(50),
    success_probability DECIMAL(5,2),
    risk_factors JSONB,
    kpi_projections JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Indexes for portfolio analysis
CREATE INDEX idx_portfolio_analysis_store_date ON portfolio_analysis (store_id, analysis_date);
CREATE INDEX idx_portfolio_analysis_type ON portfolio_analysis (portfolio_type, synergy_score DESC);
CREATE INDEX idx_portfolio_analysis_opportunity ON portfolio_analysis (revenue_opportunity DESC, success_probability DESC);
CREATE INDEX idx_portfolio_analysis_city ON portfolio_analysis (city_id, portfolio_type, analysis_date);

-- =============================================================================
-- 4. CUSTOMER BEHAVIOR SIMULATION & PATTERNS
-- =============================================================================

CREATE TABLE customer_behavior_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_period_start DATE NOT NULL,
    analysis_period_end DATE NOT NULL,
    store_id INTEGER NOT NULL,
    city_id INTEGER NOT NULL,
    customer_segment VARCHAR(50), -- 'price_sensitive', 'quality_focused', 'convenience_driven', 'health_conscious'
    demographic_profile JSONB,
    shopping_time_patterns JSONB, -- Hour-by-hour shopping preferences
    product_affinity JSONB, -- Product preference scores
    weather_sensitivity JSONB, -- How weather affects shopping
    promotion_responsiveness DECIMAL(5,2),
    brand_loyalty_score DECIMAL(5,2),
    price_elasticity DECIMAL(5,2),
    seasonal_variation JSONB,
    average_basket_size DECIMAL(10,2),
    average_transaction_value DECIMAL(10,2),
    visit_frequency DECIMAL(5,2),
    churn_probability DECIMAL(5,2),
    lifetime_value DECIMAL(10,2),
    next_visit_prediction DATE,
    preferred_channels JSONB,
    behavioral_insights TEXT,
    targeting_recommendations JSONB,
    engagement_score DECIMAL(5,2),
    satisfaction_indicators JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for customer behavior patterns
CREATE INDEX idx_customer_behavior_store ON customer_behavior_patterns (store_id, customer_segment);
CREATE INDEX idx_customer_behavior_value ON customer_behavior_patterns (lifetime_value DESC, churn_probability ASC);
CREATE INDEX idx_customer_behavior_period ON customer_behavior_patterns (analysis_period_start, analysis_period_end);
CREATE INDEX idx_customer_behavior_city ON customer_behavior_patterns (city_id, customer_segment, engagement_score DESC);

-- =============================================================================
-- 5. COMPETITIVE INTELLIGENCE ANALYSIS
-- =============================================================================

CREATE TABLE competitive_intelligence (
    intelligence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_date DATE NOT NULL,
    city_id INTEGER NOT NULL,
    product_category VARCHAR(100),
    competitor_name VARCHAR(100),
    competitor_data JSONB, -- Comprehensive competitor information
    competitor_pricing JSONB, -- Price comparison data
    market_share_estimate DECIMAL(5,2),
    promotion_frequency DECIMAL(5,2),
    promotion_effectiveness DECIMAL(5,2),
    customer_rating DECIMAL(3,1),
    service_level_score DECIMAL(5,2),
    product_variety_score DECIMAL(5,2),
    location_advantage_score DECIMAL(5,2),
    digital_presence_score DECIMAL(5,2),
    competitive_advantage TEXT,
    competitive_gaps JSONB,
    threat_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    opportunity_level VARCHAR(20),
    recommended_response JSONB,
    price_positioning VARCHAR(50), -- 'premium', 'competitive', 'value', 'discount'
    differentiation_opportunities TEXT,
    market_gaps JSONB,
    strategic_recommendations TEXT,
    monitoring_frequency VARCHAR(50),
    data_confidence DECIMAL(5,2),
    impact_assessment JSONB,
    response_timeline VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    next_analysis_date DATE
);

-- Indexes for competitive intelligence
CREATE INDEX idx_competitive_intelligence_city_date ON competitive_intelligence (city_id, analysis_date);
CREATE INDEX idx_competitive_intelligence_threat ON competitive_intelligence (threat_level, market_share_estimate DESC);
CREATE INDEX idx_competitive_intelligence_category ON competitive_intelligence (product_category, threat_level);
CREATE INDEX idx_competitive_intelligence_competitor ON competitive_intelligence (competitor_name, city_id, analysis_date);

-- =============================================================================
-- 6. PREDICTIVE MAINTENANCE & EQUIPMENT ANALYTICS
-- =============================================================================

CREATE TABLE equipment_analytics (
    equipment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    store_id INTEGER NOT NULL,
    equipment_type VARCHAR(50), -- 'refrigeration', 'pos_system', 'hvac', 'lighting', 'security'
    equipment_name VARCHAR(100),
    equipment_model VARCHAR(100),
    serial_number VARCHAR(100),
    installation_date DATE,
    last_maintenance_date DATE,
    next_scheduled_maintenance DATE,
    performance_metrics JSONB, -- Energy consumption, efficiency scores, etc.
    operational_status VARCHAR(50), -- 'optimal', 'normal', 'degraded', 'critical', 'offline'
    failure_probability DECIMAL(5,2),
    maintenance_urgency VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    predicted_failure_date DATE,
    estimated_repair_cost DECIMAL(10,2),
    estimated_replacement_cost DECIMAL(10,2),
    business_impact_score DECIMAL(5,2), -- Impact on sales if equipment fails
    energy_efficiency_rating DECIMAL(3,1),
    environmental_impact JSONB,
    maintenance_history JSONB,
    maintenance_schedule JSONB,
    vendor_information JSONB,
    warranty_status VARCHAR(50),
    warranty_expiry_date DATE,
    replacement_recommendation BOOLEAN DEFAULT FALSE,
    roi_improvement_potential DECIMAL(10,2),
    downtime_risk_hours INTEGER,
    compliance_status JSONB,
    safety_rating DECIMAL(3,1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for equipment analytics
CREATE INDEX idx_equipment_analytics_store ON equipment_analytics (store_id, equipment_type);
CREATE INDEX idx_equipment_analytics_urgency ON equipment_analytics (maintenance_urgency, predicted_failure_date);
CREATE INDEX idx_equipment_analytics_impact ON equipment_analytics (business_impact_score DESC, failure_probability DESC);
CREATE INDEX idx_equipment_analytics_status ON equipment_analytics (operational_status, maintenance_urgency);

-- =============================================================================
-- 7. ADVANCED DEMAND FORECASTING METADATA
-- =============================================================================

CREATE TABLE forecast_model_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    training_date DATE NOT NULL,
    validation_period_start DATE NOT NULL,
    validation_period_end DATE NOT NULL,
    store_id INTEGER,
    product_id INTEGER,
    city_id INTEGER,
    category_id INTEGER,
    accuracy_metrics JSONB, -- MAE, RMSE, MAPE, R², etc.
    performance_by_horizon JSONB, -- Accuracy at different forecast horizons
    feature_importance JSONB,
    model_parameters JSONB,
    training_data_quality DECIMAL(5,2),
    model_stability_score DECIMAL(5,2),
    drift_detection_score DECIMAL(5,2),
    retraining_recommendation VARCHAR(50),
    deployment_status VARCHAR(50), -- 'active', 'testing', 'deprecated', 'archived'
    a_b_test_results JSONB,
    business_impact_metrics JSONB,
    computational_cost DECIMAL(10,2),
    inference_time_ms INTEGER,
    memory_usage_mb INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for forecast model performance
CREATE INDEX idx_forecast_model_performance_name ON forecast_model_performance (model_name, model_version);
CREATE INDEX idx_forecast_model_performance_date ON forecast_model_performance (training_date, validation_period_end);
CREATE INDEX idx_forecast_model_performance_accuracy ON forecast_model_performance ((accuracy_metrics->>'mae')::decimal);
CREATE INDEX idx_forecast_model_performance_store ON forecast_model_performance (store_id, product_id);

-- =============================================================================
-- 8. BUSINESS INSIGHTS & RECOMMENDATIONS TRACKING
-- =============================================================================

CREATE TABLE business_insights (
    insight_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insight_type VARCHAR(50), -- 'optimization', 'alert', 'opportunity', 'risk', 'trend'
    category VARCHAR(50), -- 'sales', 'inventory', 'promotion', 'weather', 'customer', 'competition'
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    insight_data JSONB,
    affected_entities JSONB, -- stores, products, categories affected
    confidence_score DECIMAL(5,2),
    priority_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    potential_impact JSONB, -- revenue, cost, customer satisfaction impact
    recommended_actions JSONB,
    implementation_complexity VARCHAR(20), -- 'low', 'medium', 'high'
    estimated_implementation_time INTEGER, -- hours
    estimated_roi DECIMAL(10,2),
    success_probability DECIMAL(5,2),
    risk_assessment JSONB,
    dependencies JSONB,
    status VARCHAR(50) DEFAULT 'new', -- 'new', 'under_review', 'approved', 'implementing', 'completed', 'rejected'
    assigned_to VARCHAR(100),
    implementation_start_date DATE,
    implementation_end_date DATE,
    actual_impact JSONB,
    lessons_learned TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Indexes for business insights
CREATE INDEX idx_business_insights_type ON business_insights (insight_type, category, priority_level);
CREATE INDEX idx_business_insights_status ON business_insights (status, priority_level, created_at);
CREATE INDEX idx_business_insights_impact ON business_insights ((potential_impact->>'revenue_impact')::decimal DESC);
CREATE INDEX idx_business_insights_roi ON business_insights (estimated_roi DESC, success_probability DESC);

-- =============================================================================
-- 9. ADVANCED ANALYTICS CONFIGURATION
-- =============================================================================

CREATE TABLE analytics_configuration (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_category VARCHAR(50), -- 'alerts', 'models', 'optimization', 'reporting'
    config_name VARCHAR(100) NOT NULL,
    config_scope VARCHAR(50), -- 'global', 'city', 'store', 'product', 'category'
    scope_id INTEGER, -- ID of the specific scope (store_id, city_id, etc.)
    config_parameters JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    approved_by VARCHAR(100),
    version INTEGER DEFAULT 1,
    effective_date DATE NOT NULL,
    expiry_date DATE,
    description TEXT,
    change_reason TEXT,
    impact_assessment JSONB,
    rollback_config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for analytics configuration
CREATE INDEX idx_analytics_configuration_category ON analytics_configuration (config_category, config_scope, is_active);
CREATE INDEX idx_analytics_configuration_scope ON analytics_configuration (config_scope, scope_id, effective_date);
CREATE INDEX idx_analytics_configuration_active ON analytics_configuration (is_active, effective_date, expiry_date);

-- =============================================================================
-- 10. PERFORMANCE MONITORING & SYSTEM HEALTH
-- =============================================================================

CREATE TABLE system_performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_date DATE NOT NULL,
    metric_hour INTEGER NOT NULL, -- 0-23
    system_component VARCHAR(50), -- 'api', 'database', 'ml_models', 'data_pipeline'
    performance_metrics JSONB,
    response_times JSONB, -- P50, P95, P99 response times
    error_rates JSONB,
    throughput_metrics JSONB,
    resource_utilization JSONB, -- CPU, memory, disk, network
    user_activity_metrics JSONB,
    business_metrics JSONB, -- forecast accuracy, alert effectiveness, etc.
    alert_status VARCHAR(50), -- 'healthy', 'warning', 'critical'
    anomaly_detection_results JSONB,
    automated_actions_taken JSONB,
    manual_interventions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for system performance metrics
CREATE INDEX idx_system_performance_date ON system_performance_metrics (metric_date, metric_hour);
CREATE INDEX idx_system_performance_component ON system_performance_metrics (system_component, metric_date);
CREATE INDEX idx_system_performance_alert ON system_performance_metrics (alert_status, metric_date);

-- =============================================================================
-- VIEWS FOR ENHANCED ANALYTICS
-- =============================================================================

-- Comprehensive alert dashboard view
CREATE VIEW alert_dashboard AS
SELECT 
    r.alert_type,
    r.severity,
    COUNT(*) as alert_count,
    AVG(r.business_impact_score) as avg_impact,
    COUNT(CASE WHEN r.is_acknowledged THEN 1 END) as acknowledged_count,
    COUNT(CASE WHEN r.resolution_status = 'resolved' THEN 1 END) as resolved_count,
    AVG(EXTRACT(EPOCH FROM (r.acknowledged_at - r.created_at))/3600) as avg_response_time_hours,
    MAX(r.created_at) as latest_alert
FROM real_time_alerts r
WHERE r.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY r.alert_type, r.severity
ORDER BY alert_count DESC, avg_impact DESC;

-- Cross-store optimization opportunities view
CREATE VIEW optimization_opportunities AS
SELECT 
    c.city_id,
    c.product_id,
    c.optimization_score,
    c.potential_revenue_impact,
    c.implementation_priority,
    c.success_probability,
    p.product_name,
    ch.city_name,
    c.recommended_transfers,
    c.estimated_implementation_time
FROM cross_store_inventory c
JOIN product_hierarchy p ON c.product_id = p.product_id
JOIN city_hierarchy ch ON c.city_id = ch.city_id
WHERE c.analysis_date >= CURRENT_DATE - INTERVAL '7 days'
  AND c.optimization_score >= 70
  AND c.success_probability >= 80
ORDER BY c.potential_revenue_impact DESC, c.optimization_score DESC;

-- Business insights summary view
CREATE VIEW insights_summary AS
SELECT 
    bi.category,
    bi.insight_type,
    bi.priority_level,
    COUNT(*) as total_insights,
    COUNT(CASE WHEN bi.status = 'completed' THEN 1 END) as completed_insights,
    AVG(bi.confidence_score) as avg_confidence,
    SUM((bi.potential_impact->>'revenue_impact')::decimal) as total_potential_revenue,
    AVG(bi.estimated_roi) as avg_estimated_roi,
    AVG(bi.success_probability) as avg_success_probability
FROM business_insights bi
WHERE bi.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY bi.category, bi.insight_type, bi.priority_level
ORDER BY total_potential_revenue DESC, avg_estimated_roi DESC;

-- =============================================================================
-- FUNCTIONS FOR ENHANCED ANALYTICS
-- =============================================================================

-- Function to generate real-time alerts
CREATE OR REPLACE FUNCTION generate_real_time_alert(
    p_alert_type VARCHAR(50),
    p_severity VARCHAR(20),
    p_store_id INTEGER,
    p_product_id INTEGER DEFAULT NULL,
    p_message TEXT DEFAULT NULL,
    p_threshold_value DECIMAL(10,2) DEFAULT NULL,
    p_current_value DECIMAL(10,2) DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_alert_id UUID;
    v_city_id INTEGER;
    v_impact_score DECIMAL(5,2);
BEGIN
    -- Get city_id from store
    SELECT city_id INTO v_city_id 
    FROM store_hierarchy 
    WHERE store_id = p_store_id;
    
    -- Calculate business impact score
    v_impact_score := CASE p_severity
        WHEN 'critical' THEN 90 + RANDOM() * 10
        WHEN 'high' THEN 70 + RANDOM() * 20
        WHEN 'medium' THEN 40 + RANDOM() * 30
        ELSE 10 + RANDOM() * 30
    END;
    
    -- Insert alert
    INSERT INTO real_time_alerts (
        alert_type, severity, store_id, product_id, city_id,
        alert_message, threshold_value, current_value,
        business_impact_score,
        urgency_level,
        expires_at
    ) VALUES (
        p_alert_type, p_severity, p_store_id, p_product_id, v_city_id,
        COALESCE(p_message, 'System generated alert'),
        p_threshold_value, p_current_value,
        v_impact_score,
        CASE p_severity WHEN 'critical' THEN 5 WHEN 'high' THEN 4 WHEN 'medium' THEN 3 ELSE 2 END,
        NOW() + INTERVAL '24 hours'
    ) RETURNING alert_id INTO v_alert_id;
    
    RETURN v_alert_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate optimization score
CREATE OR REPLACE FUNCTION calculate_optimization_score(
    p_product_id INTEGER,
    p_city_id INTEGER,
    p_surplus_stores INTEGER[],
    p_deficit_stores INTEGER[]
) RETURNS DECIMAL(5,2) AS $$
DECLARE
    v_score DECIMAL(5,2) := 0;
    v_surplus_count INTEGER;
    v_deficit_count INTEGER;
    v_balance_ratio DECIMAL(5,2);
BEGIN
    v_surplus_count := array_length(p_surplus_stores, 1);
    v_deficit_count := array_length(p_deficit_stores, 1);
    
    -- Calculate balance ratio
    IF v_surplus_count > 0 AND v_deficit_count > 0 THEN
        v_balance_ratio := LEAST(v_surplus_count::DECIMAL / v_deficit_count, v_deficit_count::DECIMAL / v_surplus_count);
        v_score := v_balance_ratio * 50; -- Base score from balance
        
        -- Add bonus for high transfer potential
        v_score := v_score + LEAST(v_surplus_count + v_deficit_count, 10) * 3;
        
        -- Add product-specific factors
        v_score := v_score + RANDOM() * 20; -- Simulated product factors
        
        v_score := GREATEST(0, LEAST(100, v_score));
    END IF;
    
    RETURN v_score;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SAMPLE DATA INSERTION FUNCTIONS (FOR TESTING)
-- =============================================================================

-- Function to populate sample alerts
CREATE OR REPLACE FUNCTION populate_sample_alerts() RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER := 0;
    v_store_id INTEGER;
    v_product_id INTEGER;
BEGIN
    -- Generate sample alerts for testing
    FOR i IN 1..20 LOOP
        v_store_id := 104 + (RANDOM() * 100)::INTEGER;
        v_product_id := 4 + (RANDOM() * 67)::INTEGER;
        
        PERFORM generate_real_time_alert(
            CASE (RANDOM() * 4)::INTEGER
                WHEN 0 THEN 'stockout'
                WHEN 1 THEN 'demand_spike'
                WHEN 2 THEN 'weather_impact'
                ELSE 'promotion_opportunity'
            END,
            CASE (RANDOM() * 4)::INTEGER
                WHEN 0 THEN 'low'
                WHEN 1 THEN 'medium'
                WHEN 2 THEN 'high'
                ELSE 'critical'
            END,
            v_store_id,
            v_product_id,
            'Sample alert for testing - ' || i
        );
        
        v_count := v_count + 1;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE real_time_alerts IS 'Real-time alert system for critical business events and opportunities';
COMMENT ON TABLE cross_store_inventory IS 'Cross-store inventory optimization recommendations and analysis';
COMMENT ON TABLE portfolio_analysis IS 'Multi-product portfolio analysis for bundle optimization and cross-selling';
COMMENT ON TABLE customer_behavior_patterns IS 'Customer behavior simulation and pattern analysis for personalization';
COMMENT ON TABLE competitive_intelligence IS 'Competitive analysis and market intelligence tracking';
COMMENT ON TABLE equipment_analytics IS 'Predictive maintenance and equipment performance analytics';
COMMENT ON TABLE forecast_model_performance IS 'ML model performance tracking and optimization';
COMMENT ON TABLE business_insights IS 'Business insights generation and recommendation tracking';
COMMENT ON TABLE analytics_configuration IS 'System configuration management for analytics features';
COMMENT ON TABLE system_performance_metrics IS 'System performance monitoring and health tracking';

-- =============================================================================
-- GRANT PERMISSIONS (Adjust based on your Supabase setup)
-- =============================================================================

-- Grant permissions to authenticated users (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Enhanced analytics tables created successfully!';
    RAISE NOTICE 'Tables added: real_time_alerts, cross_store_inventory, portfolio_analysis,';
    RAISE NOTICE '              customer_behavior_patterns, competitive_intelligence,';
    RAISE NOTICE '              equipment_analytics, forecast_model_performance,';
    RAISE NOTICE '              business_insights, analytics_configuration,';
    RAISE NOTICE '              system_performance_metrics';
    RAISE NOTICE 'Views added: alert_dashboard, optimization_opportunities, insights_summary';
    RAISE NOTICE 'Functions added: generate_real_time_alert, calculate_optimization_score,';
    RAISE NOTICE '                 populate_sample_alerts';
    RAISE NOTICE '';
    RAISE NOTICE 'To populate sample data for testing, run: SELECT populate_sample_alerts();';
    RAISE NOTICE 'Total estimated storage: ~500MB for full dataset with all features';
END
$$; 