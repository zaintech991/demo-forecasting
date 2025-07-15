/**
 * Professional Dashboard JavaScript
 * Handles all API interactions, data visualization, and user interface
 * NO MOCK DATA - ALL REAL CALCULATIONS
 */

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';
let currentStore = null;
let currentCity = null;
let currentProduct = null;

// Helper function to ensure valid parameters
function getValidParameters() {
    return {
        city_id: parseInt(currentCity) || 0,
        store_id: parseInt(currentStore) || 104,
        product_id: parseInt(currentProduct) || 4
    };
}

// Chart instances
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

// Initialize dashboard components
async function initializeDashboard() {
    console.log('Initializing dashboard...');
    showLoading(true);
    
    try {
        // Initialize event listeners
        initializeEventListeners();
        console.log('Event listeners initialized');
        
        // Load all dashboard data
        await refreshAllData();
        console.log('All data loaded successfully');
        
        showLoading(false);
    } catch (error) {
        console.error('Dashboard initialization failed:', error);
        showLoading(false);
    }
}

// Initialize event listeners
function initializeEventListeners() {
    const refreshBtn = document.getElementById('refreshBtn');
    const storeSelect = document.getElementById('storeSelect');
    const citySelect = document.getElementById('citySelect');
    const productSelect = document.getElementById('productSelect');
    
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshAllData);
    }
    
    if (storeSelect) {
        storeSelect.addEventListener('change', function() {
            currentStore = parseInt(this.value);
            refreshAllData();
        });
    }
    
    if (citySelect) {
        citySelect.addEventListener('change', function() {
            currentCity = parseInt(this.value);
            refreshAllData();
        });
    }
    
    if (productSelect) {
        productSelect.addEventListener('change', function() {
            currentProduct = parseInt(this.value);
            refreshAllData();
        });
    }
}

// Refresh all dashboard data
async function refreshAllData() {
    showLoading(true);
    
    try {
        // Load data in parallel for better performance
        await Promise.all([
            loadCuratedData(),
            loadForecastingData(),
            loadWeatherData(),
            loadMarketData(),
            loadStoreData(),
            loadInventoryData(),
            loadPromotionData(),
            loadRealTimeData(),
            loadAdvancedData()
        ]);
        
        showLoading(false);
    } catch (error) {
        console.error('Data refresh failed:', error);
        showLoading(false);
    }
}

// Load curated data and update key metrics
async function loadCuratedData() {
    try {
        const response = await fetch(`${API_BASE_URL}/enhanced/curated-data`);
        const result = await response.json();
        
        if (result && result.summary) {
            updateKeyMetrics(result.summary);
        }
        
        // Load and populate dropdowns with real data
        if (result && result.data) {
            populateDropdowns(result.data);
        }
    } catch (error) {
        console.error('Failed to load curated data:', error);
    }
}

// Update key metrics display
function updateKeyMetrics(summary) {
    document.getElementById('totalRevenue').textContent = `$${(summary.total_revenue || 0).toLocaleString()}`;
    document.getElementById('totalProducts').textContent = (summary.total_products || 0).toLocaleString();
    document.getElementById('totalStores').textContent = (summary.total_stores || 0).toLocaleString();
    document.getElementById('forecastAccuracy').textContent = `${(summary.forecast_accuracy || 0).toFixed(1)}%`;
}

// Populate dropdowns with real data from database
function populateDropdowns(data) {
    // Populate cities dropdown
    const citySelect = document.getElementById('citySelect');
    if (citySelect && data.cities) {
        citySelect.innerHTML = '';
        data.cities.forEach(city => {
            const option = document.createElement('option');
            option.value = city.city_id;
            option.textContent = city.display_name;
            citySelect.appendChild(option);
        });
        currentCity = parseInt(data.cities[0]?.city_id || 0);
    }
    
    // Populate stores dropdown
    const storeSelect = document.getElementById('storeSelect');
    if (storeSelect && data.stores) {
        storeSelect.innerHTML = '';
        data.stores.forEach(store => {
            const option = document.createElement('option');
            option.value = store.store_id;
            option.textContent = store.display_name;
            storeSelect.appendChild(option);
        });
        currentStore = parseInt(data.stores[0]?.store_id || 104);
    }
    
    // Populate products dropdown
    const productSelect = document.getElementById('productSelect');
    if (productSelect && data.products) {
        productSelect.innerHTML = '';
        data.products.forEach(product => {
            const option = document.createElement('option');
            option.value = product.product_id;
            option.textContent = product.display_name;
            productSelect.appendChild(option);
        });
        currentProduct = parseInt(data.products[0]?.product_id || 4);
    }
}

// FORECASTING ANALYTICS
async function loadForecastingData() {
    try {
        // Load basic forecast
        const forecastResponse = await fetch(`${API_BASE_URL}/enhanced/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
                forecast_days: 30,
                model_type: 'ensemble',
                include_weather: true,
                include_holidays: true,
                include_promotions: true
            })
        });
        
        const forecastData = await forecastResponse.json();
        
        if (forecastData && forecastData.forecast_data) {
            createForecastChart(forecastData.forecast_data);
        }
        
        // Load confidence intervals
        const confidenceResponse = await fetch(`${API_BASE_URL}/enhanced/confidence-intervals`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const confidenceData = await confidenceResponse.json();
        
        if (confidenceData && confidenceData.confidence_intervals) {
            displayConfidenceIntervals(confidenceData);
        }
        
        // Display forecast accuracy
        if (forecastData && forecastData.model_performance) {
            displayForecastAccuracy(forecastData.model_performance);
        }
        
    } catch (error) {
        console.error('Failed to load forecasting data:', error);
    }
}

// Create forecast chart
function createForecastChart(forecastData) {
    const ctx = document.getElementById('basicForecastChart');
    if (!ctx) {
        console.error('Canvas element basicForecastChart not found');
        return;
    }
    
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded');
        return;
    }
    
    // Destroy existing chart
    if (charts.basicForecast) {
        charts.basicForecast.destroy();
    }
    
    const labels = forecastData.dates || [];
    const values = forecastData.predictions || [];
    const upperBounds = forecastData.upper_bounds || [];
    const lowerBounds = forecastData.lower_bounds || [];
    
    if (labels.length === 0 || values.length === 0) {
        console.error('No data to display in chart');
        return;
    }
    
    try {
        charts.basicForecast = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Sales',
                    data: values,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Upper Bound',
                    data: upperBounds,
                    borderColor: 'rgba(255, 99, 132, 0.5)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.1
                }, {
                    label: 'Lower Bound',
                    data: lowerBounds,
                    borderColor: 'rgba(255, 99, 132, 0.5)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        console.log('Forecast chart created successfully');
    } catch (error) {
        console.error('Error creating forecast chart:', error);
    }
}

// Display confidence intervals
function displayConfidenceIntervals(confidenceData) {
    const container = document.getElementById('confidenceIntervalsResults');
    if (!container) return;
    
    const intervals = confidenceData.confidence_intervals || {};
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${intervals.confidence_level || 'N/A'}</h4>
                    <p>Confidence Level</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${intervals.lower_bound || 'N/A'}</h4>
                    <p>Lower Bound</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${intervals.upper_bound || 'N/A'}</h4>
                    <p>Upper Bound</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Statistical Analysis</h6>
            <p>${confidenceData.analysis || 'Real-time confidence interval calculation based on historical data variance and model performance metrics.'}</p>
        </div>
    `;
}

// Display forecast accuracy
function displayForecastAccuracy(modelPerformance) {
    const container = document.getElementById('forecastAccuracyResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${modelPerformance.accuracy || 'N/A'}</h4>
                    <p>Model Accuracy</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${modelPerformance.mae || 'N/A'}</h4>
                    <p>Mean Absolute Error</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${modelPerformance.rmse || 'N/A'}</h4>
                    <p>Root Mean Square Error</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${modelPerformance.model_type || 'N/A'}</h4>
                    <p>Model Type</p>
                </div>
            </div>
        </div>
    `;
}

// WEATHER INTELLIGENCE
async function loadWeatherData() {
    try {
        // Load seasonal patterns
        const seasonalResponse = await fetch(`${API_BASE_URL}/enhanced/seasonal-patterns`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const seasonalData = await seasonalResponse.json();
        
        if (seasonalData && seasonalData.seasonal_patterns) {
            displaySeasonalPatterns(seasonalData);
        }
        
        // Load weather scenarios
        const weatherResponse = await fetch(`${API_BASE_URL}/enhanced/weather-scenarios`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const weatherData = await weatherResponse.json();
        
        if (weatherData && weatherData.success) {
            displayGenericResults('weatherScenariosResults', weatherData, 'Weather Scenarios');
        }
        
    } catch (error) {
        console.error('Failed to load weather data:', error);
    }
}

// Display seasonal patterns
function displaySeasonalPatterns(seasonalData) {
    const container = document.getElementById('seasonalPatternsResults');
    if (!container) return;
    
    const patterns = seasonalData.seasonal_patterns || {};
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h4>$${patterns.Spring?.avg_sales?.toFixed(2) || 'N/A'}</h4>
                    <p>Spring Sales</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h4>$${patterns.Summer?.avg_sales?.toFixed(2) || 'N/A'}</h4>
                    <p>Summer Sales</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h4>$${patterns.Fall?.avg_sales?.toFixed(2) || 'N/A'}</h4>
                    <p>Fall Sales</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h4>$${patterns.Winter?.avg_sales?.toFixed(2) || 'N/A'}</h4>
                    <p>Winter Sales</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Weather Impact: ${(seasonalData.weather_impact * 100).toFixed(1)}%</h6>
            <p>Best Season: ${seasonalData.best_season || 'N/A'}</p>
            <p>Temperature Correlation: ${seasonalData.temp_correlation || 'N/A'}</p>
        </div>
    `;
}

// Display weather scenarios
function displayWeatherScenarios(insights) {
    const container = document.getElementById('weatherScenariosResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.temperature_impact}%</h4>
                    <p>Temperature Impact</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.precipitation_impact}%</h4>
                    <p>Precipitation Impact</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Weather Scenarios</h6>
            <p>${insights.scenario_analysis}</p>
        </div>
    `;
}

// MARKET ANALYSIS
async function loadMarketData() {
    try {
        // Load market share
        const marketResponse = await fetch(`${API_BASE_URL}/enhanced/market-share`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const marketData = await marketResponse.json();
        
        if (marketData && marketData.market_insights) {
            displayMarketShare(marketData.market_insights);
        }
        
        // Load category correlations
        const correlationResponse = await fetch(`${API_BASE_URL}/enhanced/category-correlations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const correlationData = await correlationResponse.json();
        
        if (correlationData && correlationData.correlation_insights) {
            displayCategoryCorrelations(correlationData.correlation_insights);
        }
        
        // Load portfolio optimization
        const portfolioResponse = await fetch(`${API_BASE_URL}/enhanced/portfolio-optimization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const portfolioData = await portfolioResponse.json();
        
        if (portfolioData && portfolioData.optimization_insights) {
            displayPortfolioOptimization(portfolioData.optimization_insights);
        }
        
    } catch (error) {
        console.error('Failed to load market data:', error);
    }
}

// Display market share
function displayMarketShare(marketInsights) {
    const container = document.getElementById('marketShareResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${marketInsights.market_share?.toFixed(1) || 'N/A'}%</h4>
                    <p>Market Share</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${marketInsights.competitive_position || 'N/A'}</h4>
                    <p>Market Position</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>#${marketInsights.category_ranking || 'N/A'}</h4>
                    <p>Category Ranking</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>$${marketInsights.total_sales?.toFixed(2) || 'N/A'}</h4>
                    <p>Total Sales</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Market Analysis</h6>
            <p>Growth Rate: ${marketInsights.growth_rate || 0}% | Data Points: ${marketInsights.data_points || 0}</p>
        </div>
    `;
}

// Display category correlations
function displayCategoryCorrelations(correlationInsights) {
    const container = document.getElementById('categoryCorrelationsResults');
    if (!container) return;
    
    const correlations = correlationInsights.correlations || {};
    const strongCorrelations = correlationInsights.strong_correlations || [];
    const topCorrelation = Object.entries(correlations)[0];
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${topCorrelation ? topCorrelation[0] : 'N/A'}</h4>
                    <p>Strongest Correlation</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${topCorrelation ? topCorrelation[1].toFixed(3) : 'N/A'}</h4>
                    <p>Correlation Strength</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${correlationInsights.cross_category_impact?.toFixed(1) || 'N/A'}%</h4>
                    <p>Cross-Category Impact</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${correlationInsights.substitution_risk || 'N/A'}</h4>
                    <p>Substitution Risk</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Correlation Analysis</h6>
            <p>Strong Correlations: ${strongCorrelations.length} | Total Categories: ${correlationInsights.total_categories || 0}</p>
        </div>
    `;
}

// Display portfolio optimization
function displayPortfolioOptimization(optimizationInsights) {
    const container = document.getElementById('portfolioOptimizationResults');
    if (!container) return;
    
    const additions = optimizationInsights.recommended_additions || [];
    const removals = optimizationInsights.recommended_removals || [];
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${optimizationInsights.portfolio_efficiency?.toFixed(1) || 'N/A'}%</h4>
                    <p>Portfolio Efficiency</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>$${optimizationInsights.revenue_potential?.toFixed(2) || 'N/A'}</h4>
                    <p>Revenue Potential</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${optimizationInsights.total_products || 'N/A'}</h4>
                    <p>Total Products</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Portfolio Recommendations</h6>
            <p><strong>Add:</strong> ${additions.slice(0, 3).join(', ')} | <strong>Remove:</strong> ${removals.slice(0, 3).join(', ')}</p>
            <p>Top Performer Sales: $${optimizationInsights.top_performer_sales?.toFixed(2) || 'N/A'}</p>
        </div>
    `;
}

// STORE PERFORMANCE
async function loadStoreData() {
    try {
        // Load store clustering
        const clusterResponse = await fetch(`${API_BASE_URL}/enhanced/store-clustering`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const clusterData = await clusterResponse.json();
        
        if (clusterData && clusterData.success) {
            displayStoreClustering(clusterData);
        }
        
        // Load performance ranking
        const rankingResponse = await fetch(`${API_BASE_URL}/enhanced/performance-ranking`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const rankingData = await rankingResponse.json();
        
        if (rankingData && rankingData.success) {
            displayPerformanceRanking(rankingData);
        }
        
    } catch (error) {
        console.error('Failed to load store data:', error);
    }
}

// Display store clustering
function displayStoreClustering(clusterData) {
    const container = document.getElementById('storeClusteringResults');
    if (!container) return;
    
    const insights = clusterData.clustering_insights || {};
    const storeInsights = insights.store_insights || {};
    const summary = insights.performance_summary || {};
    
    // Find which cluster this store belongs to
    let userCluster = null;
    if (insights.all_clusters) {
        const currentStoreId = parseInt(document.getElementById('storeSelect').value) || 104;
        userCluster = insights.all_clusters.find(cluster => 
            cluster.stores && cluster.stores.includes(currentStoreId)
        );
    }
    
    const clusterName = userCluster ? userCluster.name : storeInsights.assigned_cluster || 'Unknown';
    const clusterId = userCluster ? userCluster.id : 'N/A';
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${clusterId}</h4>
                    <p>Cluster ID</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${clusterName}</h4>
                    <p>Cluster Type</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${summary.total_stores || 'N/A'}</h4>
                    <p>Total Stores Analyzed</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${summary.avg_daily_sales ? summary.avg_daily_sales.toFixed(2) : 'N/A'}</h4>
                    <p>Avg Daily Sales</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Cluster Analysis</h6>
            ${userCluster ? `
                <p><strong>Cluster Characteristics:</strong></p>
                <ul>
                    <li>Average Sales: ${userCluster.characteristics.avg_sales.toFixed(2)}</li>
                    <li>Promotion Effectiveness: ${(userCluster.characteristics.avg_promo_effectiveness * 100).toFixed(1)}%</li>
                    <li>Consistency Score: ${(userCluster.characteristics.avg_consistency * 100).toFixed(1)}%</li>
                    <li>Stores in Cluster: ${userCluster.count}</li>
                </ul>
            ` : `
                <p>Store clustering analysis based on performance metrics and customer behavior patterns.</p>
            `}
            ${summary.improvement_opportunities ? `
                <p><strong>Improvement Opportunities:</strong></p>
                <ul>
                    ${summary.improvement_opportunities.map(opp => `<li>${opp}</li>`).join('')}
                </ul>
            ` : ''}
        </div>
    `;
}

// Display performance ranking
function displayPerformanceRanking(rankingData) {
    const container = document.getElementById('performanceRankingResults');
    if (!container) return;
    
    const insights = rankingData.ranking_insights || {};
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.store_rank || 'N/A'}</h4>
                    <p>Performance Rank</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.performance_percentile ? insights.performance_percentile.toFixed(1) + '%' : 'N/A'}</h4>
                    <p>Performance Percentile</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.total_stores || 'N/A'}</h4>
                    <p>Total Stores</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.ranking_trend || 'N/A'}</h4>
                    <p>Ranking Trend</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Performance Analysis</h6>
            <p>Store ranks #${insights.store_rank || 'N/A'} out of ${insights.total_stores || 'N/A'} stores (${insights.performance_percentile ? insights.performance_percentile.toFixed(1) + 'th' : 'N/A'} percentile)</p>
            <p><strong>Trend:</strong> ${insights.ranking_trend || 'Unknown'}</p>
        </div>
    `;
}

// INVENTORY OPTIMIZATION
async function loadInventoryData() {
    try {
        // Load safety stock
        const safetyResponse = await fetch(`${API_BASE_URL}/enhanced/safety-stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const safetyData = await safetyResponse.json();
        
        if (safetyData && safetyData.success) {
            displaySafetyStock(safetyData);
        }
        
        // Load reorder optimization
        const reorderResponse = await fetch(`${API_BASE_URL}/enhanced/reorder-optimization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const reorderData = await reorderResponse.json();
        
        if (reorderData && reorderData.success) {
            displayGenericResults('reorderOptimizationResults', reorderData, 'Reorder Optimization');
        }
        
        // Load cross-store optimization
        const crossStoreResponse = await fetch(`${API_BASE_URL}/enhanced/cross-store-optimization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const crossStoreData = await crossStoreResponse.json();
        
        if (crossStoreData && crossStoreData.success) {
            displayGenericResults('crossStoreOptimizationResults', crossStoreData, 'Cross-Store Optimization');
        }
        
    } catch (error) {
        console.error('Failed to load inventory data:', error);
    }
}

// Display safety stock
function displaySafetyStock(safetyData) {
    const container = document.getElementById('safetyStockResults');
    if (!container) return;
    
    const insights = safetyData.safety_stock_insights || {};
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.recommended_safety_stock || 'N/A'} units</h4>
                    <p>Recommended Safety Stock</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.current_service_level ? insights.current_service_level.toFixed(1) + '%' : 'N/A'}</h4>
                    <p>Current Service Level</p>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.target_service_level ? insights.target_service_level + '%' : 'N/A'}</h4>
                    <p>Target Service Level</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.variance_analysis || 'N/A'}</h4>
                    <p>Demand Variance</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Safety Stock Analysis</h6>
            <p>Recommended safety stock level is <strong>${insights.recommended_safety_stock || 'N/A'} units</strong> based on demand variability analysis.</p>
            <p><strong>Service Level:</strong> Currently at ${insights.current_service_level ? insights.current_service_level.toFixed(1) : 'N/A'}%, targeting ${insights.target_service_level || 'N/A'}%</p>
            <p><strong>Variance Analysis:</strong> ${insights.variance_analysis || 'Unknown'} demand variability</p>
        </div>
    `;
}

// Generic display function for simple API responses
function displayGenericResults(containerId, data, title) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Extract the main insights object from the response
    const insights = data.insights || data[Object.keys(data).find(key => key.includes('insights'))] || {};
    
    // Create a simple display with key-value pairs
    let html = '<div class="row">';
    let count = 0;
    
    for (const [key, value] of Object.entries(insights)) {
        if (count % 2 === 0 && count > 0) {
            html += '</div><div class="row mt-3">';
        }
        
        html += `
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${value || 'N/A'}</h4>
                    <p>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                </div>
            </div>
        `;
        count++;
        
        if (count >= 4) break; // Limit to 4 items max
    }
    
    html += '</div>';
    html += `<div class="mt-3"><h6>${title}</h6><p>Real-time analysis based on current data and historical patterns.</p></div>`;
    
    container.innerHTML = html;
}

// Display reorder optimization
function displayReorderOptimization(insights) {
    const container = document.getElementById('reorderOptimizationResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <h4>${insights.reorder_point}</h4>
            <p>Reorder Point</p>
        </div>
        <div class="mt-3">
            <h6>Reorder Analysis</h6>
            <p>${insights.reorder_analysis}</p>
        </div>
    `;
}

// Display cross-store optimization
function displayCrossStoreOptimization(insights) {
    const container = document.getElementById('crossStoreOptimizationResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <h4>${insights.optimization_score}</h4>
            <p>Optimization Score</p>
        </div>
        <div class="mt-3">
            <h6>Cross-Store Analysis</h6>
            <p>${insights.cross_store_analysis}</p>
        </div>
    `;
}

// PROMOTION ANALYSIS
async function loadPromotionData() {
    try {
        // Load cross-product effects
        const crossProductResponse = await fetch(`${API_BASE_URL}/enhanced/cross-product-effects`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const crossProductData = await crossProductResponse.json();
        
        if (crossProductData && crossProductData.success) {
            displayGenericResults('crossProductEffectsResults', crossProductData, 'Cross-Product Effects');
        }
        
        // Load optimal pricing
        const pricingResponse = await fetch(`${API_BASE_URL}/enhanced/optimal-pricing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const pricingData = await pricingResponse.json();
        
        if (pricingData && pricingData.success) {
            displayGenericResults('optimalPricingResults', pricingData, 'Optimal Pricing');
        }
        
        // Load ROI optimization
        const roiResponse = await fetch(`${API_BASE_URL}/enhanced/roi-optimization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const roiData = await roiResponse.json();
        
        if (roiData && roiData.success) {
            displayGenericResults('roiOptimizationResults', roiData, 'ROI Optimization');
        }
        
    } catch (error) {
        console.error('Failed to load promotion data:', error);
    }
}

// Display cross-product effects
function displayCrossProductEffects(insights) {
    const container = document.getElementById('crossProductEffectsResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <h4>${insights.cross_effect_strength}</h4>
            <p>Cross Effect Strength</p>
        </div>
        <div class="mt-3">
            <h6>Cross-Product Analysis</h6>
            <p>${insights.cross_product_analysis}</p>
        </div>
    `;
}

// Display optimal pricing
function displayOptimalPricing(insights) {
    const container = document.getElementById('optimalPricingResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <h4>$${insights.optimal_price}</h4>
            <p>Optimal Price</p>
        </div>
        <div class="mt-3">
            <h6>Pricing Analysis</h6>
            <p>${insights.pricing_analysis}</p>
        </div>
    `;
}

// Display ROI optimization
function displayROIOptimization(insights) {
    const container = document.getElementById('roiOptimizationResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <h4>${insights.roi_score}%</h4>
            <p>ROI Score</p>
        </div>
        <div class="mt-3">
            <h6>ROI Analysis</h6>
            <p>${insights.roi_analysis}</p>
        </div>
    `;
}

// REAL-TIME INTELLIGENCE
async function loadRealTimeData() {
    try {
        // Load live alerts
        const alertsResponse = await fetch(`${API_BASE_URL}/enhanced/live-alerts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const alertsData = await alertsResponse.json();
        
        if (alertsData && alertsData.success) {
            displayGenericResults('liveAlertsResults', alertsData, 'Live Alerts');
        }
        
        // Load demand monitoring
        const demandResponse = await fetch(`${API_BASE_URL}/enhanced/demand-monitoring`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const demandData = await demandResponse.json();
        
        if (demandData && demandData.success) {
            displayGenericResults('demandMonitoringResults', demandData, 'Demand Monitoring');
        }
        
    } catch (error) {
        console.error('Failed to load real-time data:', error);
    }
}

// Display live alerts
function displayLiveAlerts(insights) {
    const container = document.getElementById('liveAlertsResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.alert_count}</h4>
                    <p>Active Alerts</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.priority_level}</h4>
                    <p>Priority Level</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Alert Summary</h6>
            <p>${insights.alert_summary}</p>
        </div>
    `;
}

// Display demand monitoring
function displayDemandMonitoring(insights) {
    const container = document.getElementById('demandMonitoringResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.demand_trend}</h4>
                    <p>Demand Trend</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.demand_volatility}%</h4>
                    <p>Demand Volatility</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Demand Analysis</h6>
            <p>${insights.demand_analysis}</p>
        </div>
    `;
}

// ADVANCED ANALYTICS
async function loadAdvancedData() {
    try {
        // Load competitive intelligence
        const competitiveResponse = await fetch(`${API_BASE_URL}/enhanced/competitive-intelligence`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const competitiveData = await competitiveResponse.json();
        
        if (competitiveData && competitiveData.success) {
            displayGenericResults('competitiveIntelligenceResults', competitiveData, 'Competitive Intelligence');
        }
        
        // Load customer behavior
        const behaviorResponse = await fetch(`${API_BASE_URL}/enhanced/customer-behavior`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const behaviorData = await behaviorResponse.json();
        
        if (behaviorData && behaviorData.success) {
            displayGenericResults('customerBehaviorResults', behaviorData, 'Customer Behavior');
        }
        
        // Load anomaly detection
        const anomalyResponse = await fetch(`${API_BASE_URL}/enhanced/anomaly-detection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...getValidParameters(),
            })
        });
        const anomalyData = await anomalyResponse.json();
        
        if (anomalyData && anomalyData.success) {
            displayGenericResults('anomalyDetectionResults', anomalyData, 'Anomaly Detection');
        }
        
    } catch (error) {
        console.error('Failed to load advanced data:', error);
    }
}

// Display competitive intelligence
function displayCompetitiveIntelligence(insights) {
    const container = document.getElementById('competitiveIntelligenceResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.competitive_position}</h4>
                    <p>Competitive Position</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.market_advantage}%</h4>
                    <p>Market Advantage</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Competitive Analysis</h6>
            <p>${insights.competitive_analysis}</p>
        </div>
    `;
}

// Display customer behavior
function displayCustomerBehavior(insights) {
    const container = document.getElementById('customerBehaviorResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.behavior_pattern}</h4>
                    <p>Behavior Pattern</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h4>${insights.customer_satisfaction}%</h4>
                    <p>Customer Satisfaction</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Behavior Analysis</h6>
            <p>${insights.behavior_analysis}</p>
        </div>
    `;
}

// Display anomaly detection
function displayAnomalyDetection(insights) {
    const container = document.getElementById('anomalyDetectionResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${insights.anomaly_count}</h4>
                    <p>Anomalies Detected</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${insights.severity_level}</h4>
                    <p>Severity Level</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h4>${insights.confidence_score}%</h4>
                    <p>Confidence Score</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h6>Anomaly Analysis</h6>
            <p>${insights.anomaly_analysis}</p>
        </div>
    `;
}

// Show/hide loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// Utility function to format numbers
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

// Utility function to format currency
function formatCurrency(num) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(num);
}

// Utility function to format percentage
function formatPercentage(num) {
    return `${num.toFixed(1)}%`;
} 