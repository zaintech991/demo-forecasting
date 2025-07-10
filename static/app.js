// Constants and configuration
const API_BASE_URL = 'http://localhost:8000/api';

// DOM Elements caching
const elements = {
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    pages: document.querySelectorAll('.page'),
    
    // Forecasting
    forecastForm: document.getElementById('forecast-form'),
    forecastResults: document.getElementById('forecast-results'),
    forecastChart: document.getElementById('forecast-chart'),
    forecastTable: document.getElementById('forecast-table'),
    storeIdInput: document.getElementById('store-id'),
    cityIdInput: document.getElementById('city-id'),
    productIdInput: document.getElementById('product-id'),
    categoryIdInput: document.getElementById('category-id'),
    
    // Promotions
    promoAnalysisForm: document.getElementById('promo-analysis-form'),
    promoAnalysisResults: document.getElementById('promo-analysis-results'),
    promoEffectivenessChart: document.getElementById('promo-effectiveness-chart'),
    promoAnalysisTable: document.getElementById('promo-analysis-table'),
    promoRecommendForm: document.getElementById('promo-recommend-form'),
    promoRecommendResults: document.getElementById('promo-recommend-results'),
    
    // Stockouts
    stockoutForm: document.getElementById('stockout-form'),
    stockoutResults: document.getElementById('stockout-results'),
    stockoutChart: document.getElementById('stockout-chart'),
    
    // Holidays
    holidayForm: document.getElementById('holiday-form'),
    holidayResults: document.getElementById('holiday-results'),
    holidayChart: document.getElementById('holiday-chart')
};

// City and store mappings
const cityMapping = {
    0: {"name": "New York", "state": "NY", "region": "Northeast"},
    1: {"name": "Los Angeles", "state": "CA", "region": "West"},
    2: {"name": "Chicago", "state": "IL", "region": "Midwest"},
    3: {"name": "Houston", "state": "TX", "region": "South"},
    4: {"name": "Phoenix", "state": "AZ", "region": "West"},
    5: {"name": "Philadelphia", "state": "PA", "region": "Northeast"},
    6: {"name": "San Antonio", "state": "TX", "region": "South"},
    7: {"name": "San Diego", "state": "CA", "region": "West"},
    8: {"name": "Dallas", "state": "TX", "region": "South"},
    9: {"name": "Austin", "state": "TX", "region": "South"},
    10: {"name": "San Jose", "state": "CA", "region": "West"},
    11: {"name": "Jacksonville", "state": "FL", "region": "South"},
    12: {"name": "Columbus", "state": "OH", "region": "Midwest"},
    13: {"name": "Seattle", "state": "WA", "region": "West"},
    14: {"name": "Denver", "state": "CO", "region": "West"},
    15: {"name": "Boston", "state": "MA", "region": "Northeast"},
    16: {"name": "Miami", "state": "FL", "region": "South"},
    17: {"name": "Atlanta", "state": "GA", "region": "South"}
};

const storeMapping = {
    0: {"name": "Downtown Market", "format": "Supermarket", "size": "Large"},
    1: {"name": "Neighborhood Fresh", "format": "Convenience", "size": "Small"},
    2: {"name": "Central Grocers", "format": "Supermarket", "size": "Medium"},
    3: {"name": "Express Mart", "format": "Convenience", "size": "Small"},
    4: {"name": "Family Foods", "format": "Supermarket", "size": "Medium"},
    5: {"name": "Urban Pantry", "format": "Convenience", "size": "Small"},
    6: {"name": "Metro Grocers", "format": "Hypermarket", "size": "Large"},
    7: {"name": "Quick Stop", "format": "Convenience", "size": "Small"},
    8: {"name": "Value Market", "format": "Discount", "size": "Medium"},
    9: {"name": "Fresh & Fast", "format": "Convenience", "size": "Small"}
};

const categoryMappings = {
    first: {
        0: "Fresh Food",
        1: "Packaged Goods",
        2: "Beverages",
        3: "Household",
        4: "Personal Care",
        5: "Frozen Foods"
    },
    second: {
        0: "Produce",
        1: "Dairy",
        2: "Meat",
        3: "Bakery",
        4: "Snacks",
        5: "Canned Goods",
        6: "Soft Drinks",
        7: "Water",
        8: "Alcoholic Beverages"
    },
    third: {
        0: "Fruits",
        1: "Vegetables",
        2: "Milk",
        3: "Cheese",
        4: "Beef",
        5: "Poultry",
        6: "Bread",
        7: "Pastries"
    }
};

// Utility functions
function getCityName(cityId) {
    return cityMapping[cityId]?.name || `City ${cityId}`;
}

function getCityWithState(cityId) {
    const city = cityMapping[cityId];
    return city ? `${city.name}, ${city.state}` : `City ${cityId}`;
}

function getStoreName(storeId) {
    return storeMapping[storeId]?.name || `Store ${storeId}`;
}

function getStoreFullInfo(storeId) {
    const store = storeMapping[storeId];
    return store ? `${store.name} (${store.format}, ${store.size})` : `Store ${storeId}`;
}

function getCategoryName(categoryId, level = 'first') {
    return categoryMappings[level][categoryId] || `Category ${categoryId}`;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDatepickers();
    populateDropdowns();
    attachEventListeners();
});

// Initialize navigation
function initNavigation() {
    elements.navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = link.getAttribute('data-page');
            
            // Update active nav link
            elements.navLinks.forEach(navLink => navLink.classList.remove('active'));
            link.classList.add('active');
            
            // Show target page, hide others
            elements.pages.forEach(page => {
                page.classList.remove('active');
                if (page.id === `${targetPage}-page`) {
                    page.classList.add('active');
                }
            });
        });
    });
}

// Initialize datepickers
function initDatepickers() {
    document.querySelectorAll('.datepicker').forEach(input => {
        flatpickr(input, {
            dateFormat: "Y-m-d",
            defaultDate: new Date()
        });
    });
}

// Populate dropdowns with real data
function populateDropdowns() {
    // Populate city dropdowns
    const citySelects = document.querySelectorAll('.city-select');
    citySelects.forEach(select => {
        select.innerHTML = '';
        Object.entries(cityMapping).forEach(([id, city]) => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${city.name}, ${city.state}`;
            select.appendChild(option);
        });
    });
    
    // Populate store dropdowns
    const storeSelects = document.querySelectorAll('.store-select');
    storeSelects.forEach(select => {
        select.innerHTML = '';
        Object.entries(storeMapping).forEach(([id, store]) => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${store.name} (${store.format})`;
            select.appendChild(option);
        });
    });
    
    // Convert existing number inputs to selects where appropriate
    convertInputsToSelects();
}

// Convert numeric inputs to select dropdowns for better UX
function convertInputsToSelects() {
    // Replace store ID inputs with select dropdowns
    replaceInputWithSelect('store-id', storeMapping, (id, store) => `${store.name} (${store.format})`);
    replaceInputWithSelect('promo-store-id', storeMapping, (id, store) => `${store.name} (${store.format})`);
    replaceInputWithSelect('rec-store-id', storeMapping, (id, store) => `${store.name} (${store.format})`);
    replaceInputWithSelect('stockout-store-id', storeMapping, (id, store) => `${store.name} (${store.format})`);
    replaceInputWithSelect('add-promo-store-id', storeMapping, (id, store) => `${store.name} (${store.format})`);
    
    // Replace city ID inputs with select dropdowns
    replaceInputWithSelect('city-id', cityMapping, (id, city) => `${city.name}, ${city.state}`);
}

function replaceInputWithSelect(inputId, mapping, labelFormatter) {
    const input = document.getElementById(inputId);
    if (!input) return;
    
    const parent = input.parentNode;
    const label = parent.querySelector('label');
    
    // Create select element
    const select = document.createElement('select');
    select.id = inputId;
    select.className = input.className + ' form-select';
    
    // Create options
    Object.entries(mapping).forEach(([id, data]) => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = labelFormatter(id, data);
        select.appendChild(option);
    });
    
    // Update label text to remove "ID"
    if (label) {
        label.textContent = label.textContent.replace(' ID', '');
    }
    
    // Replace input with select
    parent.replaceChild(select, input);
}

// Attach event listeners
function attachEventListeners() {
    // Forecast form submission
    if (elements.forecastForm) {
        elements.forecastForm.addEventListener('submit', handleForecastSubmit);
    }
    
    // Promotion analysis form submission
    if (elements.promoAnalysisForm) {
        elements.promoAnalysisForm.addEventListener('submit', handlePromoAnalysisSubmit);
    }
    
    // Promotion recommendation form submission
    if (elements.promoRecommendForm) {
        elements.promoRecommendForm.addEventListener('submit', handlePromoRecommendSubmit);
    }
    
    // Stockout form submission
    if (elements.stockoutForm) {
        elements.stockoutForm.addEventListener('submit', handleStockoutSubmit);
    }
    
    // Holiday form submission
    if (elements.holidayForm) {
        elements.holidayForm.addEventListener('submit', handleHolidaySubmit);
    }
    
    // Add promotion form
    const savePromoBtn = document.getElementById('save-promo-btn');
    if (savePromoBtn) {
        savePromoBtn.addEventListener('click', handleAddPromotion);
    }
}

// API interaction functions

// Forecast form handler
async function handleForecastSubmit(e) {
    e.preventDefault();
    
    const storeId = document.getElementById('store-id').value;
    const cityId = document.getElementById('city-id').value;
    const productId = document.getElementById('product-id').value;
    const categoryId = document.getElementById('category-id').value;
    const startDate = document.getElementById('start-date').value;
    const forecastPeriods = document.getElementById('forecast-periods').value;
    const forecastFreq = document.getElementById('forecast-freq').value;
    const includeWeather = document.getElementById('include-weather').checked;
    const includeHolidays = document.getElementById('include-holidays').checked;
    const includePromotions = document.getElementById('include-promotions').checked;
    
    try {
        // Show loading state
        
        // Make API call to get forecast
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                store_id: parseInt(storeId),
                city_id: parseInt(cityId),
                product_id: parseInt(productId),
                category_id: parseInt(categoryId),
                start_date: startDate,
                periods: parseInt(forecastPeriods),
                freq: forecastFreq,
                include_weather: includeWeather,
                include_holidays: includeHolidays,
                include_promotions: includePromotions
            })
        });
        
        if (!response.ok) {
            throw new Error('Forecast API call failed');
        }
        
        const data = await response.json();
        
        // Display forecast results
        displayForecastResults(data, {
            storeId,
            cityId,
            productId,
            categoryId
        });
    } catch (error) {
        console.error('Error generating forecast:', error);
        alert('Failed to generate forecast. Please try again.');
    }
}

// Display forecast results
function displayForecastResults(data, params) {
    // Show results container
    elements.forecastResults.classList.remove('d-none');
    
    // Format data for display
    const dates = data.forecast.map(item => item.ds);
    const forecasts = data.forecast.map(item => item.yhat);
    const lowerBounds = data.forecast.map(item => item.yhat_lower);
    const upperBounds = data.forecast.map(item => item.yhat_upper);
    
    // Update chart
    const ctx = elements.forecastChart.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Forecast',
                    data: forecasts,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Lower Bound',
                    data: lowerBounds,
                    borderColor: 'rgba(54, 162, 235, 0.3)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0
                },
                {
                    label: 'Upper Bound',
                    data: upperBounds,
                    borderColor: 'rgba(54, 162, 235, 0.3)',
                    backgroundColor: 'rgba(54, 162, 235, 0.05)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: '+1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Sales Forecast for ${getStoreName(params.storeId)} in ${getCityName(params.cityId)}`,
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sales Amount'
                    },
                    beginAtZero: true
                }
            }
        }
    });
    
    // Populate table
    const tbody = elements.forecastTable.querySelector('tbody');
    tbody.innerHTML = '';
    
    data.forecast.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.ds}</td>
            <td>${item.yhat.toFixed(2)}</td>
            <td>${item.yhat_lower.toFixed(2)}</td>
            <td>${item.yhat_upper.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}

// Implement other form handlers (handlePromoAnalysisSubmit, handlePromoRecommendSubmit, etc.) in a similar pattern
async function handlePromoAnalysisSubmit(e) {
    e.preventDefault();
    // Implementation similar to forecast handler
    console.log('Promotion analysis submitted');
}

async function handlePromoRecommendSubmit(e) {
    e.preventDefault();
    // Implementation similar to forecast handler
    console.log('Promotion recommendation submitted');
}

async function handleStockoutSubmit(e) {
    e.preventDefault();
    // Implementation similar to forecast handler
    console.log('Stockout analysis submitted');
}

async function handleHolidaySubmit(e) {
    e.preventDefault();
    // Implementation similar to forecast handler
    console.log('Holiday analysis submitted');
}

async function handleAddPromotion() {
    // Implementation for adding a new promotion
    console.log('Add promotion submitted');
} 