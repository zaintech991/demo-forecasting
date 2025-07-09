"""
Mappings for encoded values to real-world names and transformations in the FreshRetailNet-50K dataset.
"""
import json
import numpy as np

# Sales Transformation Constants
SALES_MULTIPLIERS = {
    # Management group specific multipliers (based on product type)
    0: 25.0,  # Fresh Produce: ~¥25 per unit
    1: 15.0,  # Dairy & Eggs: ~¥15 per unit
    2: 45.0,  # Meat & Seafood: ~¥45 per unit
    3: 20.0,  # Bakery: ~¥20 per unit
    4: 10.0,  # Beverages: ~¥10 per unit
    5: 30.0,  # Packaged Foods: ~¥30 per unit
    6: 35.0,  # Ready-to-Eat: ~¥35 per unit
}

# Base multiplier for converting encoded sales to real values
BASE_SALES_MULTIPLIER = 20.0  # Base price of ¥20

# City-specific price adjustments (based on local economy)
CITY_PRICE_ADJUSTMENTS = {
    0: 1.2,   # Shanghai (20% higher)
    1: 1.15,  # Guangzhou
    2: 1.15,  # Shenzhen
    3: 1.2,   # Beijing
    4: 1.0,   # Chengdu
    5: 1.1,   # Hangzhou
    6: 1.1,   # Nanjing
    7: 1.05,  # Xiamen
    8: 0.9,   # Urumqi
    9: 1.1,   # Sanya
    10: 1.0,  # Tianjin
    11: 1.0,  # Wuhan
    12: 1.0,  # Chongqing
    13: 1.1,  # Suzhou
    14: 1.05, # Ningbo
    15: 0.95, # Hefei
    16: 1.0,  # Xi'an
    17: 0.95  # Zhengzhou
}

def decode_sales_amount(encoded_amount, city_id, management_group_id):
    """Convert encoded sales amount to real-world price in CNY."""
    base_price = encoded_amount * BASE_SALES_MULTIPLIER
    city_adjustment = CITY_PRICE_ADJUSTMENTS.get(city_id, 1.0)
    category_multiplier = SALES_MULTIPLIERS.get(management_group_id, BASE_SALES_MULTIPLIER)
    
    return base_price * city_adjustment * (category_multiplier / BASE_SALES_MULTIPLIER)

def encode_sales_amount(real_amount, city_id, management_group_id):
    """Convert real-world price back to encoded value."""
    city_adjustment = CITY_PRICE_ADJUSTMENTS.get(city_id, 1.0)
    category_multiplier = SALES_MULTIPLIERS.get(management_group_id, BASE_SALES_MULTIPLIER)
    
    return real_amount / (BASE_SALES_MULTIPLIER * city_adjustment * (category_multiplier / BASE_SALES_MULTIPLIER))

def decode_hourly_sales(hours_sale_str, city_id, management_group_id):
    """Convert encoded hourly sales to real-world values."""
    try:
        hours_sale = json.loads(hours_sale_str)
        return [decode_sales_amount(x, city_id, management_group_id) for x in hours_sale]
    except (json.JSONDecodeError, TypeError):
        return []

def encode_hourly_sales(real_hours_sale, city_id, management_group_id):
    """Convert real-world hourly sales back to encoded values."""
    return [encode_sales_amount(x, city_id, management_group_id) for x in real_hours_sale]

# City Mappings (based on climate patterns and store counts)
CITY_MAPPINGS = {
    0: "Shanghai",          # Large city, moderate temp, high humidity
    1: "Guangzhou",         # Smaller presence, high temp, very high humidity
    2: "Shenzhen",         # Small presence, high temp, high humidity
    3: "Beijing",          # Large presence, moderate temp, low humidity
    4: "Chengdu",          # Medium presence, moderate temp, high humidity
    5: "Hangzhou",         # Small presence, moderate temp, high humidity
    6: "Nanjing",          # Medium presence, moderate temp, very high humidity
    7: "Xiamen",           # Small presence, high temp, high humidity
    8: "Urumqi",           # Minimal presence, moderate temp, very low humidity
    9: "Sanya",            # Minimal presence, high temp, very high humidity
    10: "Tianjin",         # Minimal presence, moderate temp, moderate humidity
    11: "Wuhan",           # Medium presence, moderate temp, high humidity
    12: "Chongqing",       # Large presence, moderate temp, high humidity
    13: "Suzhou",          # Large presence, high temp, very high humidity
    14: "Ningbo",          # Small presence, moderate temp, very high humidity
    15: "Hefei",           # Small presence, moderate temp, high humidity
    16: "Xi'an",           # Large presence, moderate temp, high humidity
    17: "Zhengzhou"        # Minimal presence, moderate temp, high humidity
}

# Management Group Mappings (based on product characteristics)
MANAGEMENT_GROUP_MAPPINGS = {
    0: "Fresh Produce",
    1: "Dairy & Eggs",
    2: "Meat & Seafood",
    3: "Bakery",
    4: "Beverages",
    5: "Packaged Foods",
    6: "Ready-to-Eat"
}

# First Category Mappings (within management groups)
FIRST_CATEGORY_MAPPINGS = {
    # Fresh Produce (0)
    5: "Vegetables",
    26: "Fruits",
    28: "Fresh Herbs",
    
    # Dairy & Eggs (1)
    3: "Milk",
    7: "Eggs",
    
    # Meat & Seafood (2)
    0: "Poultry",
    2: "Beef",
    29: "Fish",
    30: "Shellfish",
    31: "Pork",
    
    # Bakery (3)
    6: "Bread",
    9: "Pastries",
    11: "Cakes",
    12: "Cookies",
    14: "Buns",
    17: "Rolls",
    25: "Specialty Breads",
    
    # Beverages (4)
    1: "Water",
    13: "Tea",
    23: "Coffee",
    27: "Soft Drinks",
    
    # Packaged Foods (5)
    15: "Snacks",
    16: "Instant Meals",
    18: "Condiments",
    19: "Spices",
    22: "Dried Foods",
    
    # Ready-to-Eat (6)
    4: "Hot Foods",
    8: "Salads",
    10: "Sandwiches",
    20: "Sushi",
    21: "Premium Meals",
    24: "Bento Boxes"
}

# Store Type Mappings (based on store patterns and sales)
STORE_TYPE_MAPPINGS = {
    "PREMIUM": "Premium Store",      # High avg_sale_amount (> 1.2)
    "STANDARD": "Standard Store",    # Medium avg_sale_amount (0.8-1.2)
    "EXPRESS": "Express Store",      # Low avg_sale_amount (< 0.8)
}

def get_store_type(avg_sale_amount):
    """Determine store type based on average sale amount."""
    if avg_sale_amount > 1.2:
        return STORE_TYPE_MAPPINGS["PREMIUM"]
    elif avg_sale_amount < 0.8:
        return STORE_TYPE_MAPPINGS["EXPRESS"]
    else:
        return STORE_TYPE_MAPPINGS["STANDARD"]

def get_store_name(city_id, store_id, avg_sale_amount):
    """Generate a store name based on city, store ID, and type."""
    city_name = CITY_MAPPINGS[city_id]
    store_type = get_store_type(avg_sale_amount)
    
    if store_type == STORE_TYPE_MAPPINGS["PREMIUM"]:
        prefix = "Premium"
    elif store_type == STORE_TYPE_MAPPINGS["EXPRESS"]:
        prefix = "Express"
    else:
        prefix = "Market"
    
    # Generate district name based on store_id
    district_number = (store_id % 10) + 1
    
    return f"{city_name} {prefix} District {district_number}"

def get_product_name(management_group_id, first_category_id, product_id):
    """Generate a product name based on its hierarchy."""
    try:
        category = FIRST_CATEGORY_MAPPINGS[first_category_id]
        group = MANAGEMENT_GROUP_MAPPINGS[management_group_id]
        
        # Generate a specific product name based on the category
        product_number = product_id % 100 + 1
        return f"{category} #{product_number}"
    except KeyError:
        return f"Product {product_id}" 