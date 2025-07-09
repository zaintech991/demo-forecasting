"""
This module provides mapping dictionaries to translate encoded IDs to real-world values.

These are used to make the data more interpretable in the UI and reports.
"""

# Map city IDs to real US cities
CITY_MAPPING = {
    0: {"name": "New York", "state": "NY", "region": "Northeast"},
    1: {"name": "Los Angeles", "state": "CA", "region": "West"},
    2: {"name": "Chicago", "state": "IL", "region": "Midwest"},
    3: {"name": "Houston", "state": "TX", "region": "South"},
    4: {"name": "Phoenix", "state": "AZ", "region": "West"},
    5: {"name": "Philadelphia", "state": "PA", "region": "Northeast"},
    6: {"name": "San Antonio", "state": "TX", "region": "South"},
    7: {"name": "San Diego", "state": "CA", "region": "West"},
    8: {"name": "Dallas", "state": "TX", "region": "South"},
    9: {"name": "San Jose", "state": "CA", "region": "West"},
    10: {"name": "Austin", "state": "TX", "region": "South"},
    11: {"name": "Jacksonville", "state": "FL", "region": "South"},
    12: {"name": "Fort Worth", "state": "TX", "region": "South"},
    13: {"name": "Columbus", "state": "OH", "region": "Midwest"},
    14: {"name": "Indianapolis", "state": "IN", "region": "Midwest"},
    15: {"name": "Charlotte", "state": "NC", "region": "South"},
    16: {"name": "San Francisco", "state": "CA", "region": "West"},
    17: {"name": "Seattle", "state": "WA", "region": "West"}
}

# Map store IDs to realistic store names and metadata
STORE_MAPPING = {
    0: {"name": "Downtown Market", "format": "Supermarket", "size": "Large"},
    1: {"name": "Westside Fresh", "format": "Supermarket", "size": "Medium"},
    2: {"name": "Northgate Grocery", "format": "Supermarket", "size": "Large"},
    3: {"name": "Eastside Express", "format": "Convenience", "size": "Small"},
    4: {"name": "Central Mart", "format": "Hypermarket", "size": "Extra Large"},
    5: {"name": "Riverside Foods", "format": "Supermarket", "size": "Medium"},
    6: {"name": "Hilltop Market", "format": "Supermarket", "size": "Large"},
    7: {"name": "Valley Fresh", "format": "Supermarket", "size": "Medium"},
    8: {"name": "Plaza Grocery", "format": "Supermarket", "size": "Large"},
    9: {"name": "Southside Express", "format": "Convenience", "size": "Small"},
    10: {"name": "Metro Mart", "format": "Hypermarket", "size": "Extra Large"},
    11: {"name": "Harbor View Market", "format": "Supermarket", "size": "Medium"},
    12: {"name": "University Foods", "format": "Supermarket", "size": "Small"},
    13: {"name": "Parkside Grocery", "format": "Supermarket", "size": "Medium"},
    14: {"name": "Industrial District Market", "format": "Warehouse", "size": "Large"},
    15: {"name": "Suburban Fresh", "format": "Supermarket", "size": "Medium"},
    16: {"name": "Airport Express", "format": "Convenience", "size": "Small"},
    17: {"name": "Business District Mart", "format": "Supermarket", "size": "Medium"},
    18: {"name": "Mall Market", "format": "Supermarket", "size": "Small"},
    19: {"name": "Beach Road Foods", "format": "Supermarket", "size": "Medium"}
}

# Mapping for product categories (first level)
FIRST_CATEGORY_MAPPING = {
    0: "Produce",
    1: "Dairy",
    2: "Meat & Seafood",
    3: "Bakery",
    4: "Deli",
    5: "Snacks & Beverages",
    6: "Frozen Foods",
    7: "Canned Goods",
    8: "Dry Goods & Pasta",
    9: "Household & Cleaning",
    10: "Personal Care",
    11: "Health & Wellness",
    12: "Pet Supplies"
}

# Example mapping for management groups
MANAGEMENT_GROUP_MAPPING = {
    0: "Northeast Division",
    1: "Southeast Division",
    2: "Midwest Division",
    3: "Southwest Division",
    4: "Western Division",
    5: "National Operations"
}

# Convert numeric to percent discount for better UI display
def discount_to_percent(discount_rate):
    """Convert discount rate (where 1.0 is no discount) to percentage off"""
    if discount_rate >= 1.0:
        return "0%"
    
    percent_off = int(round((1.0 - discount_rate) * 100))
    return f"{percent_off}%" 