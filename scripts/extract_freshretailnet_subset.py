import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import os
import argparse

# US names for mapping (presentation only)
US_CITIES = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio',
    'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus',
    'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle'
]
US_STORES = [
    'Walmart Supercenter', 'Target', 'Kroger', 'Costco', 'Safeway', 'Publix', 'H-E-B', 'Albertsons',
    'Tom Thumb', 'Whole Foods', 'Randalls', 'Winn-Dixie', 'Meijer', 'Harris Teeter', "Trader Joe's",
    'QFC', 'Giant Eagle', 'Hy-Vee', 'Food Lion', 'Piggly Wiggly', 'Aldi', 'Sam''s Club', 'Sprouts',
    'Food 4 Less', 'WinCo Foods'
]
US_PRODUCTS = [
    'Milk, Whole, 1 Gallon', 'Eggs, Large, Dozen', 'Bread, White, Loaf', 'Bananas, 1 lb', 'Chicken Breast, 1 lb',
    'Ground Beef, 1 lb', 'Apples, 1 lb', 'Orange Juice, 1 qt', 'Yogurt, Plain, 32 oz', 'Butter, Salted, 1 lb',
    'Cheddar Cheese, 8 oz', 'Broccoli, 1 lb', 'Carrots, 1 lb', 'Potatoes, 5 lb', 'Onions, 3 lb', 'Tomatoes, 1 lb',
    'Lettuce, Romaine', 'Spinach, 8 oz', 'Strawberries, 1 lb', 'Blueberries, 6 oz', 'Grapes, 1 lb', 'Oranges, 1 lb',
    'Pasta, Spaghetti, 1 lb', 'Rice, White, 2 lb', 'Cereal, Oats, 18 oz', 'Coca-Cola, 12 Pack', 'Peanut Butter, 16 oz',
    'Jelly, Grape, 18 oz', 'Mayonnaise, 30 oz', 'Ketchup, 20 oz', 'Mustard, 14 oz', 'Chicken Thighs, 1 lb',
    'Turkey Breast, 1 lb', 'Bacon, 12 oz', 'Sausage, 16 oz', 'Ham, Sliced, 1 lb', 'Salmon Fillet, 1 lb',
    'Tilapia Fillet, 1 lb', 'Shrimp, 1 lb', 'Tuna, Canned, 5 oz', 'Sardines, Canned, 3.75 oz', 'Almonds, 8 oz',
    'Walnuts, 8 oz', 'Cashews, 8 oz', 'Peanuts, 16 oz', 'Sunflower Seeds, 8 oz', 'Pumpkin Seeds, 8 oz',
    'Raisins, 12 oz', 'Dried Cranberries, 6 oz', 'Granola Bars, 6 ct', 'Oatmeal, Instant, 10 ct', 'Bagels, 6 ct',
    'English Muffins, 6 ct', 'Tortillas, Flour, 10 ct', 'Tortillas, Corn, 10 ct', 'Pita Bread, 6 ct',
    'Hamburger Buns, 8 ct', 'Hot Dog Buns, 8 ct', 'Croissants, 4 ct', 'Danish, 4 ct', 'Muffins, Blueberry, 4 ct',
    'Donuts, Glazed, 6 ct', 'Coffee, Ground, 12 oz', 'Tea Bags, 20 ct', 'Hot Chocolate Mix, 10 ct',
    'Sugar, Granulated, 4 lb', 'Brown Sugar, 2 lb', 'Powdered Sugar, 2 lb', 'Honey, 12 oz', 'Maple Syrup, 12 oz',
    'Pancake Mix, 32 oz', 'Waffles, Frozen, 10 ct', 'French Toast, Frozen, 6 ct', 'Cinnamon Rolls, 8 ct',
    'Butter, Unsalted, 1 lb', 'Cream Cheese, 8 oz', 'Sour Cream, 16 oz', 'Cottage Cheese, 16 oz',
    'Mozzarella Cheese, 8 oz', 'Swiss Cheese, 8 oz', 'Provolone Cheese, 8 oz', 'Parmesan Cheese, 8 oz',
    'Colby Jack Cheese, 8 oz', 'American Cheese, 8 oz', 'Ice Cream, Vanilla, 1.5 qt', 'Ice Cream, Chocolate, 1.5 qt',
    'Frozen Yogurt, 1.5 qt', 'Popsicles, 12 ct', 'Frozen Pizza, 12 in', 'Frozen Vegetables, Mixed, 16 oz',
    'Frozen Peas, 16 oz', 'Frozen Corn, 16 oz', 'Frozen Green Beans, 16 oz', 'Frozen Spinach, 10 oz',
    'Frozen Berries, 12 oz', 'Frozen French Fries, 32 oz', 'Frozen Hash Browns, 32 oz', 'Frozen Chicken Nuggets, 25 oz',
    'Frozen Fish Sticks, 18 oz', 'Frozen Meatballs, 14 oz', 'Frozen Lasagna, 32 oz', 'Frozen Burritos, 8 ct',
    'Frozen Breakfast Sandwiches, 4 ct', 'Frozen Waffles, 10 ct', 'Frozen Pancakes, 12 ct', 'Frozen Pie, Apple, 37 oz',
    'Frozen Pie, Pumpkin, 37 oz', 'Frozen Pie, Cherry, 37 oz', 'Frozen Pie, Pecan, 37 oz', 'Frozen Pie, Blueberry, 37 oz',
    'Apple Sauce, 24 oz', 'Fruit Cups, Mixed, 4 ct', 'Mandarin Oranges, Canned, 15 oz', 'Pineapple, Canned, 20 oz',
    'Peaches, Canned, 15 oz', 'Pears, Canned, 15 oz', 'Fruit Cocktail, Canned, 15 oz', 'Tomato Sauce, 15 oz',
    'Tomato Paste, 6 oz', 'Diced Tomatoes, 14.5 oz', 'Stewed Tomatoes, 14.5 oz', 'Crushed Tomatoes, 28 oz',
    'Spaghetti Sauce, 24 oz', 'Alfredo Sauce, 15 oz', 'Pizza Sauce, 14 oz', 'Salsa, 16 oz', 'Guacamole, 8 oz',
    'Hummus, 10 oz', 'Pickles, Dill, 24 oz', 'Olives, Black, 6 oz', 'Olives, Green, 6 oz', 'Capers, 3.5 oz',
    'Artichoke Hearts, 14 oz', 'Hearts of Palm, 14 oz', 'Roasted Red Peppers, 12 oz', 'Mushrooms, Sliced, 8 oz',
    'Green Beans, Canned, 14.5 oz', 'Corn, Canned, 15 oz', 'Peas, Canned, 15 oz', 'Carrots, Canned, 14.5 oz',
    'Potatoes, Canned, 15 oz', 'Baked Beans, 16 oz', 'Chili, Canned, 15 oz', 'Soup, Chicken Noodle, 10.5 oz',
    'Soup, Tomato, 10.75 oz', 'Soup, Vegetable, 10.5 oz', 'Soup, Cream of Mushroom, 10.5 oz',
    'Soup, Clam Chowder, 15 oz', 'Soup, Beef Stew, 20 oz', 'Soup, Lentil, 15 oz', 'Soup, Minestrone, 15 oz',
    'Soup, Split Pea, 15 oz', 'Soup, Black Bean, 15 oz', 'Soup, Chicken Tortilla, 15 oz', 'Soup, Broccoli Cheddar, 15 oz',
    'Soup, Potato Leek, 15 oz', 'Soup, Italian Wedding, 15 oz', 'Soup, French Onion, 15 oz', 'Soup, Chicken Rice, 15 oz',
    'Soup, Vegetable Beef, 15 oz', 'Soup, Cream of Chicken, 10.5 oz', 'Soup, Cream of Celery, 10.5 oz',
    'Soup, Cream of Broccoli, 10.5 oz', 'Soup, Cream of Potato, 10.5 oz', 'Soup, Cream of Asparagus, 10.5 oz',
    'Soup, Cream of Shrimp, 10.5 oz', 'Soup, Cream of Tomato, 10.5 oz', 'Soup, Cream of Onion, 10.5 oz',
    'Soup, Cream of Carrot, 10.5 oz', 'Soup, Cream of Cauliflower, 10.5 oz', 'Soup, Cream of Spinach, 10.5 oz',
    'Soup, Cream of Pea, 10.5 oz', 'Soup, Cream of Corn, 10.5 oz', 'Soup, Cream of Mushroom, 10.5 oz',
    'Soup, Cream of Chicken, 10.5 oz', 'Soup, Cream of Celery, 10.5 oz', 'Soup, Cream of Broccoli, 10.5 oz',
    'Soup, Cream of Potato, 10.5 oz', 'Soup, Cream of Asparagus, 10.5 oz', 'Soup, Cream of Shrimp, 10.5 oz',
    'Soup, Cream of Tomato, 10.5 oz', 'Soup, Cream of Onion, 10.5 oz', 'Soup, Cream of Carrot, 10.5 oz',
    'Soup, Cream of Cauliflower, 10.5 oz', 'Soup, Cream of Spinach, 10.5 oz', 'Soup, Cream of Pea, 10.5 oz',
    'Soup, Cream of Corn, 10.5 oz', 'Soup, Cream of Mushroom, 10.5 oz', 'Soup, Cream of Chicken, 10.5 oz',
    'Soup, Cream of Celery, 10.5 oz', 'Soup, Cream of Broccoli, 10.5 oz', 'Soup, Cream of Potato, 10.5 oz',
    'Soup, Cream of Asparagus, 10.5 oz', 'Soup, Cream of Shrimp, 10.5 oz', 'Soup, Cream of Tomato, 10.5 oz',
    'Soup, Cream of Onion, 10.5 oz', 'Soup, Cream of Carrot, 10.5 oz', 'Soup, Cream of Cauliflower, 10.5 oz',
    'Soup, Cream of Spinach, 10.5 oz', 'Soup, Cream of Pea, 10.5 oz',
    # Added 49 more real daily-use products to reach 250
    'Toothpaste, 6 oz', 'Toothbrush, Soft', 'Shampoo, 12 oz', 'Conditioner, 12 oz', 'Bar Soap, 4 oz',
    'Body Wash, 16 oz', 'Deodorant, 2.6 oz', 'Razor, Disposable, 5 ct', 'Shaving Cream, 7 oz', 'Facial Tissue, 120 ct',
    'Paper Towels, 2 Rolls', 'Toilet Paper, 4 Rolls', 'Laundry Detergent, 50 oz', 'Fabric Softener, 32 oz', 'Dish Soap, 16 oz',
    'Sponges, 3 ct', 'Trash Bags, 13 gal, 20 ct', 'Aluminum Foil, 75 ft', 'Plastic Wrap, 200 ft', 'Sandwich Bags, 50 ct',
    'Freezer Bags, 20 ct', 'Storage Containers, 3 ct', 'Light Bulbs, 4 pk', 'Batteries, AA, 8 pk', 'Batteries, AAA, 8 pk',
    'Hand Sanitizer, 8 oz', 'Disinfecting Wipes, 35 ct', 'All-Purpose Cleaner, 32 oz', 'Glass Cleaner, 32 oz', 'Floor Cleaner, 32 oz',
    'Mop, Standard', 'Broom, Standard', 'Dustpan, Standard', 'Plunger, Standard', 'Toilet Bowl Cleaner, 24 oz',
    'Air Freshener, 8 oz', 'Candle, Scented, 3 oz', 'Matches, 32 ct', 'Lighter, Utility', 'Bandages, 30 ct',
    'Antibacterial Ointment, 1 oz', 'Pain Reliever, 24 ct', 'Cough Drops, 30 ct', 'Thermometer, Digital', 'Cotton Swabs, 170 ct',
    'Cotton Balls, 100 ct', 'Nail Clippers, Standard', 'Tweezers, Standard', 'Comb, Standard', 'Hair Brush, Standard'
]


def log(msg):
    print(f"[INFO] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Extract real subset from FreshRetailNet-50K for Supabase import.")
    parser.add_argument('--output_dir', type=str, default='data_export', help='Directory to save CSVs')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log("Loading FreshRetailNet-50K dataset from Hugging Face...")
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    # If ds is a DatasetDict, use the 'train' split
    if isinstance(ds, dict) and 'train' in ds:
        ds = ds['train']
    if isinstance(ds, Dataset):
        log(f"Loaded {len(ds)} rows.")
        df = ds.to_pandas()

    # Get unique IDs from the dataset as numpy arrays
    log("Extracting unique city, store, and product IDs from dataset...")
    unique_city_ids = np.array(pd.unique(df['city_id']))
    unique_store_ids = np.array(pd.unique(df['store_id']))
    unique_product_ids = np.array(pd.unique(df['product_id']))

    # Select exactly 18 cities, 25 stores, 250 products (no repeats, all must exist in the data)
    selected_city_ids = unique_city_ids[:18]
    selected_store_ids = unique_store_ids[:25]
    selected_product_ids = unique_product_ids[:250]

    # Create mapping to US names (for presentation only)
    log(f"Selected {len(selected_product_ids)} products, US_PRODUCTS has {len(US_PRODUCTS)} items.")
    if len(selected_product_ids) != len(US_PRODUCTS):
        raise ValueError(f"Number of selected products ({len(selected_product_ids)}) does not match US_PRODUCTS list ({len(US_PRODUCTS)}).")
    city_id_to_us_name = {cid: US_CITIES[i] for i, cid in enumerate(selected_city_ids)}
    store_id_to_us_name = {sid: US_STORES[i] for i, sid in enumerate(selected_store_ids)}
    product_id_to_us_name = {pid: US_PRODUCTS[i] for i, pid in enumerate(selected_product_ids)}

    # Filter DataFrame for selected IDs (convert numpy arrays to lists for isin)
    log("Filtering sales data for selected cities, stores, and products...")
    df_filtered = df[
        df['city_id'].isin(selected_city_ids.tolist()) &
        df['store_id'].isin(selected_store_ids.tolist()) &
        df['product_id'].isin(selected_product_ids.tolist())
    ].copy()
    log(f"Filtered to {len(df_filtered)} sales records.")

    # --- city_hierarchy.csv ---
    log("Preparing city_hierarchy.csv...")
    city_hierarchy = pd.DataFrame({
        'city_id': selected_city_ids,
        'city_name': [city_id_to_us_name[cid] for cid in selected_city_ids],
        'original_city_id': selected_city_ids
    })
    city_hierarchy.to_csv(os.path.join(args.output_dir, 'city_hierarchy.csv'), index=False)
    log("city_hierarchy.csv written.")

    # --- store_hierarchy.csv ---
    log("Preparing store_hierarchy.csv...")
    # For each store, get its city_id from the first matching row in the filtered data
    store_city_map = df_filtered.drop_duplicates(subset='store_id').set_index('store_id')['city_id'].to_dict()
    store_hierarchy = pd.DataFrame({
        'store_id': selected_store_ids,
        'store_name': [store_id_to_us_name[sid] for sid in selected_store_ids],
        'original_store_id': selected_store_ids,
        'city_id': [store_city_map.get(sid, None) for sid in selected_store_ids]
    })
    store_hierarchy.to_csv(os.path.join(args.output_dir, 'store_hierarchy.csv'), index=False)
    log("store_hierarchy.csv written.")

    # --- product_hierarchy.csv ---
    log("Preparing product_hierarchy.csv...")
    # For each product, get its category fields from the first matching row in the filtered data
    prod_fields = ['management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
    prod_info = df_filtered.drop_duplicates(subset='product_id').set_index('product_id')[prod_fields].to_dict('index')
    product_hierarchy = pd.DataFrame({
        'product_id': selected_product_ids,
        'product_name': [product_id_to_us_name[pid] for pid in selected_product_ids],
        'original_product_id': selected_product_ids,
        'management_group_id': [prod_info[pid]['management_group_id'] for pid in selected_product_ids],
        'first_category_id': [prod_info[pid]['first_category_id'] for pid in selected_product_ids],
        'second_category_id': [prod_info[pid]['second_category_id'] for pid in selected_product_ids],
        'third_category_id': [prod_info[pid]['third_category_id'] for pid in selected_product_ids],
    })
    product_hierarchy.to_csv(os.path.join(args.output_dir, 'product_hierarchy.csv'), index=False)
    log("product_hierarchy.csv written.")

    # --- sales_data.csv ---
    log("Writing sales_data.csv (all columns, only for selected IDs)...")
    df_filtered.to_csv(os.path.join(args.output_dir, 'sales_data.csv'), index=False)
    log("sales_data.csv written.")

    log("All done!")

if __name__ == "__main__":
    main()