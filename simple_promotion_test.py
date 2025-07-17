#!/usr/bin/env python3
"""
Simple test script to debug promotion API issues
"""

import requests
import json
import time

def test_single_promotion(discount_pct, duration_days):
    """Test a single promotion combination"""
    
    url = "http://localhost:8000/api/weather-holiday-forecast"
    
    # Test data
    data = {
        "city_ids": ["1", "2"],  # Los Angeles, Chicago
        "store_ids": ["1", "2"],  # First two stores
        "product_ids": [1, 2],   # First two products
        "discount_percentage": discount_pct / 100,  # Convert to decimal
        "promotion_duration_days": duration_days
    }
    
    try:
        print(f"\nTesting {discount_pct}% discount for {duration_days} days...")
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we have promotional analysis
            if 'promotional_analysis' in result:
                print(f"Found {len(result['promotional_analysis'])} promotional analyses")
                
                for i, analysis in enumerate(result['promotional_analysis']):
                    print(f"\nAnalysis {i+1}:")
                    print(f"  Product: {analysis.get('product_name', 'Unknown')} (ID: {analysis.get('product_id')})")
                    print(f"  City: {analysis.get('city_name', 'Unknown')} (ID: {analysis.get('city_id')})")
                    print(f"  Recommendation: {analysis.get('recommendation', 'N/A')}")
                    print(f"  Revenue Impact: ${analysis.get('revenue_impact', 0):.2f}")
                    print(f"  Sales Increase: {analysis.get('sales_increase_percentage', 0):.1f}%")
                    
                    # Check if it's a positive recommendation
                    rec = analysis.get('recommendation', '')
                    if 'RECOMMENDED' in rec and 'NOT RECOMMENDED' not in rec:
                        print(f"  ✅ POSITIVE RECOMMENDATION FOUND!")
                        return True
                        
            else:
                print("No promotional_analysis found in response")
                print(f"Response keys: {list(result.keys())}")
                
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
        
    return False

def main():
    """Main test function"""
    
    print("Testing promotional recommendations...")
    
    # Test various combinations
    test_combinations = [
        (1, 1),   # 1% for 1 day
        (2, 1),   # 2% for 1 day
        (3, 1),   # 3% for 1 day
        (5, 1),   # 5% for 1 day
        (1, 3),   # 1% for 3 days
        (2, 3),   # 2% for 3 days
        (3, 3),   # 3% for 3 days
        (5, 3),   # 5% for 3 days
        (1, 7),   # 1% for 7 days
        (2, 7),   # 2% for 7 days
        (3, 7),   # 3% for 7 days
        (5, 7),   # 5% for 7 days
        (10, 1),  # 10% for 1 day
        (10, 3),  # 10% for 3 days
        (10, 7),  # 10% for 7 days
        (15, 1),  # 15% for 1 day
        (15, 3),  # 15% for 3 days
        (15, 7),  # 15% for 7 days
    ]
    
    positive_found = []
    
    for discount, duration in test_combinations:
        if test_single_promotion(discount, duration):
            positive_found.append((discount, duration))
            
        # Small delay to avoid overwhelming server
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    
    if positive_found:
        print(f"✅ Found {len(positive_found)} positive recommendations:")
        for discount, duration in positive_found:
            print(f"  - {discount}% discount for {duration} days")
    else:
        print("❌ No positive recommendations found")
        print("\nThis suggests the promotion logic needs adjustment.")
        
        # Let's test one more time with very detailed output
        print("\nTesting 1% discount for 1 day with detailed output...")
        test_single_promotion(1, 1)

if __name__ == "__main__":
    main() 