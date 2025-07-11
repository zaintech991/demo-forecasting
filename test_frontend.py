#!/usr/bin/env python3
"""
Test script to run the forecasting application with the new AI features.
This script will start the application and provide testing guidance.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "asyncpg",
        "python-dotenv",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("To install them, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_database():
    """Check if database is accessible."""
    try:
        import asyncpg
        import asyncio

        async def test_connection():
            try:
                conn = await asyncpg.connect(
                    dsn=os.getenv(
                        "DATABASE_URL",
                        "postgresql://user:password@localhost:5432/dbname",
                    )
                )
                await conn.close()
                return True
            except Exception as e:
                print(f"Database connection error: {e}")
                return False

        return asyncio.run(test_connection())
    except Exception as e:
        print(f"Database check failed: {e}")
        return False


def start_application():
    """Start the FastAPI application."""
    print("🚀 Starting the Retail Forecasting Application...")
    print("   This includes all 6 use cases:")
    print("   1. Daily Sales Forecasting")
    print("   2. Promotion Uplift Estimation")
    print("   3. Stockout-Aware Demand Estimation")
    print("   4. ⭐ Weather-Sensitive Demand Modeling")
    print("   5. ⭐ Category-Level Demand Forecasting")
    print("   6. ⭐ Store Clustering & Behavior Segmentation")
    print()

    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    # Start uvicorn server
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    try:
        process = subprocess.Popen(cmd)

        # Wait a moment for server to start
        time.sleep(3)

        # Open browser
        print("🌐 Opening browser at http://localhost:8000")
        webbrowser.open("http://localhost:8000")

        print("\n" + "=" * 60)
        print("📊 RETAIL FORECASTING APPLICATION - TESTING GUIDE")
        print("=" * 60)

        print("\n🔍 TESTING THE DYNAMIC AI FEATURES:")
        print("\n1. 🌡️ WEATHER-SENSITIVE DEMAND (FULLY DYNAMIC):")
        print("   • Click the 'Weather Impact' tab")
        print("   • Select different products (Coffee vs Fresh Produce)")
        print("   • Notice REAL weather correlations calculated from YOUR data")
        print("   • See PRODUCT-SPECIFIC temperature/humidity/rain thresholds")
        print("   • View WEATHER-PROMOTION integration insights")
        print("   • 🎯 NO MORE hardcoded 65%, 45%, 30%, 25% values!")

        print("\n2. 📊 CATEGORY-LEVEL FORECASTING (REAL DATA):")
        print("   • Click the 'Category Analysis' tab")
        print("   • Select different categories from the dropdown")
        print("   • View REAL market share calculated from your sales data")
        print("   • See ACTUAL growth rates (recent vs historical performance)")
        print("   • Notice REAL seasonality patterns from your monthly data")
        print("   • Compare performance across different categories")

        print("\n3. 🏪 STORE CLUSTERING (PERFORMANCE-BASED):")
        print("   • Click the 'Store Clustering' tab")
        print("   • Select different stores to analyze")
        print("   • View REAL cluster assignments based on performance data")
        print("   • See ACTUAL performance ranking (Top X%)")
        print("   • Notice STORE-SPECIFIC recommendations based on real metrics")
        print("   • Compare weekend vs weekday performance patterns")

        print("\n🎯 DYNAMIC TESTING TIPS:")
        print("   • Try different product combinations - see how weather impact varies")
        print("   • Compare stores - notice different cluster assignments and rankings")
        print("   • Switch categories - observe different seasonality patterns")
        print("   • All insights now come from YOUR actual database data!")
        print("   • Check browser console for debugging info")
        print("   • API docs available at http://localhost:8000/docs")

        print("\n⚡ API ENDPOINTS (NEW):")
        print("   • POST /api/weather/analyze - Weather sensitivity analysis")
        print("   • POST /api/category/performance - Category performance metrics")
        print("   • POST /api/stores/insights - Store clustering insights")

        print("\n🛑 To stop the server: Press Ctrl+C")
        print("=" * 60)

        # Wait for the process to complete
        process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"❌ Error starting application: {e}")


def main():
    """Main function to run the test."""
    print("🔧 RETAIL FORECASTING - FRONTEND TEST SETUP")
    print("=" * 50)

    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        print("❌ Please install missing packages first.")
        return

    print("✅ All required packages are available.")

    # Check database (optional)
    print("🗄️  Checking database connection...")
    db_ok = check_database()
    if not db_ok:
        print("⚠️  Database not accessible - will use fallback data")
    else:
        print("✅ Database connection successful.")

    print("\n" + "🎉 Setup complete! Starting application..." + "\n")

    # Start application
    start_application()


if __name__ == "__main__":
    main()
