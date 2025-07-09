"""
Data preprocessing module for the FreshRetailNet-50K dataset.
Handles feature engineering, missing values, outliers, and normalization.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import joblib

class DataPreprocessor:
    def __init__(self, save_path='models/preprocessor'):
        """Initialize the preprocessor with path to save scaling models."""
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        
    def process_sequence_data(self, sequence):
        """Convert sequence data to a list."""
        try:
            if isinstance(sequence, str):
                return json.loads(sequence)
            return list(sequence)
        except (json.JSONDecodeError, TypeError):
            return []

    def add_time_features(self, df):
        """Add time-based features."""
        df = df.copy()
        
        # Convert sale_date to datetime if it's not already
        if df['sale_date'].dtype != 'datetime64[ns]':
            df['sale_date'] = pd.to_datetime(df['sale_date'])
        
        # Extract time features
        df['day_of_week'] = df['sale_date'].dt.dayofweek
        df['month'] = df['sale_date'].dt.month
        df['is_weekend'] = df['sale_date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['day_of_month'] = df['sale_date'].dt.day
        df['week_of_year'] = df['sale_date'].dt.isocalendar().week
        
        return df

    def add_weather_features(self, df):
        """Add weather-derived features."""
        df = df.copy()
        
        # Temperature levels
        df['temperature_level'] = pd.cut(
            df['avg_temperature'],
            bins=[-np.inf, 15, 22, 28, np.inf],
            labels=['Cold', 'Mild', 'Warm', 'Hot']
        )
        
        # Humidity levels
        df['humidity_level'] = pd.qcut(
            df['avg_humidity'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Precipitation and wind features
        df['is_rainy'] = (df['precpt'] > 0).astype(int)
        df['is_windy'] = (df['avg_wind_level'] > df['avg_wind_level'].median()).astype(int)
        
        # Weather condition combinations
        df['weather_condition'] = df.apply(
            lambda x: self._get_weather_condition(
                x['is_rainy'],
                x['is_windy'],
                x['temperature_level'],
                x['humidity_level']
            ),
            axis=1
        )
        
        return df

    def _get_weather_condition(self, is_rainy, is_windy, temp_level, humidity_level):
        """Determine overall weather condition."""
        conditions = []
        if is_rainy:
            conditions.append('Rainy')
        if is_windy:
            conditions.append('Windy')
        if temp_level in ['Cold', 'Hot']:
            conditions.append(temp_level)
        if humidity_level == 'Very High':
            conditions.append('Humid')
        
        return '_'.join(conditions) if conditions else 'Normal'

    def add_sales_features(self, df):
        """Add sales-derived features."""
        df = df.copy()
        
        # Discount features
        df['has_discount'] = (df['discount'] > 0).astype(int)
        df['discount_level'] = pd.qcut(
            df['discount'].fillna(0),
            q=4,
            labels=['No Discount', 'Low', 'Medium', 'High']
        )
        
        # Stock features
        df['hours_sale_list'] = df['hours_sale'].apply(self.process_sequence_data)
        df['hours_stock_status_list'] = df['hours_stock_status'].apply(self.process_sequence_data)
        
        # Calculate stock-related features
        df['daily_sales_pattern'] = df['hours_sale_list'].apply(
            lambda x: 'Normal' if not x else (
                'Morning_Peak' if np.argmax(x) < 6 else (
                'Evening_Peak' if np.argmax(x) > 12 else 'Midday_Peak'
                )
            )
        )
        
        df['stockout_risk'] = df.apply(
            lambda x: len([1 for status in x['hours_stock_status_list'] if status == 0]) / 
                    len(x['hours_stock_status_list']) if x['hours_stock_status_list'] else 0,
            axis=1
        )
        
        # Clean up temporary columns
        df = df.drop(['hours_sale_list', 'hours_stock_status_list'], axis=1)
        
        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Create a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Weather data - fill with city-specific means
        weather_cols = ['avg_temperature', 'avg_humidity', 'precpt', 'avg_wind_level']
        for col in weather_cols:
            city_means = df.groupby('city_id')[col].transform('mean')
            df[col] = df[col].fillna(city_means)
        
        # Sales data
        df['discount'] = df['discount'].fillna(0)
        df['stock_hour6_22_cnt'] = df['stock_hour6_22_cnt'].fillna(0)
        
        # Handle missing sequence data
        df['hours_sale'] = df['hours_sale'].fillna('[]')
        df['hours_stock_status'] = df['hours_stock_status'].fillna('[]')
        
        return df

    def handle_outliers(self, df):
        """Detect and handle outliers using IQR method."""
        df = df.copy()
        
        # Calculate IQR for sales amount
        Q1 = df['sale_amount'].quantile(0.25)
        Q3 = df['sale_amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Flag outliers
        df['is_sales_outlier'] = (
            (df['sale_amount'] < lower_bound) | 
            (df['sale_amount'] > upper_bound)
        ).astype(int)
        
        # Create capped version of sales amount
        df['sale_amount_capped'] = df['sale_amount'].clip(lower_bound, upper_bound)
        
        # Store outlier bounds for future reference
        self.outlier_bounds = {
            'sale_amount': {
                'lower': lower_bound,
                'upper': upper_bound
            }
        }
        
        return df

    def normalize_features(self, df):
        """Normalize numerical features using StandardScaler."""
        df = df.copy()
        
        numeric_features = [
            'sale_amount',
            'sale_amount_capped',
            'avg_temperature',
            'avg_humidity',
            'precpt',
            'avg_wind_level'
        ]
        
        # Fit and transform the data
        scaled_features = self.scaler.fit_transform(df[numeric_features])
        
        # Add scaled features to dataframe
        for i, col in enumerate(numeric_features):
            df[f'{col}_scaled'] = scaled_features[:, i]
        
        # Save the scaler
        joblib.dump(self.scaler, self.save_path / 'standard_scaler.joblib')
        
        return df

    def preprocess(self, df):
        """Apply all preprocessing steps in sequence."""
        print("Starting preprocessing pipeline...")
        
        print("1. Handling missing values...")
        df = self.handle_missing_values(df)
        
        print("2. Adding time features...")
        df = self.add_time_features(df)
        
        print("3. Adding weather features...")
        df = self.add_weather_features(df)
        
        print("4. Adding sales features...")
        df = self.add_sales_features(df)
        
        print("5. Handling outliers...")
        df = self.handle_outliers(df)
        
        print("6. Normalizing features...")
        df = self.normalize_features(df)
        
        print("Preprocessing completed!")
        return df

    def save_preprocessor_state(self):
        """Save preprocessor state and parameters."""
        state = {
            'outlier_bounds': self.outlier_bounds
        }
        joblib.dump(state, self.save_path / 'preprocessor_state.joblib')

    def load_preprocessor_state(self):
        """Load preprocessor state and parameters."""
        state = joblib.load(self.save_path / 'preprocessor_state.joblib')
        self.outlier_bounds = state['outlier_bounds']
        self.scaler = joblib.load(self.save_path / 'standard_scaler.joblib') 