"""
Promotion uplift model for estimating promotion effectiveness.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class PromoUpliftModel:
    """Model for predicting promotion effectiveness"""
    
    def __init__(self):
        """Initialize the promotion uplift model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'discount', 'activity_flag', 
            'avg_temperature', 'avg_humidity',
            'precpt', 'holiday_flag'
        ]
        
    def prepare_data(self, df):
        """Prepare data for uplift modeling"""
        # Calculate uplift as percentage increase over baseline sales
        # For this demo, we'll use a simple approach: compare sales to the average sales for the product
        promo_df = df.copy()
        
        # Group by product to calculate average sales
        product_avg = promo_df.groupby('product_id')['sale_amount'].mean().reset_index()
        product_avg.rename(columns={'sale_amount': 'avg_sales'}, inplace=True)
        
        # Merge back to original dataframe
        promo_df = promo_df.merge(product_avg, on='product_id')
        
        # Calculate uplift
        promo_df['uplift'] = (promo_df['sale_amount'] - promo_df['avg_sales']) / promo_df['avg_sales']
        
        # Prepare features - use only those that are available
        available_features = [col for col in self.feature_columns if col in promo_df.columns]
        X = promo_df[available_features]
        
        # Fill missing values with medians
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
            
        y = promo_df['uplift']
        
        return X, y
    
    def fit(self, df):
        """Fit the uplift model"""
        X, y = self.prepare_data(df)
        
        # Update feature columns to only those available
        self.feature_columns = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained with MSE: {self.mse:.4f}, R²: {self.r2:.4f}")
        return self
    
    def predict(self, df):
        """Predict uplift for new data"""
        # Extract available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features]
        
        # Fill missing columns with medians or zeros
        for col in self.feature_columns:
            if col not in X.columns:
                if col == 'discount':
                    X[col] = 1.0  # No discount
                elif col == 'activity_flag' or col == 'holiday_flag':
                    X[col] = 0  # No activity or holiday
                else:
                    X[col] = 0  # Set to zero for other features
        
        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        # Predict
        uplift = self.model.predict(X_scaled)
        
        return uplift 