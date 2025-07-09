"""
Promotion uplift model to estimate the impact of promotions on sales.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromoUpliftModel:
    """
    Model to predict promotion uplift (how much sales increase due to promotions).
    """
    
    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initialize promotion uplift model.
        
        Args:
            model_type (str): Type of model to use - 'random_forest', 'gradient_boost', or 'elastic_net'
            model_params (dict): Parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.pipeline = None
        self.feature_importance = None
        self.trained = False
        self.scaler = StandardScaler()
        
        # Initialize model based on type
        self._initialize_model()
        
        logger.info(f"Initialized PromoUpliftModel with model_type={model_type}")
    
    def _initialize_model(self):
        """Initialize the appropriate model based on model_type."""
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
            # Override with user params if provided
            params.update(self.model_params)
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'gradient_boost':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            # Override with user params if provided
            params.update(self.model_params)
            self.model = GradientBoostingRegressor(**params)
        elif self.model_type == 'elastic_net':
            params = {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'random_state': 42
            }
            # Override with user params if provided
            params.update(self.model_params)
            self.model = ElasticNet(**params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. "
                             f"Supported types are: 'random_forest', 'gradient_boost', 'elastic_net'")
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
    
    def prepare_data(self, df, target_col='uplift', drop_cols=None):
        """
        Prepare data for modeling by creating features and target.
        
        Args:
            df (pd.DataFrame): Input dataframe with sales and promotion data
            target_col (str): Column name containing the target variable (uplift)
            drop_cols (list): Columns to drop from features
            
        Returns:
            tuple: X (features), y (target)
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Check if the target column exists
        if target_col not in data.columns:
            logger.warning(f"Target column '{target_col}' not found. "
                          f"Will attempt to calculate uplift from sales data.")
            
            # Check if we can calculate uplift
            required_cols = ['sale_amount', 'promo_flag']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns for uplift calculation not found: {required_cols}")
            
            # Calculate uplift as the difference between promo and non-promo sales
            # Group by relevant dimensions and calculate average sales with/without promo
            group_cols = [col for col in ['store_id', 'product_id', 'city_id', 'first_category_id'] 
                         if col in data.columns]
            
            if not group_cols:
                raise ValueError("No grouping columns found for uplift calculation")
            
            # Calculate average sales with and without promos
            avg_sales = data.groupby(group_cols + ['promo_flag'])['sale_amount'].mean().reset_index()
            avg_sales_pivot = avg_sales.pivot_table(
                index=group_cols, 
                columns='promo_flag', 
                values='sale_amount'
            ).reset_index()
            
            # Rename columns
            avg_sales_pivot.columns.name = None
            avg_sales_pivot = avg_sales_pivot.rename(columns={True: 'promo_sales', False: 'non_promo_sales'})
            
            # Add debug prints before pivot
            if hasattr(df, 'shape'):
                print('prepare_data input shape:', df.shape)
            else:
                print('prepare_data input length:', len(df))
            print('promo_flag counts:', df['promo_flag'].value_counts(dropna=False))
            print('Grouping columns:', df.columns.tolist())
            # After creating avg_sales_pivot
            if 'promo_sales' not in avg_sales_pivot.columns or 'non_promo_sales' not in avg_sales_pivot.columns:
                shape_info = df.shape if hasattr(df, 'shape') else len(df)
                raise ValueError(f'Not enough promo/non-promo sales for uplift calculation. Data shape: {shape_info}, promo_flag counts: {df["promo_flag"].value_counts(dropna=False).to_dict()}, columns: {df.columns.tolist()}')
            avg_sales_pivot['uplift'] = avg_sales_pivot['promo_sales'] - avg_sales_pivot['non_promo_sales']
            avg_sales_pivot['uplift_pct'] = (avg_sales_pivot['uplift'] / avg_sales_pivot['non_promo_sales']) * 100
            
            # Merge uplift back to original data
            data = pd.merge(data, avg_sales_pivot[group_cols + ['uplift', 'uplift_pct']], on=group_cols)
            target_col = 'uplift'
        
        # Prepare features (X) and target (y)
        y = data[target_col]
        
        # Determine columns to drop
        cols_to_drop = drop_cols or []
        cols_to_drop.append(target_col)
        
        # Add any target-related columns to drop
        for col in data.columns:
            if 'uplift' in col and col != target_col:
                cols_to_drop.append(col)
        
        # Drop non-feature columns
        drop_cols_existing = [col for col in cols_to_drop if col in data.columns]
        X = data.drop(columns=drop_cols_existing)
        
        # Convert categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Handle date columns
        date_cols = [col for col in X.columns if 'date' in col.lower()]
        for date_col in date_cols:
            if pd.api.types.is_datetime64_any_dtype(X[date_col]):
                # Extract date features
                X[f'{date_col}_day'] = X[date_col].dt.day
                X[f'{date_col}_month'] = X[date_col].dt.month
                X[f'{date_col}_year'] = X[date_col].dt.year
                X[f'{date_col}_dayofweek'] = X[date_col].dt.dayofweek
                X.drop(columns=[date_col], inplace=True)
        
        logger.info(f"Prepared data with {X.shape[1]} features")
        return X, y
    
    def train(self, df, target_col='uplift', drop_cols=None, test_size=0.2):
        """
        Train the promotion uplift model.
        
        Args:
            df (pd.DataFrame): Input dataframe with sales and promotion data
            target_col (str): Column name containing the target variable (uplift)
            drop_cols (list): Columns to drop from features
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        # Prepare data
        X, y = self.prepare_data(df, target_col, drop_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the model
        logger.info(f"Training {self.model_type} model with {X_train.shape[0]} samples")
        self.pipeline.fit(X_train, y_train)
        self.trained = True
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Top 5 features: {', '.join(self.feature_importance['feature'].head().tolist())}")
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model evaluation metrics - MAE: {metrics['mae']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        metrics['cv_mae'] = -cv_scores.mean()
        
        logger.info(f"5-fold CV MAE: {metrics['cv_mae']:.4f}")
        
        return metrics
    
    def predict(self, df, drop_cols=None):
        """
        Predict promotion uplift for new data.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            drop_cols (list): Columns to drop from features
            
        Returns:
            np.ndarray: Predicted uplift values
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features (assuming no target column)
        X, _ = self.prepare_data(df, target_col='_dummy_', drop_cols=drop_cols)
        
        # Ensure all feature columns from training are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Keep only the original feature columns and in the same order
        X = X[self.feature_names]
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        
        return predictions
    
    def analyze_promotion_effectiveness(self, df, group_cols=None, target_col='uplift'):
        """
        Analyze promotion effectiveness by different dimensions.
        
        Args:
            df (pd.DataFrame): Input dataframe with sales and promotion data
            group_cols (list): Columns to group by for analysis
            target_col (str): Column name containing the target variable (uplift)
            
        Returns:
            pd.DataFrame: Analysis results
        """
        if not self.trained:
            logger.warning("Model not trained, will use actual data for analysis")
            
        # Default grouping columns
        if group_cols is None:
            group_cols = ['store_id', 'product_id']
            # Add optional grouping columns if they exist
            for col in ['first_category_id', 'city_id', 'promotion_type', 'discount_percentage']:
                if col in df.columns:
                    group_cols.append(col)
        
        # Check if uplift column exists
        if target_col in df.columns:
            analysis_df = df.copy()
        else:
            # Make predictions
            predictions = self.predict(df)
            analysis_df = df.copy()
            analysis_df[target_col] = predictions
        
        # Group by specified dimensions
        group_results = analysis_df.groupby(group_cols).agg({
            target_col: ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        # Flatten multi-level column names
        group_results.columns = ['_'.join(col).strip('_') for col in group_results.columns.values]
        
        # Add percentile ranks to see which combinations perform best
        group_results[f'{target_col}_percentile'] = group_results[f'{target_col}_mean'].rank(pct=True) * 100
        
        # Sort by effectiveness
        group_results = group_results.sort_values(f'{target_col}_mean', ascending=False)
        
        return group_results
    
    def save_model(self, path=None):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        if path is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/saved')
            os.makedirs(model_dir, exist_ok=True)
            path = os.path.join(model_dir, f"promo_uplift_{self.model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib")
        
        # Save model
        joblib.dump(self.pipeline, path)
        
        # Save feature names separately if needed
        feature_path = path.replace('.joblib', '_features.joblib')
        joblib.dump({
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }, feature_path)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load_model(self, path):
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            bool: True if loading was successful
        """
        try:
            # Load model
            self.pipeline = joblib.load(path)
            self.model = self.pipeline.named_steps['model']
            self.scaler = self.pipeline.named_steps['scaler']
            self.trained = True
            
            # Load feature names
            feature_path = path.replace('.joblib', '_features.joblib')
            if os.path.exists(feature_path):
                features_data = joblib.load(feature_path)
                self.feature_names = features_data.get('feature_names', [])
                self.feature_importance = features_data.get('feature_importance', None)
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample features
    data = {
        'store_id': np.random.randint(1, 20, size=n_samples),
        'product_id': np.random.randint(100, 150, size=n_samples),
        'sale_date': [datetime(2022, 1, 1) + timedelta(days=i % 365) for i in range(n_samples)],
        'sale_amount': np.random.normal(100, 20, size=n_samples),
        'promo_flag': np.random.choice([True, False], size=n_samples, p=[0.3, 0.7]),
        'discount': np.random.uniform(0, 0.3, size=n_samples),
        'city_id': np.random.randint(1, 5, size=n_samples),
        'first_category_id': np.random.randint(1, 10, size=n_samples)
    }
    
    # Increase sales for promotions
    df = pd.DataFrame(data)
    df.loc[df['promo_flag'], 'sale_amount'] *= (1 + df.loc[df['promo_flag'], 'discount'] * 3)
    
    # Create model
    model = PromoUpliftModel(model_type='gradient_boost')
    
    # Train model
    metrics = model.train(df, drop_cols=['sale_date'])
    
    # Analyze promotion effectiveness
    effectiveness = model.analyze_promotion_effectiveness(df)
    print(effectiveness.head())
