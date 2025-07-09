"""
Prophet forecasting model wrapper.
"""
import pandas as pd
from prophet import Prophet

class ProphetForecaster:
    """Wrapper class for Facebook Prophet forecasting model"""
    
    def __init__(self, include_weather=True, include_holidays=True, include_promotions=True):
        """Initialize the Prophet model with appropriate components"""
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        self.include_weather = include_weather
        self.include_holidays = include_holidays
        self.include_promotions = include_promotions
        self.trained = False  # Add trained attribute
        self.metrics = None  # Add metrics attribute
        
    def add_regressors(self, df):
        """Add relevant regressors to the model based on configuration"""
        if self.include_weather:
            if 'avg_temperature' in df.columns:
                self.model.add_regressor('avg_temperature')
            if 'avg_humidity' in df.columns:
                self.model.add_regressor('avg_humidity')
            if 'precpt' in df.columns:
                self.model.add_regressor('precpt')
            
        if self.include_promotions:
            if 'discount' in df.columns:
                self.model.add_regressor('discount', mode='multiplicative')
            if 'activity_flag' in df.columns:
                self.model.add_regressor('activity_flag', mode='multiplicative')
            
        return df
    
    def prepare_data(self, df):
        """Prepare data for Prophet model"""
        # Prophet requires ds (date) and y (target) columns
        forecast_df = df.copy()
        
        print("Columns before renaming:", forecast_df.columns.tolist())
        print("DataFrame shape before renaming:", forecast_df.shape)
        
        # Convert date column if needed
        if 'dt' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'dt': 'ds', 'sale_amount': 'y'})
        elif 'sale_date' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'sale_date': 'ds', 'sale_amount': 'y'})
        else:
            raise ValueError("Input DataFrame must have 'dt' or 'sale_date' column")
        
        if 'y' not in forecast_df.columns:
            raise ValueError("Input DataFrame must have 'sale_amount' column to rename to 'y'")
        
        print("Columns after renaming:", forecast_df.columns.tolist())
        print("DataFrame shape after renaming:", forecast_df.shape)
        print("First few rows:", forecast_df.head())
        
        # Handle any missing values
        forecast_df = forecast_df.dropna(subset=['ds', 'y'])
        
        # Convert date to datetime if it's not already
        if forecast_df['ds'].dtype == 'object':
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        
        # Add regressors if enabled
        forecast_df = self.add_regressors(forecast_df)
        
        return forecast_df
    
    def fit(self, df):
        """Fit the Prophet model"""
        forecast_df = self.prepare_data(df)
        self.model.fit(forecast_df)
        self.trained = True  # Set trained flag
        return self
    
    def train(self, df):
        """Train the model (alias for fit)"""
        result = self.fit(df)
        # Get the prepared DataFrame that was used for training
        prepared_df = self.prepare_data(df)
        # Set basic metrics
        self.metrics = {
            'data_points': len(prepared_df),
            'date_range': {
                'start': prepared_df['ds'].min().strftime('%Y-%m-%d'),
                'end': prepared_df['ds'].max().strftime('%Y-%m-%d')
            },
            'target_stats': {
                'mean': prepared_df['y'].mean(),
                'std': prepared_df['y'].std(),
                'min': prepared_df['y'].min(),
                'max': prepared_df['y'].max()
            }
        }
        return result
    
    def forecast_future(self, future_df=None, periods=30, freq='D'):
        """Generate forecast for future periods"""
        if not self.trained:
            raise ValueError("Model must be trained before forecasting")
        
        forecast = self.predict(periods=periods, freq=freq, future_df=future_df)
        return forecast
    
    def predict(self, periods=30, freq='D', future_df=None):
        """Generate forecast for the specified periods"""
        print("DEBUG: Starting predict method")
        print(f"DEBUG: periods={periods}, freq={freq}, future_df provided={future_df is not None}")
        
        if future_df is not None:
            print("DEBUG: Using provided future_df")
            print(f"DEBUG: Future dataframe shape: {future_df.shape}")
            print(f"DEBUG: Future dataframe columns: {future_df.columns.tolist()}")
            
            # For future data, we don't need to rename columns - just ensure ds column exists
            future = future_df.copy()
            if 'ds' not in future.columns:
                print("DEBUG: ERROR - Future DataFrame missing 'ds' column")
                raise ValueError("Future DataFrame must have 'ds' column")
            
            # Convert date to datetime if it's not already
            if future['ds'].dtype == 'object':
                print("DEBUG: Converting 'ds' column to datetime")
                future['ds'] = pd.to_datetime(future['ds'])
            
            print(f"DEBUG: Final future dataframe shape: {future.shape}")
            print(f"DEBUG: Final future dataframe columns: {future.columns.tolist()}")
        else:
            print("DEBUG: Creating future dataframe using model.make_future_dataframe")
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Add regressor values if they were included in training
            if hasattr(self.model, 'extra_regressors'):
                for regressor_name in self.model.extra_regressors:
                    # Use the mean value from training for forecasting
                    value = self.model.history[regressor_name].mean()
                    future[regressor_name] = value
        
        print("DEBUG: Calling model.predict...")
        # Generate forecast
        forecast = self.model.predict(future)
        print(f"DEBUG: Forecast completed. Shape: {forecast.shape}")
        print(f"DEBUG: Forecast columns: {forecast.columns.tolist()}")
        return forecast
    
    def plot_components(self):
        """Plot forecast components"""
        fig = self.model.plot_components(self.model.forecast)
        return fig
