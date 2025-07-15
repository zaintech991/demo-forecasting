# Professional Forecasting Dashboard

A comprehensive, real-time business intelligence dashboard for retail forecasting and analytics.

## 🚀 Features

### Core Analytics Modules

1. **Forecasting Analytics**
   - Real-time sales forecasting with 30-day predictions
   - Confidence intervals and trend analysis
   - Interactive line charts showing actual vs predicted values

2. **Weather Intelligence**
   - Seasonal sales pattern analysis
   - Weather impact correlation
   - Temperature, humidity, and precipitation effects

3. **Market Analysis**
   - Market share visualization with pie charts
   - Category correlation analysis
   - Competitive positioning insights

4. **Store Performance**
   - Store clustering and performance ranking
   - Radar charts for multi-dimensional analysis
   - Performance optimization recommendations

5. **Inventory Optimization**
   - Safety stock level calculations
   - Reorder point optimization
   - Cross-store inventory balancing

6. **Promotion Analysis**
   - ROI analysis for different promotion types
   - Price elasticity insights
   - Campaign effectiveness metrics

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5.3.0
- **Charts**: Chart.js 4.4.0
- **Icons**: Font Awesome 6.4.0
- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with AsyncPG

## 📊 Dashboard Components

### Header Section
- **Store Selector**: Choose from 25 representative stores
- **City Selector**: Select from 18 cities across regions
- **Product Selector**: Pick from 15 high-volume products
- **Real-time Refresh**: Manual refresh button for latest data

### Key Metrics Cards
- **Total Revenue**: Real-time revenue calculations
- **Total Products**: Product count from database
- **Total Stores**: Store count from database
- **Forecast Accuracy**: Model performance metrics

### Interactive Charts
- **Line Charts**: Time series forecasting data
- **Bar Charts**: Seasonal patterns and correlations
- **Pie Charts**: Market share distribution
- **Radar Charts**: Multi-dimensional store performance

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- PostgreSQL database
- Virtual environment

### Database Setup
1. Ensure PostgreSQL is running
2. Database should contain tables: `sales_data`, `store_hierarchy`, `product_hierarchy`, `city_hierarchy`, `weather_data`
3. Update connection settings in `.env` file

### Environment Configuration
```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://retail_user:retail_pass@localhost:5432/retail_forecast
DB_HOST=localhost
DB_PORT=5432
DB_NAME=retail_forecast
DB_USER=retail_user
DB_PASSWORD=retail_pass
```

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Access the Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:8000/professional_dashboard.html
```

## 📈 API Endpoints

### Core Endpoints
- `GET /enhanced/curated-data` - Dropdown data (cities, stores, products)
- `POST /enhanced/forecast` - Sales forecasting
- `POST /enhanced/seasonal-patterns` - Weather and seasonal analysis
- `POST /enhanced/market-share` - Market share analysis
- `POST /enhanced/category-correlations` - Category correlation matrix
- `POST /enhanced/store-clustering` - Store performance clustering
- `POST /enhanced/safety-stock` - Inventory safety stock levels
- `POST /enhanced/reorder-optimization` - Reorder point optimization
- `POST /enhanced/roi-optimization` - Promotion ROI analysis

### Request Format
All POST endpoints expect JSON with:
```json
{
  "city_id": 0,
  "store_id": 104,
  "product_id": 21
}
```

## 🎨 UI/UX Features

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized
- Collapsible accordion sections
- Touch-friendly controls

### Visual Design
- Modern gradient backgrounds
- Glass-morphism effects
- Smooth animations and transitions
- Professional color scheme
- Accessibility compliance (WCAG 2.1)

### Interactive Elements
- Hover effects on charts and cards
- Loading states with spinners
- Error handling with user feedback
- Real-time data updates

## 🔄 Data Flow

1. **User Selection**: User selects city, store, and product
2. **API Calls**: JavaScript makes parallel API calls
3. **Data Processing**: Backend processes requests with real database queries
4. **Chart Updates**: Charts update with new data
5. **Insights Display**: AI-powered insights shown in panels

## 🚨 Error Handling

### Frontend
- Network error handling
- Empty data state management
- Loading state indicators
- Graceful degradation

### Backend
- Database connection error handling
- Parameter validation
- Fallback data generation
- Comprehensive logging

## 📱 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔒 Security Features

- Input validation and sanitization
- CORS policy configuration
- SQL injection prevention
- XSS protection

## 📊 Performance Optimization

- Parallel API calls
- Chart.js optimization
- Lazy loading for large datasets
- Efficient database queries
- Connection pooling

## 🐛 Troubleshooting

### Common Issues

1. **Charts not displaying**
   - Check browser console for JavaScript errors
   - Verify Chart.js library is loaded
   - Ensure canvas elements exist in HTML

2. **API errors**
   - Check server logs for database connection issues
   - Verify parameter types (integers not strings)
   - Ensure database tables exist

3. **Slow loading**
   - Check database query performance
   - Verify connection pool settings
   - Monitor network requests in browser dev tools

### Debug Mode
Enable debug logging by setting `LOG_LEVEL=DEBUG` in `.env`

## 🔄 Future Enhancements

- Real-time WebSocket updates
- Advanced filtering options
- Export functionality (PDF, Excel)
- Custom dashboard layouts
- Mobile app integration
- Advanced ML model integration

## 📞 Support

For technical support or feature requests, please check the application logs and verify all dependencies are properly installed.

## 📄 License

This project is part of the BoolMind AI forecasting system.

---

**Last Updated**: July 2025
**Version**: 1.0.0
**Compatibility**: Python 3.9+, PostgreSQL 12+ 