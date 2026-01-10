# Aureus 📈
*Where Algorithms Predict Value*

A sophisticated predictive analytics system that forecasts gold prices weekly using multiple ML models and visualizes results through an elegant web interface.

## 🌐 Live Dashboard

**View the live predictions:** [https://mahdiarsadeghi.github.io/gold/](https://mahdiarsadeghi.github.io/gold/)

## Features

- 🔮 **Weekly Price Predictions**: Automatically predicts gold prices for the upcoming week
- 🤖 **Multiple ML Models**: Compares 5 different algorithms (Random Forest, Linear Regression, Ridge, Gradient Boosting, SVR)
- 📊 **Interactive Web Dashboard**: Minimal, beautiful visualization of predictions and historical data
- 📝 **Prediction Logging**: Maintains a complete history of all predictions in JSON format
- 📈 **Technical Indicators**: Incorporates Moving Averages, RSI, Volatility, and momentum indicators
- 🎯 **Real-time Data**: Fetches live gold price data using Yahoo Finance API
- 📉 **Model Comparison**: Track and compare accuracy of different models over time

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/gold-price-predictor.git
cd gold-price-predictor
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run a Single Prediction

```bash
python gold_predictor.py
```

### Run with Shell Script

```bash
chmod +x run_prediction.sh
./run_prediction.sh
```

### Schedule Weekly Predictions (macOS/Linux)

Add to your crontab to run every Monday at 9 AM:

```bash
crontab -e
```

Add this line:
```
0 9 * * 1 cd /path/to/gold-price-predictor && ./run_prediction.sh
```

### View Prediction History

All predictions are automatically saved to `predictions_log.json`. You can:

1. View the JSON file directly
2. Run the predictor to see the last 5 predictions displayed
3. Parse the JSON programmatically for analysis

## Technical Details

### Features Used for Prediction

- **Moving Averages**: 7-day, 30-day, and 90-day moving averages
- **Volatility**: 30-day rolling standard deviation
- **Daily Returns**: Percentage change in price
- **RSI (Relative Strength Index)**: 14-period RSI for momentum
- **Price Changes**: Absolute price differences

### Models

The predictor uses 5 different ML algorithms for comparison:

1. **Random Forest Regressor** - Ensemble method with 100 decision trees
2. **Linear Regression** - Simple linear model for baseline comparison
3. **Ridge Regression** - L2 regularized linear model
4. **Gradient Boosting** - Sequential ensemble method
5. **Support Vector Regression (SVR)** - Non-linear kernel-based model

**Training Data**: 2 years of historical gold price data  
**Data Source**: Yahoo Finance (GC=F - Gold Futures)

## Output Format

Each prediction includes predictions from all 5 models:

```
======================================================================
GOLD PRICE PREDICTIONS - MULTI-MODEL COMPARISON
======================================================================
Prediction Date: 2026-01-10 14:30:00
Target Date: 2026-01-17
Current Price: $2045.50
----------------------------------------------------------------------
Model                     Predicted Price       Change
----------------------------------------------------------------------
Random Forest                      $2067.30       +1.07%
Linear Regression                  $2055.20       +0.47%
Ridge Regression                   $2058.40       +0.63%
Gradient Boosting                  $2071.80       +1.29%
SVR                                $2062.10       +0.81%
======================================================================
```

## Web Dashboard

The project includes a minimal, beautiful web interface that displays:

- **Historical Price Chart**: 30-day price history with smooth visualization
- **Model Predictions**: All 5 models' predictions with color-coded changes
- **Live Status**: Current price, last update time, and target date
- **Interactive Charts**: Built with Chart.js for responsive, modern visualization

The dashboard automatically updates when you run new predictions.

## Prediction Log Structure

```json
{
  "current_price": 2045.50,
  "predictions": {
    "Random Forest": {
      "predicted_price": 2067.30,
      "price_change_percent": 1.07
    },
    "Linear Regression": {
      "predicted_price": 2055.20,
      "price_change_percent": 0.47
    }
  },
  "prediction_date": "2026-01-10 14:30:00",
  "target_date": "2026-01-17",
  "historical_data": {
    "dates": ["2025-12-11", "..."],
    "prices": [2040.20, "..."]
  }
}
```

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- scikit-learn

## Disclaimer

⚠️ **This tool is for educational and informational purposes only.** 

- Predictions are based on historical data and technical indicators
- Past performance does not guarantee future results
- Do not use this as the sole basis for investment decisions
- Always consult with financial advisors before making investment decisions
- The accuracy of predictions may vary based on market conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

**Proprietary License with Educational Use**

This project is protected under a custom license that:
- ✅ **Allows** educational, personal, and research use (FREE)
- ✅ **Allows** academic teaching and non-commercial study
- ❌ **Restricts** commercial use without permission
- ❌ **Protects** algorithms and predictive methodologies from commercial exploitation

The algorithms, predictive models, and evaluation techniques in this repository are proprietary intellectual property.

**Commercial use requires permission.** See [LICENSE](LICENSE) for full terms.

For commercial licensing inquiries, contact [@mahdiarsadeghi](https://github.com/mahdiarsadeghi).

## Author

**Aureus** was created by Mahdiar Sadeghi as an intelligent predictive analytics system for financial markets, starting with gold price forecasting.

## Future Enhancements

- [x] Multiple ML models for comparison
- [x] Web dashboard with Chart.js visualization
- [ ] Model accuracy tracking and evaluation
- [ ] Add sentiment analysis from news sources
- [ ] Email/SMS notifications for predictions
- [ ] Include prediction confidence intervals
- [ ] Add support for other precious metals
- [ ] Automated GitHub Actions for weekly runs
- [ ] Historical accuracy metrics dashboard

---

**Last Updated**: January 2026
