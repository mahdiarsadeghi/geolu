# Gold Price Predictor 📈

A Python-based gold price prediction system that automatically forecasts gold prices weekly and maintains a comprehensive log of all predictions over time.

## Features

- 🔮 **Weekly Price Predictions**: Automatically predicts gold prices for the upcoming week
- 📊 **Machine Learning Model**: Uses Random Forest algorithm with multiple technical indicators
- 📝 **Prediction Logging**: Maintains a complete history of all predictions in JSON format
- 📈 **Technical Indicators**: Incorporates Moving Averages, RSI, Volatility, and momentum indicators
- 🎯 **Real-time Data**: Fetches live gold price data using Yahoo Finance API

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

### Model

- **Algorithm**: Random Forest Regressor
- **Training Data**: 2 years of historical gold price data
- **Data Source**: Yahoo Finance (GC=F - Gold Futures)

## Output Format

Each prediction includes:
- Current gold price
- Predicted price for next week
- Percentage change expected
- Prediction timestamp
- Target date

Example output:
```
============================================================
GOLD PRICE PREDICTION
============================================================
Prediction Date: 2026-01-10 14:30:00
Target Date: 2026-01-17
Current Price: $2045.50
Predicted Price (1 week): $2067.30
Expected Change: +1.07%
============================================================
```

## Prediction Log Structure

```json
[
  {
    "current_price": 2045.50,
    "predicted_price": 2067.30,
    "price_change_percent": 1.07,
    "prediction_date": "2026-01-10 14:30:00",
    "target_date": "2026-01-17"
  }
]
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

MIT License - feel free to use this project for your own purposes.

## Author

Created as an automated gold price prediction and logging system.

## Future Enhancements

- [ ] Add sentiment analysis from news sources
- [ ] Implement ensemble models
- [ ] Create web dashboard for visualization
- [ ] Add email notifications for predictions
- [ ] Include prediction accuracy metrics
- [ ] Add support for other precious metals

---

**Last Updated**: January 2026
