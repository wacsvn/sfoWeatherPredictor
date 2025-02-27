# SFO Weather Predictor

A machine learning application that predicts the maximum temperature at San Francisco International Airport (SFO) based on historical weather data. The application uses a Ridge Regression model to make next-day temperature predictions with a simple and intuitive graphical user interface.

![SFO Weather Predictor GUI](https://github.com/wacsvn/sfoWeatherPredictor/assets/81664765/03dfe1e6-d456-40f1-b4ac-b9131fc7d6ac)

## Features

- **Temperature Prediction**: Predicts the next day's maximum temperature based on current weather conditions
- **User-friendly GUI**: Simple interface for entering dates and viewing predictions
- **Error Metrics**: Displays Mean Absolute Error (MAE) alongside predictions for accuracy assessment
- **Data Preprocessing**: Automatically handles missing values and prepares data for prediction

## Technical Details

- Built with Python using PyQt5 for the GUI
- Uses scikit-learn's Ridge Regression algorithm for predictions
- Features include precipitation, maximum temperature, and minimum temperature as predictors
- Automatically handles missing data through forward-fill imputation

## Installation

### Option 1: Download the Executable (Windows)

1. Download the ZIP file from: [sfoWeatherPredict.zip](https://www.mediafire.com/file/stj4u4b0lq2s0l5/sfoWeatherPredict.zip/file)
2. Extract the ZIP file to a location of your choice
3. Run the `main` executable file

### Option 2: Run from Source Code

1. Clone the repository:
   ```
   git clone https://github.com/wacsvn/sfoWeatherPredictor.git
   ```

2. Install the required dependencies:
   ```
   pip install pandas scikit-learn PyQt5
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage

1. Launch the application
2. Enter a date in YYYY-MM-DD format in the input field
3. Click the "Predict" button
4. View the predicted maximum temperature and model accuracy (MAE)

## Data Source

The application uses historical weather data from San Francisco International Airport (SFO) stored in `sfoWeather.csv`. This file should be placed in the same directory as the application.

## Requirements

- Python 3.6 or higher
- pandas
- scikit-learn
- PyQt5

## Project Structure

- `main.py` - Main application file with GUI implementation
- `weatherPredictor.py` - Core prediction functionality and data processing
- `sfoWeather.csv` - Historical weather data (not included in repository)
