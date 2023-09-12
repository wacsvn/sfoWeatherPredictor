import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


def read_weather_data(file_path):
    """
    Reads weather data from a CSV file and returns a DataFrame.
    """
    weather = pd.read_csv(file_path, index_col="DATE")
    return weather


def preprocess_weather_data(weather):
    """
    Preprocesses weather data by selecting relevant columns, handling missing values, and data type conversion.
    """
    core = weather[["PRCP", "TMAX", "TMIN"]].copy()
    core.columns = ["precip", "tempMax", "tempMin"]
    core["precip"].fillna(0, inplace=True)
    core = core.fillna(method="ffill")
    core.index = pd.to_datetime(core.index)
    return core


def create_target_column(core):
    """
    Creates the 'target' column in the core DataFrame for temperature prediction.
    """
    core["target"] = core["tempMax"].shift(-1)
    return core


def train_weather_model(core, predictors):
    """
    Trains a weather prediction model using Ridge Regression.
    """
    regr = Ridge(alpha=0.1)
    train = core.dropna()
    test = core.dropna()
    regr.fit(train[predictors], train["target"])
    return regr, test


def predict_weather(model, test_data, input_date, predictors):
    input_date = pd.to_datetime(input_date)
    input_date = input_date.replace(hour=0, minute=0, second=0)

    input_data = test_data[test_data.index.date == input_date.date()]  # Compare only the date part
    print("Input Data (Formatted):")
    print(input_data)
    if not input_data.empty:
        predictions = model.predict(input_data[predictors])
        mae = mean_absolute_error(input_data["target"], predictions)
        return predictions[0], mae
    else:
        return None, None


def main():
    # Define file path to your weather data CSV
    file_path = "sfoWeather.csv"

    # Read and preprocess weather data
    weather = read_weather_data(file_path)
    core = preprocess_weather_data(weather)

    # Create the 'target' column
    core = create_target_column(core)

    # Define predictors for the model
    predictors = ["precip", "tempMax", "tempMin"]

    # Train the weather prediction model
    model, test_data = train_weather_model(core, predictors)

    # Get user input for the date they want to predict
    input_date = input("Enter the date (YYYY-MM-DD) for weather prediction: ")

    # Predict the weather for the specified date
    prediction, mae = predict_weather(model, test_data, input_date, predictors)

    if prediction is not None:
        print("Predicted average temperature for {}: {:.2f}°F".format(input_date, prediction))
        print("Mean Absolute Error:", mae)
    else:
        print("No data available for the specified date.")


if __name__ == "__main__":
    main()
