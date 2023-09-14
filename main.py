import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from weatherPredictor import read_weather_data, preprocess_weather_data, create_target_column, train_weather_model, predict_weather

class WeatherPredictionApp(QMainWindow):
    def __init__(self, core, predictors):
        super().__init__()
        self.setWindowTitle("Weather Prediction App")
        self.core = core
        self.predictors = predictors

        # Create GUI components
        self.date_label = QLabel("Enter Date (YYYY-MM-DD):")
        self.date_input = QLineEdit()
        self.predict_button = QPushButton("Predict")
        self.result_label = QLabel("")

        # Set font for labels
        label_font = QFont()
        label_font.setPointSize(12)
        self.date_label.setFont(label_font)
        self.result_label.setFont(label_font)

        # Center-align labels
        self.date_label.setAlignment(Qt.AlignCenter)
        self.result_label.setAlignment(Qt.AlignCenter)

        # Set style for Predict button
        button_style = "QPushButton { background-color: #AA336A; color: white; border: none; padding: 10px 20px; }"
        button_style += "QPushButton:hover { background-color: #702963; }"
        self.predict_button.setStyleSheet(button_style)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.date_label)
        layout.addWidget(self.date_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect button click event to prediction function
        self.predict_button.clicked.connect(self.predict_weather)

    def predict_weather(self):
        input_date = self.date_input.text()
        model, test_data = train_weather_model(self.core, self.predictors)
        prediction, mae = predict_weather(model, test_data, input_date, self.predictors)
        if prediction is not None:
            self.result_label.setText(f"Predicted Temperature: {prediction:.2f}°C\nMean Absolute Error: {mae:.2f}")
        else:
            self.result_label.setText("No data available for the specified date.")

def main():
    # Read weather data and preprocess it
    weather = read_weather_data("sfoWeather.csv")
    core = preprocess_weather_data(weather)
    core = create_target_column(core)
    predictors = ["precip", "tempMax", "tempMin"]

    app = QApplication(sys.argv)
    window = WeatherPredictionApp(core, predictors)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
