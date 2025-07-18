import os
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Enable FastF1 caching
cache_dir = './f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# loading the 2024 british gp race results from fastf1
session_2024 = fastf1.get_session(2024, 'British Grand Prix', 'R')
session_2024.load()


# extracting lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] =  laps_2024["LapTime"].dt.total_seconds()


# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Oscar Piastri", "Lando Norris",
                "George Russell", "Lewis Hamilton", "Charles Leclerc",
                "Kimi Antonelli", "Oliver Bearman", "Fernando Alonso",
                "Pierre Gasly", "Carlos Sainz", "Yuki Tsunoda",
                "Issac Hadjar", "Alexander Alon", "Esteban Ocon",
                "Liam Lawson", "Gabriel Bortoleto", "Lance Stroll",
                "Nico Hulkenberg", "Franco Colapinto"],
    "QualifyingTime (s)": [84.892, 84.995, 85.010, 85.029, 85.095,
                           85.121, 85.374, 85.471, 85.621, 85.785,
                           85.746, 85.826, 85.864, 85.889, 85.950,
                           86.440, 86.446, 86.504, 86.574, 87.060]
})

# Map full names to fastf1 3-letter codes
driver_mapping = {
    "Max Verstappen": "VER",
    "Oscar Piastri": "PIA",
    "Lando Norris": "NOR",
    "George Russell": "RUS",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Kimi Antonelli": "ANT",
    "Oliver Bearman": "BEA",
    "Fernando Alonso": "ALO",
    "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI",
    "Yuki Tsunoda": "TSU",
    "Issac Hadjar": "HAD",
    "Alexander Albonso": "ALB",
    "Esteban Ocon": "OCO",
    "Liam Lawson": "LAW",
    "Gabriel Bortoleto": "BOR",
    "Lance Stroll": "STR",
    "Nico Hulkenberg": "HUL",
    "Franco Colapinto": "COL"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!!!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying Times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print the final result
print("\n Predicted 2025 British Grand Prix Winner \n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")