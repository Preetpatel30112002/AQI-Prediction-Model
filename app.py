from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

aqi_df = pd.read_csv('Air_quality_index.csv')

aqi_df = aqi_df.dropna()

label_encoders = {}
for column in ['state', 'city']:
    le = LabelEncoder()
    aqi_df[column] = le.fit_transform(aqi_df[column])
    label_encoders[column] = le

state_city_dict = {}

for _, row in aqi_df.iterrows():
    state = label_encoders['state'].inverse_transform([row['state']])[0]
    city = label_encoders['city'].inverse_transform([row['city']])[0]
    if state not in state_city_dict:
        state_city_dict[state] = []
    if city not in state_city_dict[state]:
        state_city_dict[state].append(city)

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_state = request.form.get('state', '')
    cities = []
    aqi_data = []  # Initialize aqi_data to an empty list by default

    if selected_state:
        cities = state_city_dict.get(selected_state, [])  # Fetch cities for selected state
    
    return render_template('index2.html', state_city_dict=state_city_dict, selected_state=selected_state, cities=cities, aqi_data=aqi_data)

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    city = request.form.get('city')
    aqi_data = []  # Initialize AQI data

    if not state or not city:
        return render_template('index2.html', state_city_dict=state_city_dict, selected_state=state,selected_city=city, cities=state_city_dict.get(state, []),
                               prediction_text="State or city not selected.", aqi_data=aqi_data)

    try:
        state_encoded = label_encoders['state'].transform([state])[0]
        city_encoded = label_encoders['city'].transform([city])[0]
    except ValueError:
        return render_template('index2.html', state_city_dict=state_city_dict, selected_state=state,selected_city=city,cities=state_city_dict.get(state, []),
                               prediction_text="Invalid state or city selection.", aqi_data=aqi_data)

    additional_data = aqi_df[(aqi_df['state'] == state_encoded) & 
                             (aqi_df['city'] == city_encoded)]

    if additional_data.empty:
        return render_template('index2.html', state_city_dict=state_city_dict, selected_state=state,selected_city=city,cities=state_city_dict.get(state, []),
                               prediction_text="No data found for the selected state and city", aqi_data=aqi_data)

    additional_data = additional_data.iloc[0]
    avg_pollutant_avg = additional_data['pollutant_avg']
    avg_pollutant_min = additional_data['pollutant_min']
    avg_pollutant_max = additional_data['pollutant_max']

    # Prepare AQI data for chart
    aqi_data = [avg_pollutant_min, avg_pollutant_avg, avg_pollutant_max]

    final_features = np.array([[avg_pollutant_min, avg_pollutant_max, avg_pollutant_avg]])

    try:
        model = pickle.load(open('aqi_model.pkl', 'rb'))
        prediction = model.predict(final_features)
    except Exception as e:
        return render_template('index2.html', state_city_dict=state_city_dict, selected_state=state,selected_city=city,cities=state_city_dict.get(state, []),
                               prediction_text=f"Error occurred during prediction: {e}", aqi_data=aqi_data)

    aqi_category = map_prediction_to_aqi_category(prediction[0])

    return render_template('index2.html', state_city_dict=state_city_dict, selected_state=state,selected_city=city,cities=state_city_dict.get(state, []),
                           prediction_text=f'Predicted AQI Category: {aqi_category}', aqi_data=aqi_data)


def map_prediction_to_aqi_category(predicted_category):
    if predicted_category == 0:
        return "Good (0 - 50) or Moderate (51 - 100)"
    elif predicted_category == 1:
        return "Unhealthy for Sensitive Groups (101 - 150) or Higher"
    elif predicted_category == 2:
        return "Unhealthy (151 - 200) or Higher"
    else:
        return "Out of Range"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
