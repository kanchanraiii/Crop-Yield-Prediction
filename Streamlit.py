import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import os
import sklearn

st.sidebar.title("Crop Yield Predictor")

with st.sidebar:
    selected=option_menu(
        menu_title=None,
        options=["Homepage","Crop Yield Predictor"],
        icons=["house-fill","flower1"]
    )

if selected == "Homepage":
    st.title("Crop Yield Predictor🌱")
    st.image("crop_image.jpg",use_column_width=True)
    st.title("Overview of the Model")
    st.header(" About the model")
    st.write("Crop yield prediction is an essential aspect of agricultural planning and decision-making. Accurate predictions of crop yields can help farmers, agronomists, and policymakers make informed decisions to enhance productivity, manage resources efficiently, and ensure food security. Our crop yield prediction model leverages machine learning techniques to forecast the yield of various crops based on several input features, such as soil pH, rainfall, temperature, area, and production. By analyzing historical data and identifying patterns, the model aims to provide reliable yield estimates to support agricultural activities.")
    st.header(" Farming in India: Context and Relevance")
    st.markdown("""India is an agrarian country with a diverse range of climatic zones, soil types, and agricultural practices. Farming is a crucial sector in the Indian economy, providing livelihoods to a significant portion of the population. The country's agricultural landscape is characterized by:

1) Diverse Cropping Patterns: India cultivates a variety of crops, including staples like rice and wheat, commercial crops like cotton and sugarcane, and horticultural crops like fruits and vegetables. This diversity necessitates region-specific farming strategies and yield predictions.

2) Monsoon Dependency: A substantial part of Indian agriculture relies on monsoon rains. Accurate rainfall predictions and their impact on crop yield are vital for planning and managing agricultural activities.

3) Technological Advancements: The adoption of modern farming techniques, improved seed varieties, and precision agriculture is gradually transforming Indian agriculture. Integrating machine learning models for yield prediction can further enhance productivity and sustainability.

4) Government Initiatives: Various government schemes and policies aim to support farmers by providing subsidies, insurance, and access to markets. Yield prediction models can complement these initiatives by offering data-driven insights for better resource allocation and risk management.""")

    st.header(" How to use the model? ")

    st.markdown("""
Using the crop yield prediction model is straightforward. Follow the steps below to get predictions for crop yields:

1. **Select the State**:
    - Use the dropdown menu to select the state where the crop is being cultivated. This helps the model understand the geographical context of your input.

2. **Select the Crop**:
    - Choose the specific crop for which you want to predict the yield. The model supports a variety of crops, including staples, commercial, and horticultural crops.

3. **Select the Season**:
    - Select the appropriate season from the dropdown menu. This allows the model to consider seasonal variations in crop yield.

4. **Input Soil pH**:
    - Enter the pH level of the soil. Soil pH is a crucial factor that affects crop growth and yield. We recommend using [this youtube guide](https://www.youtube.com/watch?v=mZgxUqoJMcg) in order to measure the pH of your soil at home.

5. **Input Rainfall**:
    - Enter the amount of rainfall (in millimeters) that the area has received. Rainfall is a key determinant of crop health and yield. We recommend using sites like [https://mausam.imd.gov.in/responsive/rainfallinformation.php](https://mausam.imd.gov.in/responsive/rainfallinformation.php) in order to determine the rainfall in your area

6. **Input Temperature**:
    - Enter the average temperature (in degrees Celsius) for the growing season. Temperature influences crop development and yield. We recommend using the [Accuweather](https://www.accuweather.com/) website in order to get the temperature of the area you are living in

7. **Input Area**:
    - Enter the area (in hectares) under cultivation for the selected crop. This helps in understanding the scale of cultivation.

8. **Input Production**:
    - Enter the total production (in tons) of the crop. This historical production data aids in making accurate predictions.

9. **Click the Predict Button**:
    - After entering all the required inputs, click the "Predict" button to get the yield prediction.

### Interpreting the Results

- The model will output the predicted yield for the selected inputs, expressed in tons per hectare.
- Use this information to make informed decisions about crop management, resource allocation, and planning for future agricultural activities.

By following these steps, you can leverage the power of machine learning to get accurate and data-driven crop yield predictions. This can significantly enhance your agricultural practices and contribute to better productivity and sustainability.
""")

if selected == "Crop Yield Predictor":

    working_dir = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open(f'{working_dir}/ensemble.sav', 'rb'))

    # Define the options for each dropdown
    states = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 
            'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 
            'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 
            'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 
            'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 
            'Uttarakhand', 'West Bengal']

    crops = ['Arecanut', 'Barley', 'Banana', 'Blackpepper', 'Brinjal', 'Cabbage', 'Cardamom', 'Cashewnuts', 'Cauliflower', 
            'Coriander', 'Cotton', 'Garlic', 'Grapes', 'Horsegram', 'Jowar', 'Jute', 'Ladyfinger', 'Maize', 
            'Mango', 'Moong', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Potato', 'Rapeseed', 'Ragi', 'Rice', 
            'Sesamum', 'Soyabean', 'Sunflower', 'Sweetpotato', 'Tapioca', 'Tomato', 'Turmeric', 'Wheat']

    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']

    # Collect user inputs
    state = st.selectbox('Select State', states)
    crop = st.selectbox('Select Crop', crops)
    season = st.selectbox('Select Season', seasons)
    pH = st.number_input('pH of the soil')
    rainfall = st.number_input('Rainfall (in mm) ')
    temperature = st.number_input('Temperature')
    area = st.number_input('Area (in hectares) ')
    production = st.number_input('Production (in tons) ')

    # Create a predict button
    if st.button('Predict'):
        # Preprocess input
        state_lower = state.lower()
        crop_lower = crop.lower()
        season_lower = season.lower()

        state_encoded = [0] * (len(states) - 1) if state_lower == 'andaman and nicobar islands' else [1 if s.lower() == state_lower else 0 for s in states if s.lower() != 'andaman and nicobar islands']
        crop_encoded = [0] * (len(crops) - 1) if crop_lower == 'arecanut' else [1 if c.lower() == crop_lower else 0 for c in crops if c.lower() != 'arecanut']
        season_encoded = [0] * (len(seasons) - 1) if season_lower == 'kharif' else [1 if s.lower() == season_lower else 0 for s in seasons if s.lower() != 'kharif']

        input_features = np.array(state_encoded + crop_encoded + season_encoded + [pH, rainfall, temperature, area, production]).reshape(1, -1)
            # Check if any input feature is zero
        if any(feature == 0 for feature in input_features[0]):
            st.error("All input fields must be non-zero.")
        else:
            # Ensure the input_features array has the correct shape
            expected_num_features = len(states) + len(crops) + len(seasons) - 3 + 5  # Exclude the dropped categories and include additional fields
            if input_features.shape[1] != expected_num_features:
                st.error(f"Feature shape mismatch, expected: {expected_num_features}, got: {input_features.shape[1]}")
            else:
                # Make prediction
                predicted_yield = model.predict(input_features)[0]  # Call your prediction function

                # Display prediction
                st.header('Predicted Crop Yield')
                st.write(f'The predicted yield for the selected inputs is: {predicted_yield:.2f} tons/hectare')
