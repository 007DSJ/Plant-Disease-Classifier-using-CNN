import os
import json
from PIL import Image
from bs4 import BeautifulSoup
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import folium
from streamlit_folium import st_folium,folium_static

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to calculate correct and incorrect predictions
def calculate_accuracy(predictions, true_labels):
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    total = len(true_labels)
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy


def scrape_google_search(query):
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    
    url = f"https://www.google.com/search?q={query}-benefits"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    search_results = []
    for item in soup.select(".tF2Cxc"):
        title = item.select_one(".DKV0Md").text
        link = item.select_one(".yuRUbf a")["href"]
        search_results.append({"title": title, "link": link})
    
    driver.quit()
    return search_results

OPENFARM_BASE_URL = "https://openfarm.cc/en/crops/"

def fetch_optimal_conditions(query):
    try:
        url = OPENFARM_BASE_URL + query
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            table_rows = soup.find("table", class_="crop").find_all("tr")
            optimal_conditions = {}
            for row in table_rows:
                cells = row.find_all("td")
                if len(cells) == 2:
                    key = cells[0].text.strip()
                    value = cells[1].text.strip()
                    optimal_conditions[key] = value
            return optimal_conditions
        else:
            return "Failed to fetch data. Please try again later."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_plantation_locations(query):
    try:
        with open('locations.json', 'r') as f:
            locations = json.load(f)
            if query in locations:
                return [(query.capitalize(), loc["latitude"], loc["longitude"]) for loc in locations[query]]
            else:
                return []
    except FileNotFoundError:
        print("Locations file not found.")
        return []



# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    resized_img = image.resize((150, 150))
    st.image(resized_img)
    
    if st.button('Classify'):
        # Preprocess the uploaded image and predict the class
        prediction = predict_image_class(model, uploaded_image, class_indices)
        true_label = uploaded_image.name.split('.')[0]  # Assuming the filename contains the true label
        st.success(f'Prediction: {str(prediction)}')

        # Fetch information from Google search
        search_query = prediction.split('___')[0]
        search_results = scrape_google_search(search_query)

        st.title('Optimal Growing Conditions for '+search_query.capitalize())

        query = search_query
        if query:
            optimal_conditions = fetch_optimal_conditions(query.lower())
            st.subheader(f"Optimal Growing Conditions for {query.capitalize()}:")
            if isinstance(optimal_conditions, dict):
                for key, value in optimal_conditions.items():
                    st.write(f"- **{key}**: {value}")
            else:
                st.write(optimal_conditions)

        # Display search results
        st.subheader(f"Addition Information for {query}:")
        for item in search_results[:3]:  # Show top 3 results
            st.write(f"- [{item['title']}]({item['link']})")

        # Display plantation locations on the map
        st.title('Plantation Locations')
        locations = get_plantation_locations(query)
        if locations:
            m = folium.Map(location=[locations[0][1], locations[0][2]], zoom_start=6)
            for loc in locations:
                # folium.Marker(location=[loc[1], loc[2]], popup=loc[0]).add_to(m)
                folium.CircleMarker(location=[loc[1], loc[2]], popup=loc[0],
                        radius=5, color='red').add_to(m)
            folium_static(m)
        else:
            st.write("No plantation locations found.")
