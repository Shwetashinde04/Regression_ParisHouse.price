import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Paris.csv\ParisHousing.csv")  # Make sure you have a CSV file with your dataset

# Prepare data
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
scaler.fit(X_train)  # Fit scaler on training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Function to make predictions
def predict(model, data):
    scaled_data = scaler.transform(data)
    return model.predict(scaled_data)

# Create Streamlit app
st.title('Paris House Price Prediction')

# Add image by providing the local file path
st.image(r"https://media.istockphoto.com/id/604371970/photo/cityscape-of-paris-by-the-sunset.jpg?s=1024x1024&w=is&k=20&c=7i-uBZa62P6hsTElyQEbit1fovHzNlZYY8pxace7w6U=", caption='Your Image Caption', use_column_width=True)


# Sidebar for user input
st.sidebar.header('Enter House Details')
area = st.sidebar.slider('Area (sqft)', min_value=0, max_value=5000, value=1000)
rooms = st.sidebar.slider('Number of Rooms', min_value=1, max_value=10, value=2)
balcony = st.sidebar.selectbox('Balcony', [0, 1])
garden = st.sidebar.selectbox('Garden', [0, 1])
squareMeters = st.sidebar.slider("squareMeters",min_value=0, max_value=80771,value=1000)
numberOfRooms = st.sidebar.slider("numberOfRooms",min_value=3, max_value=100, value=1000)
hasYard = st.sidebar.selectbox("hasYard", [0,1])
hasPool = st.sidebar.selectbox("hasPool",[0,1])
floors = st.sidebar.slider("floors",min_value=1, max_value=100, value=100)
cityCode = st.sidebar.slider("cityCode",min_value=3, max_value=99953, value=1000)
cityPartRange = st.sidebar.slider("cityPartRange",min_value=1, max_value=10, value=1000)
numPrevOwners = st.sidebar.slider("numPrevOwners",min_value=1, max_value=10, value=1000)
made = st.sidebar.slider("made",min_value=1990, max_value=2021, value=1000)
isNewBuilt = st.sidebar.radio("isNewBuilt",[0,1])
hasStormProtector = st.sidebar.radio("hasStormProtector",[0,1])
basement = st.sidebar.slider("basement",min_value=0, max_value=1000, value=1000)
attic = st.sidebar.slider("attic",min_value=0, max_value=1000, value=1000)
garage = st.sidebar.slider("garage",min_value=100, max_value=1000, value=1000)
hasStorageRoom = st.sidebar.radio("hasStorageRoom",[0,1])
hasGuestRoom = st.sidebar.radio("hasGuestRoom",[0,1])


# Button to trigger prediction
if st.sidebar.button('Predict'):
   isNewBuilt = st.sidebar.selectbox("isNewBuilt",[0,1])
query_point = np.array([[squareMeters,attic,numberOfRooms,hasYard,hasPool,floors,cityCode,garage,cityPartRange,numPrevOwners,made,isNewBuilt,hasStormProtector,basement,hasGuestRoom,hasStorageRoom]])
linear_reg_pred = predict(linear_reg, query_point)


st.write('### Linear Regression Prediction:', linear_reg_pred[0])



    # Optionally, you can calculate Mean Squared Error
    # y_pred = linear_reg.predict(X_test_scaled)
    # mse = calculate_mse(y_test, y_pred)
    # st.write('Mean Squared Error:', mse)
