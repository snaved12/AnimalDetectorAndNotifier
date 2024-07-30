import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
import requests

# Load the trained model
model = load_model(r'C:\Users\ssyed\Downloads\ResNet50_DEL.h5')

# Define the labels
labels = ['Butterfly', 'Caterpillar', 'Chicken', 'Crab', 'Deer', 'Duck', 'Eagle', 'Fish',
          'Frog', 'Giraffe', 'Goose', 'Horse', 'Jellyfish', 'Ladybug', 'Lion', 'Lizard',
          'Monkey', 'Owl', 'Parrot', 'Penguin', 'Shark', 'Snail', 'Snake', 'Sparrow',
          'Spider', 'Squirrel', 'Tiger', 'Tortoise', 'Whale']

# Define dangerous animals
dangerous_animals = ['Lion', 'Tiger', 'Shark', 'Snake']

# Pushover configuration
pushover_user_key = 'uikpj58koj8ca165z3tztvc8gb1nus'  # Replace with your Pushover user key
pushover_api_token = 'ac1enmt1d6we7f1bidv8ud5q4dpqod'  # Replace with your Pushover API token

# Function to predict the class of an input frame
def predict_animal(frame, model, labels):
    # Preprocess the frame
    img_resized = cv.resize(frame, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Get the label of the predicted class
    predicted_label = labels[predicted_class[0]]
    
    return predicted_label

# Function to send data to ThingSpeak
def send_to_thingspeak(value):
    api_key = '7UDSR5KQTPF5C9KT'  # Replace with your ThingSpeak Write API key
    url = f'https://api.thingspeak.com/update?api_key={api_key}&field1={value}'
    response = requests.get(url)
    if response.status_code == 200:
        print('Data sent to ThingSpeak successfully.')
    else:
        print('Failed to send data to ThingSpeak.')

# Function to send notification via Pushover
def send_pushover_notification(message):
    url = 'https://api.pushover.net/1/messages.json'
    data = {
        'token': pushover_api_token,
        'user': pushover_user_key,
        'message': message
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print('Notification sent successfully.')
    else:
        print('Failed to send notification.')

# Function to process video input and detect animals
def process_video(video_path, model, labels, dangerous_animals):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read and process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the animal in the current frame
        predicted_label = predict_animal(frame, model, labels)
        
        # Check if the predicted animal is dangerous
        if predicted_label in dangerous_animals:
            send_to_thingspeak(1)
            #send_pushover_notification(f"Dangerous animal detected: {predicted_label}")
            send_pushover_notification(f"{predicted_label} Detected nearby, Please find a shelter for safety")
            print(f"Dangerous animal detected: {predicted_label}")
        
        # Display the frame with the predicted label
        cv.putText(frame, f"Predicted Animal: {predicted_label}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('Animal Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

# Test the function with an example video
video_path = r'C:\Users\ssyed\OneDrive\Desktop\The Loin Walk HD _ Slow Motion _ Wild Life.mp4'
process_video(video_path, model, labels, dangerous_animals)
