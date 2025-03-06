from flask import Flask, render_template, request
import tensorflow as tf
from transformers import AutoTokenizer
import zipfile
import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Function to unzip the saved model
def unzip_model(zip_path, extract_to_path):
    if not os.path.exists(extract_to_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)

# Function to load the trained model from SavedModel directory
def load_trained_model(model_dir):
    model = tf.saved_model.load(model_dir)
    print("Model loaded successfully.")
    return model

# Function to tokenize the input text
def tokenize_input(text, tokenizer):
    return tokenizer(text, return_tensors='tf', padding=True, truncation=True)

# Function to predict the class based on input
def predict_class(model, tokenizer, input_text):
    print(f"Predicting class for input: {input_text}")
    inputs = tokenize_input(input_text, tokenizer)
    prediction = model(inputs)

    # Handle output shape
    if len(prediction.shape) > 1:
        predicted_class = prediction[0].numpy().argmax()
        confidence = prediction[0].numpy()[predicted_class]
    else:
        predicted_class = prediction.numpy().argmax()
        confidence = prediction.numpy()[predicted_class]

    # Define class labels
    label_map = {
        0: "Satisfied",
        1: "Neutral",
        2: "Slightly Dissatisfied",
        3: "Dissatisfied",
        4: "Highly Dissatisfied"
    }

    predicted_label = label_map.get(predicted_class, "Unknown")
    print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")
    return predicted_label, confidence

# Function to assign an agent dynamically based on label and availability
def assign_agent_to_complaint(connection, complaint_id, label):
    label_to_experience = {
        "Satisfied": "No Agent Required",
        "Neutral": ["low", "mid", "high"],
        "Slightly Dissatisfied": ["low", "mid", "high"],
        "Dissatisfied": ["mid", "high"],
        "Highly Dissatisfied": ["high"]
    }
    experience_options = label_to_experience.get(label, ["low", "mid", "high"])

    try:
        cursor = connection.cursor(dictionary=True)
        print(f"Assigning agent for complaint ID: {complaint_id}, Label: {label}")

        if label == "Satisfied":
            return "No Agent Required"

        for experience_level in experience_options:
            query = """SELECT agent_id, agent_name 
                       FROM agents 
                       WHERE experience_level = %s AND status = 'free' 
                       LIMIT 1"""
            cursor.execute(query, (experience_level,))
            agent = cursor.fetchone()
            if agent:
                print(f"Agent found: {agent['agent_name']} with experience level: {experience_level}")
                update_agent_status = """UPDATE agents 
                                         SET status = 'occupied', assigned_complaint_id = %s 
                                         WHERE agent_id = %s"""
                cursor.execute(update_agent_status, (complaint_id, agent['agent_id']))
                connection.commit()
                return agent['agent_name']

        print("No available agent found.")
        return "No available agent.Please try again later"
    except Error as e:
        print(f"Error while assigning agent: {e}")
        return "Error"
    finally:
        cursor.close()

# Function to store complaint data in MySQL
def store_complaint_in_db(complaint_text, label, confidence):
    try:
        print("Connecting to MySQL database...")
        connection = mysql.connector.connect(
            host='localhost',
            database='complaint_db',
            user='root',
            password='Admin@123'
        )
        if connection.is_connected():
            print("Database connection successful.")
            cursor = connection.cursor(dictionary=True)

            # Insert complaint into database
            insert_query = """INSERT INTO complaints (complaint_text, classified_label, confidence, timestamp) 
                              VALUES (%s, %s, %s, %s)"""
            print(f"Inserting complaint: '{complaint_text}', Label: {label}, Confidence: {confidence}")
            cursor.execute(insert_query, (complaint_text, label, float(confidence), datetime.now()))
            connection.commit()

            complaint_id = cursor.lastrowid
            print(f"Complaint inserted with ID: {complaint_id}")
            agent_name = assign_agent_to_complaint(connection, complaint_id, label)

            # Fetch the assigned agent's experience level
            if agent_name and agent_name != "No Agent Required" and agent_name != "No available agent":
                experience_query = """SELECT experience_level 
                                      FROM agents 
                                      WHERE assigned_complaint_id = %s LIMIT 1"""
                cursor.execute(experience_query, (complaint_id,))
                agent_data = cursor.fetchone()
                experience_level = agent_data['experience_level'] if agent_data else "N/A"
            else:
                experience_level = "N/A"

            cursor.close()
            return complaint_id, agent_name, experience_level
    except Error as e:
        print(f"Error while connecting to MySQL or inserting complaint: {e}")
        return "Error", "Error", "N/A"
    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed.")

# Load model and tokenizer
model_dir = "Nishanth_saved_model"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Unzip and load model
unzip_model("Nishanth_saved_model.zip", model_dir)
model = load_trained_model(model_dir)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling text classification requests
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    predicted_label, confidence = predict_class(model, tokenizer, user_input)

    # Store complaint in the database and get the additional details
    complaint_id, assigned_agent, experience_level = store_complaint_in_db(user_input, predicted_label, confidence)

    # Ensure confidence is a valid float value
    confidence = confidence if confidence is not None else 0.0

    return render_template('index.html', 
                           prediction=predicted_label, 
                           confidence=confidence, 
                           user_input=user_input, 
                           assigned_agent=assigned_agent,
                           complaint_id=complaint_id,
                           experience_level=experience_level)

if __name__ == "__main__":
    app.run(debug=True)
