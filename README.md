# Hand-Gesture-Detection
In this project, my team and I created a Hand Gesture Detection model using Google's Teachable machine to help mute people communicate easily

🤟 Hand Gesture Detection for Mute Communication
💡 Project Overview
This project is a Hand Gesture Detection System developed using Google Teachable Machine. It is designed to assist mute individuals in communicating more effectively by recognizing specific hand gestures and translating them into readable text or voice outputs.

👥 Team Members

Muhammad Ibrahim

Muhammad Hamza Nawaz

🎯 Objectives

  Help people with speech impairments communicate easily.
  
  Use machine learning to detect hand gestures with high accuracy.
  
  Provide a user-friendly interface for real-time gesture recognition.

🛠️ Technologies Used

  Google Teachable Machine (Image Project)
  
  TensorFlow.js (for deploying the model)
  
  Webcam (for real-time gesture input)

🚀 How It Works

  Training the Model
    We used Google Teachable Machine to train the model on various hand gestures. Each gesture represents a different phrase or letter (like "Hello", "Yes", "No", etc.).
  
  Exporting the Model
    After training, the model was exported and integrated into a web application using TensorFlow.js.
  
  Real-Time Detection
    The webcam captures the user's hand gestures, and the model predicts the corresponding gesture class in real time.
  
  Output Display
    The recognized gesture is displayed on the screen as text (and optionally converted to speech using Web Speech API).

📈 Future Improvements
  Add support for more gestures or full sign language alphabet.
  
  Improve accuracy in different lighting conditions.
  
  Add voice output using the Web Speech API.
  
  Deploy the web app online for easier access.

❤️ Acknowledgements
  Google Teachable Machine
  
  TensorFlow.js
