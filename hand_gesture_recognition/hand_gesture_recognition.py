# Step 2: Import required libraries
import cv2
import mediapipe as mp

# Step 3: Initialize MediaPipe Hands module and drawing utility
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Step 4: Set up webcam capture
cap = cv2.VideoCapture(0)

# Step 5: Create a loop for real-time processing
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        # Step 6: Inside the loop, read frames from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Step 7: Convert and flip the image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Step 8: Process the image using MediaPipe Hands
        results = hands.process(image)

        # Convert the image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Step 9: Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Step 10: Display the annotated image
        cv2.imshow('MediaPipe Hands', image)

        # Step 11: Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Step 12: Release the webcam and close the window when done
cap.release()
cv2.destroyAllWindows()
