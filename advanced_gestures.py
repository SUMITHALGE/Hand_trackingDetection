import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hand Detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def detect_dynamic_gestures(landmarks_history, current_landmarks):
    """Detect dynamic gestures based on hand landmark history"""
    # This function could be expanded to detect swipes, circular motions, etc.
    # Currently just a placeholder for future implementation
    if len(landmarks_history) < 10:
        return "Collecting Data..."
    
    # Example: Detect a simple left-to-right swipe
    if (landmarks_history[0][0].x - landmarks_history[-1][0].x) > 0.3:
        return "Swipe Right"
    elif (landmarks_history[-1][0].x - landmarks_history[0][0].x) > 0.3:
        return "Swipe Left"
    
    return None

def recognize_advanced_gesture(hand_landmarks):
    """Recognize more complex static gestures"""
    landmarks = hand_landmarks.landmark
    
    # Finger tips and bases
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [1, 5, 9, 13, 17]
    
    # Check if fingers are extended
    extended_fingers = []
    
    # Thumb is special - compare with wrist and index finger base
    thumb_extended = landmarks[4].x > landmarks[3].x if landmarks[17].x > landmarks[5].x else landmarks[4].x < landmarks[3].x
    extended_fingers.append(1 if thumb_extended else 0)
    
    # For other fingers
    for i in range(1, 5):
        if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
            extended_fingers.append(1)  # Extended
        else:
            extended_fingers.append(0)  # Flexed
    
    # Rock gesture: thumb, pinky and index finger extended
    if extended_fingers == [1, 1, 0, 0, 1]:
        return "Rock On"
    
    # OK gesture: thumb and index finger form a circle
    thumb_index_dist = calculate_distance(landmarks[4], landmarks[8])
    if thumb_index_dist < 0.05 and extended_fingers[2:] == [1, 1, 1]:
        return "OK"
    
    # Thumbs down
    if extended_fingers == [1, 0, 0, 0, 0] and landmarks[4].y > landmarks[9].y:
        return "Thumbs Down"
    
    # Spider-man: thumb, index, and pinky extended
    if extended_fingers == [1, 1, 0, 0, 1]:
        return "Spider-Man"
        
    return None

def main():
    cap = cv2.VideoCapture(0)
    landmarks_history = []
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break
        
        # Flip the image horizontally
        image = cv2.flip(image, 1)
        
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Store landmarks for dynamic gesture recognition
                if len(landmarks_history) > 20:
                    landmarks_history.pop(0)
                landmarks_history.append(hand_landmarks.landmark)
                
                # Recognize advanced static gesture
                gesture = recognize_advanced_gesture(hand_landmarks)
                
                # Try to recognize dynamic gesture if no static gesture was detected
                if gesture is None:
                    gesture = detect_dynamic_gestures(landmarks_history, hand_landmarks.landmark)
                
                # Display gesture name
                if gesture:
                    cv2.putText(
                        image,
                        f"Gesture: {gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
        
        # Display the resulting frame
        cv2.imshow('Advanced Hand Gesture Detection', image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
