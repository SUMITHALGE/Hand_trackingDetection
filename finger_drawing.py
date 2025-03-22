import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hand Detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only track one hand for drawing
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a blank canvas for drawing
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Default drawing settings
    drawing_color = (0, 255, 0)  # Green color
    thickness = 4
    
    # Track previous position of index finger
    prev_x, prev_y = 0, 0
    is_drawing = False
    
    # Drawing mode (index finger up for drawing)
    drawing_mode = True
    
    # Eraser mode (index and middle finger up for erasing)
    eraser_mode = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break
        
        # Flip the image horizontally for a more intuitive mirror view
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
                
                # Get index and middle finger landmarks
                index_finger_tip = hand_landmarks.landmark[8]
                middle_finger_tip = hand_landmarks.landmark[12]
                
                # Convert normalized coordinates to pixel coordinates
                curr_x = int(index_finger_tip.x * width)
                curr_y = int(index_finger_tip.y * height)
                
                # Check if index finger is up and middle finger is down (drawing mode)
                index_up = index_finger_tip.y < hand_landmarks.landmark[6].y
                middle_up = middle_finger_tip.y < hand_landmarks.landmark[10].y
                
                if index_up and not middle_up:
                    # Drawing mode
                    eraser_mode = False
                    
                    # Start drawing from the second frame
                    if is_drawing and prev_x > 0 and prev_y > 0:
                        # Draw line on canvas
                        cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), drawing_color, thickness)
                    
                    # Update previous position
                    prev_x, prev_y = curr_x, curr_y
                    is_drawing = True
                
                elif index_up and middle_up:
                    # Eraser mode (both index and middle finger up)
                    eraser_mode = True
                    if is_drawing and prev_x > 0 and prev_y > 0:
                        # Erase by drawing black
                        cv2.circle(canvas, (curr_x, curr_y), thickness * 5, (0, 0, 0), -1)
                    
                    # Update previous position
                    prev_x, prev_y = curr_x, curr_y
                    is_drawing = True
                else:
                    # Reset when no fingers are up
                    is_drawing = False
                    prev_x, prev_y = 0, 0
                
                # Display mode text
                mode_text = "Drawing Mode" if not eraser_mode else "Eraser Mode"
                cv2.putText(image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Combine the canvas with the camera feed (50% transparency)
        combined_image = cv2.addWeighted(image, 1.0, canvas, 0.5, 0)
        
        # Display the resulting frame
        cv2.imshow('Finger Drawing', combined_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Clear canvas if 'c' is pressed
        if key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Change color if 'r', 'g', 'b', 'w' is pressed
        elif key == ord('r'):
            drawing_color = (0, 0, 255)  # Red (BGR format)
        elif key == ord('g'):
            drawing_color = (0, 255, 0)  # Green
        elif key == ord('b'):
            drawing_color = (255, 0, 0)  # Blue
        elif key == ord('w'):
            drawing_color = (255, 255, 255)  # White
        elif key == ord('y'):
            drawing_color = (0, 255, 255)  # Yellow
        elif key == ord('p'):
            drawing_color = (255, 0, 255)  # Purple
        
        # Increase/decrease brush thickness with '+' and '-'
        elif key == ord('+') or key == ord('='):
            thickness = min(thickness + 1, 20)
        elif key == ord('-') or key == ord('_'):
            thickness = max(thickness - 1, 1)
        
        # Save the canvas if 's' is pressed
        elif key == ord('s'):
            cv2.imwrite('drawing.png', canvas)
            print("Drawing saved to 'drawing.png'")
        
        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
