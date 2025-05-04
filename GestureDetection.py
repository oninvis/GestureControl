import cv2
import HandTracking as hp
import subprocess
import os
import time
# this coordinates with mac os to set the volume
def set_system_volume(volume):
    # this sets the limit for the volume to be between 0 to 100
    volume = max(0,min(100,volume))
    print(f"Setting volume to {volume}")
    # executes apple script command via mac os to set it to the given volume
    os.system(f"osascript -e 'set volume output volume {volume}'")

# this function will convert the distance between thumb and index finger to volume tracker
def convert_distance_to_volume(distance,min_dist = 0.05 , max_dist = 0.5 , min_vol = 0 , max_vol = 100):
    if distance is None:
        return 0
    # setting a range for distance
    distance = max(min_dist,min(max_dist,distance))
    # this equation is to calculate volume ( this is taken from Grok I don't know about this equation's math ) , known as linear interpolation
    volume = min_vol + (distance - min_dist) * (max_vol - min_vol) / (max_dist - min_dist)
    return int(volume)

fixed_volume = None
last_volume = None
HoldTime = 2

cap = cv2.VideoCapture(0)
hand_tracker = hp.HandDetector(False,2,0.5,0.5)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = hand_tracker.detectHands(img)  # Detect hands
    hand_landmarks = hand_tracker.find_landmarks(img)  # Get landmarks

    if hand_landmarks:

        distance = hand_tracker.calculateDistance(4, 8, hand_landmarks)  # Distance between thumb and index finger
        if distance is not None:
            if fixed_volume is None:
                volume = convert_distance_to_volume(distance)
                set_system_volume(volume)  # Set volume
                cv2.putText(img, f"Volume: {volume}% Dist: {distance:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if last_volume is None or abs(volume-last_volume) >2:
                        last_volume = volume
                        start_time = time.time()
                else:
                        hold_time = time.time() - start_time
                        if hold_time > HoldTime:
                            fixed_volume = volume
            else:
                set_system_volume(fixed_volume)
                cv2.putText(img, f"Fixed Volume: {fixed_volume}% (Hand detected)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Could not calculate distance",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if fixed_volume is not None:
            cv2.putText(img, f"Fixed Volume: {fixed_volume}% (No hand)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(img, "No hands detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fixed_volume = None  # Reset when hand disappears
        last_volume = None




    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()