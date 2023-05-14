from flask import Flask, request, jsonify
# OpenCV library for computer vision tasks
import cv2
# Library for numerical operations
import numpy as np
# Library for data manipulation and analysis
import pandas as pd
# Library for creating and training deep learning models
import tensorflow as tf
# Function from sklearn library to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Function from sklearn library to calculate the accuracy of a machine learning model
from sklearn.metrics import accuracy_score
# library to add AI sound
import pyttsx3

app = Flask(__name__)


class Height:

    def __init__(self, CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH,
                 CAM_SENSOR_WIDTH, CAM_SENSOR_HEIGHT, MIN_OBJECT_HEIGHT):
        self.CAM_WIDTH = CAM_WIDTH
        self.CAM_HEIGHT = CAM_HEIGHT
        self.CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH
        self.CAM_SENSOR_WIDTH = CAM_SENSOR_WIDTH
        self.CAM_SENSOR_HEIGHT = CAM_SENSOR_HEIGHT
        self.MIN_OBJECT_HEIGHT = MIN_OBJECT_HEIGHT


def play_audio():
    # Set the text for the audio message
    message1 = "I am about to measure your height now"
    message2 = "Although I reach a precision upto ninety eight percent"
    message = message1 + message2

    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)  # Set the speaking rate

    # Convert the message to speech and play the audio
    engine.say(message)
    engine.runAndWait()


# function to display the height in window
def display_height(output_file, frame, max_contour, height_pixels, CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH, CAM_SENSOR_WIDTH, CAM_SENSOR_HEIGHT, MIN_OBJECT_HEIGHT):
    # Only process the contour if its height exceeds the minimum threshold, if the height is greater than the min height which is 50 cm
    if height_pixels >= MIN_OBJECT_HEIGHT:
        # Convert the height to centimeters using camera calibration
        # Find the width of the camera sensor in centimeters
        sensor_width_cm = CAM_SENSOR_WIDTH / 10
        # Find the height of the camera sensor in centimeters
        sensor_height_cm = CAM_SENSOR_HEIGHT / 10
        # Find the width of the single pixel in centimeters
        pixel_width = sensor_width_cm / CAM_WIDTH
        # Find the height of the single pixel in centimeters
        pixel_height = sensor_height_cm / CAM_HEIGHT
        # Calculate the height of the object which is human in centimeteres by multiplying the above variables in centimeters
        height_cm = height_pixels * pixel_height * CAM_FOCAL_LENGTH
        # print("hello",  height_cm)

        # Draw a bounding box around the person, find the vertices of the minimum rectangle that bounds the max contour
        # The function returns a tuple that includes the center point, dimensions(width, height), and angle of the minimum area reectangle
        box = cv2.boxPoints(cv2.minAreaRect(max_contour))
        # Convert floating array to integer array to the nearest rounding integer
        box = np.int0(box)
        # Draw conntours of the object detected in the frame, the color of the contour is green, 2 means thickness of the pixels
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # Display the height on the person, 10 and 30 specify the coordinates, then font type, font scale which is 0.8, and color is red, 2 means thinkness
        cv2.putText(frame, "Height: {:.2f} cm".format(
            height_cm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write the height to the output file
        output_file.write("{:.2f}\n".format(height_cm))










# fucntion to define the apply_dip_techniques_to_detect_height
def apply_dip_techniques_to_detect_height(frame):
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Guassinan blur to reduce noise and smooth the image, the kernal size is 7x7 and 0 means stdviation generate automatically
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Apply Canny edge detection, the lower and upper thresholds control the sensitivity, 3 means Sobel karnel size which is 3x3 here
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    # Find contours in the image, the second argument is the contour retrieval mode and the third argument is the contour approximation method
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area, set initial max_area to 0
    max_area = 0
    # Set inkitial max_contour to None
    max_contour = None
    # Loop through the list of contours one by one
    for contour in contours:
        # calculate the contour area by multiplying the no. of pixels inside the contour and multiplying it by the area of each pixel in square units
        area = cv2.contourArea(contour)
        # If the contour area is greater then max area
        if area > max_area:
            # Assign max area to the contour area
            max_area = area
            # Assign max contour to the selected contour from the list of the contours as in the loop
            max_contour = contour

    # If a contour was found, calculate the height of the person, if the max contour above is not None it has some value
    if max_contour is not None:
        # Find the height of the person in pixels, it returns a tuple containing the center coordinates, width, height and angle of rotation of the rectangle.
        # first [1] refers to the width and height, second [1] refers to specifically to the height
        # overall this gives the height of the largest contour
        height_pixels = cv2.minAreaRect(max_contour)[1][1]

        return max_contour, height_pixels









# function to define the integration_tensorflow
def prediction_results_from_tensorflow(output_file, rounded_predictions):
    # Store the predicted labels in a text file named output.txt
    with open(output_file, "w") as file:
        # The values recived from the above rounded_predictions, select one by one
        for label in rounded_predictions:
            # If the label is 0 means short, put this to the output.txt file
            if label == 0:
                file.write("short\n")
            # If the label is 1 means medium, put this to the output.txt file
            elif label == 1:
                file.write("medium\n")
            # If the label is 2 means medium, put this to the output.txt file
            else:
                file.write("tall\n")







# function to define the integraion_tensorflow
def integraion_tensorflow(data, numerical_labels):
    # Split the data into training and testing sets, trian_test_split splits the data into training and testing datasets randomly
    # Test size percentage is 20% and remianing 80% is for the training dataset
    train_data, test_data, train_labels, test_labels = train_test_split(
        data["height"], numerical_labels, test_size=0.2, random_state=0
    )

    # Create the machine learning model, creates a simple neural network model from Keras API from Tensorflow.
    # The model has only one layer, a dense layer with 1 unit. The model takes only one input feature which is the height of the object
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    # Compile the model using Adam optimizer which is a popular model of deep learning, loss is mean squared error which is the difference between predicted and actual labels
    model.compile(optimizer=tf.keras.optimizers.Adam(1),
                  loss="mean_squared_error")

    # Train the model for 100 epochs
    model.fit(train_data.values.astype(float), train_labels, epochs=100)

    # Test the model on the test_data dataset generated above
    predictions = model.predict(test_data.values.astype(float)).flatten()
    # Round the predictions from the test_data and convert the float values to integers
    rounded_predictions = [int(round(prediction))
                           for prediction in predictions]

    # Call the prediction_results_from_tensorflow with the following paramters
    prediction_results_from_tensorflow("output.txt", rounded_predictions)






@app.route('/start_video_to_detect_height')
# function to detect human height using live video
def start_video_to_detect_height(CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH,
                     CAM_SENSOR_WIDTH, CAM_SENSOR_HEIGHT, MIN_OBJECT_HEIGHT):
    # Start the video capture
    cap = cv2.VideoCapture(0)
    # Set the width of the video capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    # Set the width of the video capture
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Open/Create a txt file for writing named as heights.txt
    output_file = open("heights.txt", "w")

    # Loop runs infinitely until a stop condtion is applied in this case q is pressed
    while True:
        # Try block if the code executes without any error
        try:
            # Read a frame from the video capture
            ret, frame = cap.read()

        # Catch block to catch the exception if it occurs
        except Exception as e:
            # If any other exception occurs, print the error message and continue the loop
            print("Error:", e)
            # loop contines again if an exception is caught
            continue

        # Call the apply_dip_techniques_to_detect_height with no paramters
        max_contour, height_pixels = apply_dip_techniques_to_detect_height(
            frame)

        display_height(output_file, frame, max_contour, height_pixels, CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH,
                     CAM_SENSOR_WIDTH, CAM_SENSOR_HEIGHT, MIN_OBJECT_HEIGHT)

        # Display the frame
        cv2.imshow("Height Detection", frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    # Destroy the window
    cv2.destroyAllWindows()

    # Close the output file
    output_file.close()

    # Read the heights from the output file into a pandas dataframe
    df = pd.read_csv("heights.txt", header=None, names=["height"])

    # Export the dataframe to an excel file
    writer = pd.ExcelWriter("heights.xlsx", engine="xlsxwriter")
    # Exports height column of a Pandas dataframe df to an excel file
    df['height'].to_excel(writer, sheet_name="Sheet1", index=False)
    # Retrives the worksheet named Sheet1 from the Excel writer object created earlier
    worksheet = writer.sheets["Sheet1"]
    # Close the writer object
    writer.close()

    # Load the heights data from the heights.xlsx file
    data = pd.read_excel("heights.xlsx")

    # Define the low threshold for height
    short_threshold = 150
    # Define the upper threshold for height
    tall_threshold = 180

    # Convert the heights into labels
    labels = []
    # Select one height from the height column
    for height in data["height"]:
        # If the height is less than the lower threshold
        if height < short_threshold:
            # Append short word to the labels list
            labels.append("short")
        # If the height is greater than upper threshold
        elif height > tall_threshold:
            # Append tall word to the labels list
            labels.append("tall")
        # If the height is nor greater nor lower
        else:
            # Append medium word to the labels list
            labels.append("medium")

    # Convert the labels into numerical values, 0 for short, 1 for medium, 2 for tall
    label_encoder = {"short": 0, "medium": 1, "tall": 2}
    # Creates a numpy array, maps each label (string) to the unique integer value
    numerical_labels = np.array([label_encoder[label] for label in labels])

    # Call the integraion_tensorflow function with the following paramters
    integraion_tensorflow(data, numerical_labels)


# accuracy = accuracy_score(test_labels, rounded_predictions)
# print("Accuracy: {:.2f}%".format(accuracy * 100))


def main():
    # Width of the camera feed
    CAM_WIDTH = 1280
    # Height of the camera feed
    CAM_HEIGHT = 720

    # Focal length of the camera lens (estimated from calibration)
    CAM_FOCAL_LENGTH = 1015
    # Width of the camera sensor (estimated from calibration)
    CAM_SENSOR_WIDTH = 5.6
    # Height of the camera sensor (estimated from calibration)
    CAM_SENSOR_HEIGHT = 4.2

    # Minimum height of the detected object in pixels
    MIN_OBJECT_HEIGHT = 50


    # Play the audio message before starting the video capture
    play_audio()

    
    # Call the detect height function with the following parameters
    start_video_to_detect_height(CAM_WIDTH, CAM_HEIGHT, CAM_FOCAL_LENGTH,
                     CAM_SENSOR_WIDTH, CAM_SENSOR_HEIGHT, MIN_OBJECT_HEIGHT)


if __name__ == "__main__":
    main()
