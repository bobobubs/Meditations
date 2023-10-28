import cv2
from moviepy.editor import *
import numpy as np

# Global variables to hold the coordinates of the boxes
start_box = []
end_box = []

def draw_rectangle(event, x, y, flags, param):
    global start_box, end_box, drawing, current_box, img_copy  # Ensure all global variables are declared
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_box = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_copy.copy()
            # Ensure a 1:1 aspect ratio
            dx = dy = max(x - current_box[0][0], y - current_box[0][1])

            # Calculate the corners of the rectangle
            top_left = (current_box[0][0], current_box[0][1])
            bottom_right = (int(current_box[0][0] + dx), int(current_box[0][1] + dy))

            cv2.rectangle(img_temp, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imshow('Draw Start and End Boxes', img_temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Ensure a 1:1 aspect ratio
        dx = dy = max(x - current_box[0][0], y - current_box[0][1])

        # Calculate the corners of the rectangle
        top_left = (current_box[0][0], current_box[0][1])
        bottom_right = (int(current_box[0][0] + dx), int(current_box[0][1] + dy))

        if not start_box:
            start_box = [top_left, bottom_right]
        elif not end_box:
            end_box = [top_left, bottom_right]



def get_boxes(image_path):
    global start_box, end_box, img_copy
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Determine scaling factor to fit the image to your screen
    screen_res = 1920, 1080  # Replace with your screen resolution
    scale_width = screen_res[0] / width
    scale_height = screen_res[1] / height
    scale = min(scale_width, scale_height)

    # Resize the image and the copy used for drawing
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    img_copy = img.copy()

    cv2.imshow('Draw Start and End Boxes', img)
    cv2.setMouseCallback('Draw Start and End Boxes', draw_rectangle)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and start_box and end_box:
            break
    cv2.destroyAllWindows()

    # Scale the boxes back to the original image dimensions
    start_box = [(int(x / scale), int(y / scale)) for x, y in start_box]
    end_box = [(int(x / scale), int(y / scale)) for x, y in end_box]



    return start_box, end_box


def smoothstep(edge0, edge1, x):
    # Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Evaluate polynomial
    return x * x * (3 - 2 * x)

def clamp(x, lower_limit, upper_limit):
    return max(lower_limit, min(x, upper_limit))

def get_audio_duration(audio_path):
    audio = AudioFileClip(audio_path)
    return audio.duration

def create_zoom_video_with_audio(image_path, start_box, end_box, video_path, audio_path):
    audio_duration = get_audio_duration(audio_path)
    video_duration = audio_duration + 3  # add 3 seconds to audio duration
    
    # Load the image
    img = cv2.imread(image_path)
    
    # Set the dimensions of the video frames to maintain a 9:16 aspect ratio
    frame_width = frame_height = 900
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))
    
    # Calculate the number of frames based on the duration and frame rate
    num_frames = int(video_duration * 30)
    
    for i in range(num_frames):
        # Interpolate the coordinates of the box for the current frame
        alpha = i / num_frames
        curr_box = [
            (
                int(start_box[0][0] * (1 - alpha) + end_box[0][0] * alpha),
                int(start_box[0][1] * (1 - alpha) + end_box[0][1] * alpha)
            ),
            (
                int(start_box[1][0] * (1 - alpha) + end_box[1][0] * alpha),
                int(start_box[1][1] * (1 - alpha) + end_box[1][1] * alpha)
            )
        ]
        
        # Extract the region of interest from the image
        roi = img[curr_box[0][1]:curr_box[1][1], curr_box[0][0]:curr_box[1][0]]
        
        # Resize the region of interest to the dimensions of the video frames
        frame = cv2.resize(roi, (frame_width, frame_height))
        
        # Write the frame to the video
        out.write(frame)
    
    # Release the video writer
    out.release()
    
    # Load the generated video and the audio file
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Set the audio of the video
    final_video = video.set_audio(audio)

    # Write the result to a file
    final_video.write_videofile("output_with_audio.avi", codec="libx264", fps=24)
start_box, end_box = get_boxes('./images/1_1.png')
create_zoom_video_with_audio('./images/1_1.png', start_box, end_box, 'output.avi', './audio/1_1.mp3')








