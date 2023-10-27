import cv2
import numpy as np

def get_rois(img):
    # Use OpenCV's saliency detection to find salient regions
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(img)
    saliency_map = (saliency_map * 255).astype(np.uint8)

    # Threshold the saliency map and find contours
    _, thresh_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rois.append((x, y, x+w, y+h))
    return rois

def apply_ken_burns(image_path, video_path, duration=5, fps=30):
    # Load the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    rois = get_rois(img)
    print(rois)

    # Define the new dimensions for a 9:16 aspect ratio
    new_width = width
    new_height = int(width * 16 / 9)
    x_offset = 0
    y_offset = (new_height - height) // 2  # Center vertically

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Calculate the number of frames
    total_frames = duration * fps

    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_width, roi_height = x2 - x1, y2 - y1
        start_factor, end_factor = 1.0, 0.7
        start_position = (x1, y1)
        end_position = (
            x1 + (roi_width - roi_width * end_factor) // 2,
            y1 + (roi_height - roi_height * end_factor) // 2
        )

        for frame_num in range(total_frames):
            # Interpolate the zoom/pan factors and positions
            factor = np.interp(frame_num, [0, total_frames], [start_factor, end_factor])
            position = (
                np.interp(frame_num, [0, total_frames], [start_position[0], end_position[0]]),
                np.interp(frame_num, [0, total_frames], [start_position[1], end_position[1]])
            )

            # Generate the transformation matrix
            M = np.float32([[factor, 0, position[0]], [0, factor, position[1]]])

            # Apply the affine transformation
            dst = cv2.warpAffine(img, M, (width, height))

            # Write the frame to the video
            out.write(dst)

    # Release the video writer
    out.release()