from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# Initialize an empty list to store all the clips
clips = []

# Loop through the numbers 1 to 10
for i in range(1, 11):
    # File names
    img_file = f'./images/picture{i}.png'
    audio_file = f'./audio/picture{i}.mp3'

    # Load the image and audio
    img_clip = ImageClip(img_file)
    audio_clip = AudioFileClip(audio_file)

    # Set the duration of the image clip to the duration of the audio
    img_clip = img_clip.set_duration(audio_clip.duration)

    # Set the audio of the image clip
    img_clip = img_clip.set_audio(audio_clip)

    # Add the clip to our list
    clips.append(img_clip)

# Concatenate all the clips into one video
final_clip = concatenate_videoclips(clips)

# Write the final video to a file
final_clip.write_videofile('final_video.mp4', codec='libx264', fps=24)