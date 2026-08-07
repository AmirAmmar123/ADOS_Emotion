from moviepy import VideoFileClip
import os

def convert_avi_to_gif(input_path, output_path, start_sec, end_sec, fps=10, resize_factor=0.5):
    """
    Converts a specific segment of an .avi video to a .gif file.
    
    Parameters:
    - input_path (str): Path to the source .avi file.
    - output_path (str): Path to save the resulting .gif.
    - start_sec (float/int): Start time of the clip in seconds.
    - end_sec (float/int): End time of the clip in seconds.
    - fps (int): Frames per second for the GIF. Lower is better for smaller file sizes.
    - resize_factor (float): Scales the resolution (e.g., 0.5 cuts width & height in half).
    """
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} does not exist.")
        return

    try:
        print(f"Loading video: {input_path}")
        # Load the original video
        clip = VideoFileClip(input_path)
        
        # --- FIX 1: Use .subclipped() instead of .subclip() ---
        subclip = clip.subclipped(start_sec, end_sec)
        
        # --- FIX 2: Use .resized() instead of .resize() ---
        if resize_factor and resize_factor != 1.0:
            subclip = subclip.resized(resize_factor)
            
        print(f"Extracting from {start_sec}s to {end_sec}s... Generating GIF...")
        
        # Write the GIF to disk
        subclip.write_gif(output_path, fps=fps)
        
        # Free up system memory
        subclip.close()
        clip.close()
        print(f"\nSuccessfully created GIF: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    
    # Define your parameters here
    INPUT_VIDEO = "/home/amir/Ados_lab/videos2/openface_marisa/143/143_smile_rendered.avi"
    OUTPUT_GIF = "reaction_smile.gif"
    
    # Time in seconds (can be floats, e.g., 10.5 to 12.3)
    START_TIME = 57  
    END_TIME = 63  
    
    convert_avi_to_gif(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_GIF,
        start_sec=START_TIME,
        end_sec=END_TIME,
        fps=12,              # 12-15 fps is usually the sweet spot for smooth GIFs
        resize_factor=0.5    # Shrinks a 1080p video to 540p to save space
    )