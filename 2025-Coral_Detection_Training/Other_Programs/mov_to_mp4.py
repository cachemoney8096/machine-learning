import subprocess
import sys
import os

def convert_mov_to_mp4(input_file, output_file=None):
    """
    Convert a MOV file to MP4 format using FFmpeg.
    
    Args:
        input_file: Path to the input MOV file
        output_file: Path to the output MP4 file (optional)
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.mp4'
    
    # FFmpeg command
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',      # Audio codec
        '-strict', 'experimental',
        '-b:a', '192k',     # Audio bitrate
        '-y',               # Overwrite output file if exists
        output_file
    ]
    
    try:
        print(f"Converting {input_file} to {output_file}...")
        subprocess.run(command, check=True)
        print(f"Conversion successful! Output saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg first.")
        print("Install with: brew install ffmpeg (Mac) or apt-get install ffmpeg (Linux)")
        return False

if __name__ == "__main__":
    
    input_file = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/people_walking.mov"
    output_file = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/people_walking.mp4"
    
    convert_mov_to_mp4(input_file, output_file)