"""
Script to load and print the contents of a .npy file containing processed MIDI data.
"""
import argparse
import numpy as np


def main():
    """
    Main function to parse arguments and print the contents of a .npy file.
    """
    parser = argparse.ArgumentParser(
        description="Print the contents of a .npy file containing processed MIDI data."
    )
    parser.add_argument("npy_file", type=str, help="Path to the .npy file")
    args = parser.parse_args()

    try:
        # Load the numpy array. Using allow_pickle=True in case the array contains objects.
        data = np.load(args.npy_file, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    except ValueError as e:
        print(f"Error loading file: {e}")
        return

    print("Contents of the .npy file:")
    print(data)

if __name__ == "__main__":
    main()
