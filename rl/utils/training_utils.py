import os
import shutil


def find_latest_checkpoint(directory):
    checkpoint_dirs = [
        name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))
                                                  and name.startswith('checkpoint_')
    ]

    if not checkpoint_dirs:
        return None

    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))
    return latest_checkpoint_dir


def delete_checkpoint_folders(directory):
    # Iterate over the subdirectories in the given directory
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            # Check if the subfolder name starts with "checkpoint_"
            if name.startswith("checkpoint_") or name.startswith("experiment_"):
                folder_path = os.path.join(root, name)
                # Delete the subfolder
                print(f"Deleting folder: {folder_path}")
                try:
                    shutil.rmtree(folder_path)
                    print("Folder deleted successfully.")
                except OSError as e:
                    print(f"Error occurred while deleting folder: {str(e)}")
