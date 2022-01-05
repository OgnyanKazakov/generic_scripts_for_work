import os
import sys


def add_to_path_if_not_exist(path):
    if path not in sys.path:
        sys.path.append(path)


def add_paths():
    current_relative = os.path.dirname(__file__)
    add_to_path_if_not_exist(current_relative)
    list_of_skipped_paths = [".git", ".vscode", "__pycache__", "MLCV", "Documentation", "Log", "log", "bin"]
    
    path_index = 0
    for path, subdirs, files in os.walk(current_relative):
        is_pattern_found = False
        for item in list_of_skipped_paths:
            if item in path:
                is_pattern_found = True
                break
        if is_pattern_found is False:
            path_index += 1
            add_to_path_if_not_exist(path)


if __name__ == "__main__":
    add_paths()
