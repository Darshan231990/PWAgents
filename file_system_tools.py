import os
import glob
from langchain_core.tools import tool
from typing import List

# --- Directory Tools ---

@tool
def search_listDirectory(directory: str = ".") -> str:
    """
    Lists all files and subdirectories in a specified directory.
    Returns a newline-separated list of items.
    """
    try:
        items = os.listdir(directory)
        if not items:
            return f"Directory '{directory}' is empty."
        return f"Contents of '{directory}':\n" + "\n".join(items)
    except FileNotFoundError:
        return f"Error: Directory not found at '{directory}'"
    except NotADirectoryError:
        return f"Error: The path '{directory}' is not a directory."
    except PermissionError:
        return f"Error: Permission denied to access '{directory}'."
    except Exception as e:
        return f"Error listing directory '{directory}': {e}"

@tool
def edit_createDirectory(directory_path: str) -> str:
    """
    Creates a new directory at the specified path.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return f"Successfully created directory (or it already existed): '{directory_path}'"
    except PermissionError:
        return f"Error: Permission denied to create directory at '{directory_path}'."
    except Exception as e:
        return f"Error creating directory '{directory_path}': {e}"

# --- File Read/Write Tools ---

@tool
def search_readFile(file_path: str) -> str:
    """
    Reads the entire content of a specified file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                return f"File '{file_path}' is empty."
            return content
    except FileNotFoundError:
        return f"Error: File not found at '{file_path}'"
    except IsADirectoryError:
        return f"Error: The path '{file_path}' is a directory, not a file."
    except PermissionError:
        return f"Error: Permission denied to read file at '{file_path}'."
    except UnicodeDecodeError:
        return f"Error: Could not decode file '{file_path}'. It may be binary."
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

@tool
def edit_createFile(file_path: str, content: str) -> str:
    """
    Creates and writes content to a new file. If the file exists, it is overwritten.
    """
    try:
        # Ensure the directory exists before writing the file
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{file_path}'"
    except IsADirectoryError:
        return f"Error: The path '{file_path}' is a directory. Cannot write to it."
    except PermissionError:
        return f"Error: Permission denied to write to file at '{file_path}'."
    except Exception as e:
        return f"Error writing to file '{file_path}': {e}"

# --- File Search Tools ---

@tool
def search_fileSearch(glob_pattern: str, base_dir: str = ".") -> str:
    """
    Searches for files matching a specific glob pattern (e.g., 'tests/*.py', '**/*.md')
    starting from a base directory. Returns a newline-separated list of matching files.
    """
    try:
        # Use os.path.join to safely combine base_dir and glob_pattern
        full_pattern = os.path.join(base_dir, glob_pattern)
        
        # Set recursive=True if the pattern contains '**' to search subdirectories
        is_recursive = '**' in glob_pattern
        
        results = glob.glob(full_pattern, recursive=is_recursive)
        
        if not results:
            return f"No files found matching pattern: '{glob_pattern}' in '{base_dir}'"
        
        return "Found matching files:\n" + "\n".join(results)
    except Exception as e:
        return f"Error during file search with pattern '{glob_pattern}': {e}"

@tool
def search_textSearch(search_query: str, file_path: str) -> str:
    """
    Searches for a specific text string (case-sensitive) within a single file.
    Returns the line number and the content of the first matching line.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if search_query in line:
                    return f"Found match in '{file_path}' at line {i + 1}:\n{line.strip()}"
        
        return f"No match found for '{search_query}' in '{file_path}'."
    except FileNotFoundError:
        return f"Error: File not found at '{file_path}'"
    except IsADirectoryError:
        return f"Error: The path '{file_path}' is a directory, not a file."
    except PermissionError:
        return f"Error: Permission denied to read file at '{file_path}'."
    except UnicodeDecodeError:
        return f"Error: Could not decode file '{file_path}'. It may be binary."
    except Exception as e:
        return f"Error searching in file '{file_path}': {e}"

def get_tools() -> List[tool]:
    """Returns a list of all file system tools."""
    return [
        search_listDirectory,
        edit_createDirectory,
        search_readFile,
        edit_createFile,
        search_fileSearch,
        search_textSearch
    ]

if __name__ == "__main__":
    # Example usage for testing the tools
    print("Testing File System Tools...")
    
    # Create a test directory
    print(edit_createDirectory("test_dir/subdir"))
    
    # Create a test file
    test_content = "Hello world!\nThis is a test file for search_textSearch.\nLine 3."
    print(edit_createFile("test_dir/subdir/test.txt", test_content))
    
    # List directory
    print(search_listDirectory("test_dir"))
    
    # Read file
    print(search_readFile("test_dir/subdir/test.txt"))
    
    # Text search (positive)
    print(search_textSearch("world", "test_dir/subdir/test.txt"))
    
    # Text search (negative)
    print(search_textSearch("goodbye", "test_dir/subdir/test.txt"))
    
    # File search (glob)
    print(search_fileSearch("**/*.txt", base_dir="test_dir"))
    
    # Test error: read directory
    print(search_readFile("test_dir"))
    
    # Test error: file not found
    print(search_readFile("non_existent_file.txt"))

