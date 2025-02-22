import sys
from pathlib import Path

def set_root(level: int = 1) -> Path:
    """
    Set the root directory for the project.

    This function sets the root directory for the project by traversing up the directory
    tree a specified number of levels and appending it to the system path.

    Parameters
    ----------
    level : int, default=1
        The number of levels to traverse up the directory tree. If 0, the current working
        directory is used.

    Returns
    -------
    PROJECT_DIR : Path
        The root directory of the project.
    """
    if level != 0:
        for i in range(level):
            if i == 0:
                PROJECT_DIR = Path.cwd().parent
            else:
                PROJECT_DIR = PROJECT_DIR.parent
    else:
        PROJECT_DIR = Path.cwd()
    sys.path.append(str(PROJECT_DIR))
    return PROJECT_DIR
    

def create_dir(*args):
    """
    Create directories from a mix of Path objects and string paths.

    This function creates directories from the provided arguments. If a directory
    already exists, it does nothing for that directory.

    Parameters
    ----------
    *args : list of str or Path
        A list of directory paths to create. Each argument can be a string path or a
        Path object.

    Returns
    -------
    None
    """
    for arg in args:
        path = Path(arg)  # Ensure arg is a Path object
        path.mkdir(parents=True, exist_ok=True)