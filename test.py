from rich.tree import Tree
from rich import print
import os

def make_tree(path, tree=None, skip_dirs=None):
    if skip_dirs is None:
        skip_dirs = ["__pycache__",".venv",".git",".idea"]  # add any other directories you want to skip

    if tree is None:
        tree = Tree(path)

    for f in os.listdir(path):
        if f in skip_dirs:
            continue  # skip unwanted directories

        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            subtree = tree.add(f)
            make_tree(full_path, subtree, skip_dirs)
        else:
            tree.add(f)
    return tree

# Example usage
print(make_tree(r"C:\Users\Dash\PycharmProjects\NeuralNetworks"))