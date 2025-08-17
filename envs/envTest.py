# check_versions.py
import importlib

packages = [
    "transformers",
    "torch",
    "torchvision",
    "matplotlib",
    "PIL",
    "numpy",
    "einops",
    "pandas",
    "tqdm",
    "wandb",
    "torchview",
    "piq",
    "kornia",
    "cv2",  
    "IPython",
]

def get_version(pkg_name):
    try:
        pkg = importlib.import_module(pkg_name)
        if hasattr(pkg, '__version__'):
            return pkg.__version__
        elif hasattr(pkg, 'VERSION'):
            return pkg.VERSION
        else:
            return "Version attribute not found"
    except ModuleNotFoundError:
        return "Not installed"

if __name__ == "__main__":
    print("Package Versions:")
    for pkg_name in packages:
        print(f"{pkg_name}: {get_version(pkg_name)}")
