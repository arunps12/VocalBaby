import subprocess
import sys

def install_packages(requirements_file='requirements.txt'):
    with open(requirements_file, 'r') as file:
        packages = file.readlines()

    for package in packages:
        package = package.strip()
        if package:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == "__main__":
    install_packages()