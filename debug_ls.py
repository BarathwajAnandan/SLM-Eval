import os
import subprocess

def check_dirs():
    print(f"Current dir: {os.getcwd()}")
    print(f"Contents of tests/: {os.listdir('tests')}")
    if os.path.exists('tests/disco'):
        print(f"Contents of tests/disco: {os.listdir('tests/disco')}")
    else:
        print("tests/disco does not exist")

if __name__ == '__main__':
    check_dirs()
