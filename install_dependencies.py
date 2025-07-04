import subprocess
import sys


def install_packages():
    """
    Reads requirements.txt and installs packages using pip.
    """
    try:
        print("--- Starting Dependency Installation ---")

        # Using sys.executable ensures we use the pip from the correct python environment
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

        print("\n--- All dependencies have been successfully installed! ---")
        print("You can now run the training and evaluation scripts.")

    except FileNotFoundError:
        print("\n[ERROR] 'requirements.txt' not found.")
        print(
            "Please make sure 'requirements.txt' is in the same directory as this script."
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] An error occurred during installation: {e}")
        print("Please check your internet connection and pip configuration.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    install_packages()
