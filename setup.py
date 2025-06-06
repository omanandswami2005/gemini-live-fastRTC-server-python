from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gemini_live_voice_only",
    version="0.1.1",
    description="A real-time Gemini API streaming library with FastAPI integration",
    long_description=long_description,
    long_description_content_type="text/markdown",  # change if using rst
    author="Omanand Swami",
    author_email="omanandswami2005@gmail.com",
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*,"]
    ),
    requires=["python (>=3.12)"],
    install_requires=[
        "fastapi",
        "uvicorn",
        "numpy",
        "python-dotenv",
        "fastrtc",         # Ensure this dependency is available or provide installation instructions.
        "google-genai",     # Replace with the actual package name if different.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)