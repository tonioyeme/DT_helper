import streamlit.web.cli as stcli
import os
import sys

if __name__ == "__main__":
    # Get the absolute path of the app
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(dir_path, "app", "main.py")
    
    # Run the app
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501", "--server.headless=false"]
    sys.exit(stcli.main()) 