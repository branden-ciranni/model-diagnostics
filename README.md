# model-diagnostics
Model Diagnostics App for Logistic Regression.

## Get this running

1. Clone the repo
  ```zsh
  git clone https://github.com/branden-ciranni/model-diagnostics.git
  cd model-diagnostics
  ```

2. Create a virtual environment (recommended but not necessary)

    Windows:

      ```zsh
      python -m venv env

      env\scripts\activate.bat
      ```

    Mac:

      ```zsh
      python3 -m venv env

      source env/bin/activate
      ```
  
3. Install the requirements
  ``` zsh
  pip install -r app/requirements.txt
  ```
  
4. Run the app
  ```zsh
  cd app
  streamlit run app.py
  ```
