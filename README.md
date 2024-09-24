# IEEE 2024 Suicide Prevention Challenge

## Instructions
1) Install a plain vanilla python 3.8 or newer
2) open a command prompt in the root folder and create a virtual environment using
`python -m venv name_of_my_env` where you can replace name_of_my_env with any name you like
3) activate that env using `name_of_my_env\Scripts\activate` on windows, or `source name_of_my_env/bin/activate` on MacOS/Linux. Update your package manager
using `pip install --upgrade pip`
4) install all requirements using `pip install -r requirements.txt`
5) Place the evaluation data set in the root folder by replacing the "evaluation_data.xlsx" dummy file with the unseen evaluation data. Make sure it contains a column named `post`, just like the training data
6) Run the code using `python open_ai.py` which will create a file `Calculators.xlsx` which contains our predictions for the holdout data. 


## Authors
Stefan Pasch
Jannic Cutura
