<p align="center"><img src="https://github.com/nathanbronson/BotOrNot/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____

# BotOrNot
explainable AI authorship detection

## About
BotOrNot explainably detects ChatGPT authorship. BotOrNot takes user input and gives a report classifying the text as ChatGPT- or human-written and highlighting the parts of the text most important to its decision. BotOrNot uses a random forest model trained on thousands of writing samples. On unfamiliar samples from this dataset, BotOrNot achieved >95% accuracy.

Data and data generation files are stored elsewhere.

This codebase has not been actively maintained since 2023

## Usage
BotOrNot can be run within `standalone_version/` by running the command
```
$ python3 main_module.py
```
Custom files can be specified at line 57 of `standalone_version/main_module.py`

BotOrNot can be compiled with a user interface for any platform using Briefcase from The BeeWare Project. BotOrNot has been tested for macOS. To launch the Toga interface without building the project, navigate to `app/BotOrNot/BotOrNot/src` and run
```
$ python3 -m BotOrNot/app.py
```

## License
See `LICENSE`.