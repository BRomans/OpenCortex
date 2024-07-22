# Cortex Streamer
CortexStreamer is a fully-featured EEG streaming app to stream EEG data and markers.

It includes:
- a GUI to plot EEG in real-time
- signal filters (visualization only)
- signal quality estimators
- a button to save custom markers on the data
- an inlet stream that can receive event markers through LSL from an external source
- an outlet stream that can send raw EEG to an external source
- a general-purpose classifier class that can be initialized with any model from Scikit-Learn
- cross-validation plots with ROC curve and Confusion Matrix
- educational Jupyter notebooks to analyse EEG data using the library MNE and to perform classification tasks using Scikit-Learn.


## Table of Contents

- [Supported Devices](#supported-devices)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Issues](#issues)
- [Usage](#usage)
- [Examples](#examples)
- [Notebooks](#notebooks)
- [Building](#building)
    - [Wheel Distribution](#wheel-distribution)
    - [Source Distribution](#source-distribution)
- [Contributing](#contributing)
- [License](#license)

## Supported Devices
- Any EEG board listed on (Brainflow)[https://brainflow.readthedocs.io/en/stable/SupportedBoards.html]
- (Coming Soon) Emotiv Epoc and other consumer EEG devices

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

List any software or dependencies that need to be installed before setting up the project.

- Python 3.6 or higher
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

```bash
# Example: 
# Python 3
sudo apt-get install python3
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/BRomans/CortexToolkit.git
cd CortexToolkit
```
2. Create a virtual environment
```bash
# Using venv
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

```
3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Issues
If you encounter issues installing PyBluez, please refer to the latest comments on the project [issues page](https://github.com/pybluez/pybluez/issues/431).

## Usage
To run any example, use the following command:
```bash
cd examples
python <example_name>.py
```
To run the EEG Streamer app, use the following command:
```bash
python main.py
```

## Examples
The [examples](examples) folder contains single runnable scripts that demonstrate how to handle data collected
using g.tec hardware.

## Notebooks
The [notebooks](notebooks) folder contains some examples on how to use the utilities provided in this repository. You 
can run the notebooks using Jupyter, Jupyter Lab or Google Colab.

## Building
This project can be built as pip package using the following commands

### Wheel Distribution
To build a wheel distribution, run the following command:
```bash
python setup.py sdist bdist_wheel
```
Copy the content of the `dist` folder to the desired location and install the package using pip:
```bash
pip install <package_name>.whl
``` 

### Source Distribution
Alternatively, you if you want to build a source distribution, run the following command:
```bash
python setup.py sdist
```
Copy the content of the `dist` folder to the desired location and install the package using pip:
```bash
pip install <package_name>.tar.gz
```



## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome. 
1. Fork the project.
2. Create a new branch.
3. Make your changes and commit them.
4. Push to the branch and create a pull request.


## Credits
This project is freely available to anyone and is not intended for commercial use. If you use this project for academic 
purposes, please cite the original authors.

## License
This project is licensed under the CC License - see the [LICENSE](LICENSE) file for details.

## Authors
- [Michele Romani](https://bromans.github.io/)

Please make sure to update the [AUTHORS](AUTHORS) file if you are contributing to the project.


## Acknowledgments
- [Brainflow](https://brainflow.readthedocs.io/en/stable/index.html)
- [LabStreamingLayer](https://labstreaminglayer.org/)
- [MNE](https://mne.tools/stable/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/)

