# Unicorn Python Essentials Toolkit
A collection of utilities for interfacing with g.tec Unicorn devices series, load and plot data.
Complete with examples to analyse EEG data using the library MNE and to perform classification tasks using scikit-learn.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

List any software or dependencies that need to be installed before setting up the project.

- Python 3.6 or higher

```bash
# Example: 
# Python 3
sudo apt-get install python3
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/BRomans/UnicornUtilities.git
cd UnicornUtilities
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

## Usage
To run any example, use the following command:
```bash
python examples/<example_name>.py
```

## Examples
The [examples](examples) folder contains single runnable scripts that demonstrate how to handle data collected
using g.tec hardware.

## Notebooks
The [notebooks](notebooks) folder contains some examples on how to use the utilities provided in this repository. You 
can run the notebooks using Jupyter, Jupyter Lab or Google Colab.


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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
- [Michele Romani](https://bromans.github.io/)

Please make sure to update the [AUTHORS](AUTHORS) file if you are contributing to the project.


## Acknowledgments
- [Brainflow](https://brainflow.readthedocs.io/en/stable/index.html)
- [g.tec](https://www.gtec.at/)
- [MNE](https://mne.tools/stable/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/)

