# LHGNN (Local-Higher Order Graph Neural Networks)

## Overview
LHGNN is a Python-based implementation of Local-Higher Order Graph Neural Networks designed to enhance the performance of graph-based learning tasks. This project aims to provide a robust framework for researchers and practitioners working with graph data, enabling them to leverage higher-order relationships effectively.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Structure

The project is organized as follows:

```
LHGNN/
├── .github                   # GitHub Actions workflows
│
├── configs                   # Hydra configs
│   ├── callbacks             # Callbacks configs
│   ├── data                  # Data configs
│   ├── debug                 # Debugging configs
│   ├── experiment            # Experiment configs
│   ├── extras                # Extra utilities configs
│   ├── hparams_search        # Hyperparameter search configs
│   ├── hydra                 # Hydra configs
│   ├── local                 # Local configs
│   ├── logger                # Logger configs
│   ├── model                 # Model configs
│   ├── paths                 # Project paths configs
│   └── trainer               # Trainer configs
│       ├── eval.yaml        # Main config for evaluation
│       └── train.yaml       # Main config for training
│
├── data                      # Project data
│
├── logs                      # Logs generated by Hydra and Lightning loggers
│
├── notebooks                 # Jupyter notebooks
│                             # Naming convention: number (for ordering), creator's initials,
│                             # and a short `-` delimited description,
│                             # e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                   # Shell scripts
│
├── src                       # Source code
│   ├── data                  # Data scripts
│   ├── models                # Model scripts
│   ├── utils                 # Utility scripts
│   ├── eval.py               # Run evaluation
│   └── train.py              # Run training
│
├── tests                     # Tests of any kind
│
├── .env.example              # Example of file for storing private environment variables
├── .gitignore                # List of files ignored by git
├── .pre-commit-config.yaml   # Configuration of pre-commit hooks for code formatting
├── .project-root             # File for inferring the position of project root directory
├── environment.yaml          # File for installing conda environment
├── Makefile                  # Makefile with commands like `make train` or `make test`
├── pyproject.toml            # Configuration options for testing and linting
├── requirements.txt          # File for installing Python dependencies
├── setup.py                  # File for installing project as a package
└── README.md                 # Project documentation
```

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shubhrsingh22/LHGNN.git
   cd LHGNN
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the model, use the following command:
```bash
python src/train.py --config configs/train.yaml
```
You can modify the configuration file to adjust parameters such as learning rate, number of epochs, and dataset paths.

## Configuration
The project uses YAML files for configuration. You can find the default configuration in `configs/train.yaml`. Modify this file to set parameters for your experiments.

## Testing
To run the tests, use the following command:
```bash
pytest tests/
```
Make sure to have `pytest` installed in your environment.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your message"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for graph neural network functionalities.
- [NetworkX](https://networkx.org/) for graph manipulation and analysis.
- [Any other libraries or contributors you want to acknowledge]
