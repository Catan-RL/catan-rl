#!/bin/bash

# Function to activate the Conda environment
activate_env() {
    echo "Attempting to activate Conda environment 'catan-rl-39'..."
    if conda activate catan-rl-39; then
        echo "Conda environment 'catan-rl-39' activated successfully."
        return 0
    else
        echo "Failed to activate the Conda environment 'catan-rl-39'."
        read -p "Do you want to continue? DO NOT CONTINUE IF YOU HAVE NOT ALREADY ACTIVATED IT MANUALLY (yes/no): " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo "Continuing without activating the Conda environment."
                ;;
            *)
                echo "Exiting script."
                exit 1
                ;;
        esac
    fi
}

# Activate the environment
activate_env

# Set the MARLlib directory, defaulting to ../MARLlib if no input is provided
MARLLIB_DIR=${1:-../MARLlib}
echo "Using MARLlib directory: $MARLLIB_DIR"

cp ./requirements.txt "$MARLLIB_DIR/"

# Copy the necessary files to the MARLlib base environment
echo "Copying files to the MARLlib base environment..."
cp ./src/catan.py "$MARLLIB_DIR/marllib/envs/base_env/"
echo "Copied catan.py to $MARLLIB_DIR/marllib/envs/base_env/"
cp ./src/config/catanEnv.yaml "$MARLLIB_DIR/marllib/envs/base_env/config"
echo "Copied catanEnv.yaml to $MARLLIB_DIR/marllib/envs/base_env/config"
cp -r ./src/catan_env/ "$MARLLIB_DIR/marllib/envs/base_env/"
echo "Copied catan_env/ directory to $MARLLIB_DIR/marllib/envs/base_env/"

# Navigate to the MARLlib directory and install the package
echo "Navigating to MARLlib directory and installing the package..."
cd "$MARLLIB_DIR"
pip install .
echo "Package installed successfully in MARLlib."
