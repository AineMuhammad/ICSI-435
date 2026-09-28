(While Workspace is open)
# Create the environment
python -m venv rl-env

# Activate it (Windows)
.\rl-env\Scripts\activate

# Activate it (Mac/Linux)
source rl-env/bin/activate

# Install Stable-Baselines3 (the algorithms) and Gymnasium (the standard API)
pip install stable-baselines3[extra] gymnasium

# Install the Atari emulator extension for Gymnasium
pip install ale-py autorom[accept-rom-license]

# Download game ROMs
AutoROM --accept-license

# Run the Test Program 
One everything is installed, your terminal lines should begin with "(rl-env)"
Test if everything works by running test.py, wait until the training stops,
then run test_play.py. You should see a visualization of the model trying to complete
a game, the results should be mixed, with the AI failing some runs, while getting a
perfect 100 on others.
