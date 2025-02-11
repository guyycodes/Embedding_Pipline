/opt/homebrew/bin/python3.11 -m venv quantum_env_311
rm -rf quantum_env_311
source quantum_env_311/bin/activate
pip install -r requirements.txt
pip install --upgrade pip