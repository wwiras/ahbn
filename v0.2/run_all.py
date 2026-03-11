import os
import subprocess

CONFIG_DIR = "configs"

for file in sorted(os.listdir(CONFIG_DIR)):

    if file.endswith(".yaml"):

        path = os.path.join(CONFIG_DIR, file)

        print("Running:", path)

        subprocess.run(["python", "run_one.py", path])