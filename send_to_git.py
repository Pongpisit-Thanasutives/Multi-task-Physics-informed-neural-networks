import os
from os.path import join
import shutil
import subprocess

comand = "bash clean.sh"
process = subprocess.Popen(comand.split(), stdout=subprocess.PIPE)
process.communicate()
print("Done some cleanings")

unwanted_files = ['.DS_store', '.virtual_documents', '.ipynb_checkpoints', '__pycache__']
print("List of tmp files:", unwanted_files)

command = input()
if command.lower() == 'print': func = print
elif command.lower() == 'del': func = shutil.rmtree

for root, dirs, files in os.walk('./'):    
    for f in unwanted_files:
        if f in dirs:
            print(join(root, f))
            func(join(root, f))

inp = input("Want to push ?")
if inp.lower() == "yes":
    comand= "bash push_to_git.sh"
    process = subprocess.Popen(comand.split(), stdout=subprocess.PIPE)
    process.communicate()
else: print("Not pushing")
