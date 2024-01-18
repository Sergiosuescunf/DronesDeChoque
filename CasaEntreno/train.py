import os

os_name = os.name

if os_name == 'posix':
    PYTHON_COMMAND = 'python3'
elif os_name == 'nt':
    PYTHON_COMMAND = 'python.exe'

print('Entrenando modelo')

n_actions = [1, 2, 3, 4] # 20 30 40

for i in n_actions:
    os.system(f"{PYTHON_COMMAND} CasaEntreno.py -o {str(i)} -mg 200 -t True")

print('Entrenamiento terminado')