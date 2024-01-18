import os

print('Entrenando modelo')

n_actions = [1, 2, 3, 4] # 20 30 40

for i in n_actions:
    os.system(f"python.exe CasaEntreno.py -o {str(i)} -mg 200 -t True -ng True")

print('Entrenamiento terminado')