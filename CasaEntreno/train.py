import os

print('Entrenando modelo')

n_actions = [6, 8, 10]

for i in n_actions:
    os.system(f"python3 CasaEntreno.py -a {str(i)} -mg 100")

print('Entrenamiento terminado')