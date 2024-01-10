import os

print('Entrenando modelo')

n_actions = [4, 6, 8, 10] # 20 30 40

for i in n_actions:
    os.system(f"python3 CasaEntreno.py -a {str(i)} -mg 200 -t True")

print('Entrenamiento terminado')