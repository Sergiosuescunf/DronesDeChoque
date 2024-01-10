import os

print('Entrenando modelo')

os.system(f"python.exe CasaEntreno.py -o 1 -g 170 -mg 200 -t True")
os.system(f"python.exe CasaEntreno.py -o 2 -g 99 -mg 200 -t True")
os.system(f"python.exe CasaEntreno.py -o 3 -g 89 -mg 200 -t True")

print('Entrenamiento terminado')