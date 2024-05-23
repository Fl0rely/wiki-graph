import random
def randow_walk(liste, n_chemin, profondeur):
    n=len(liste)
    file=open("random_walk.text","w")
    for i in range(n_chemin):
        liste=[]
        x=random.randint(0,n-1)
        for y in range(profondeur):
            liste[x]