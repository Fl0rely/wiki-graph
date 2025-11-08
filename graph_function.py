import networkx as nx
from lxml import etree
import time
from igraph import *
import os

def load(pos, max, rate):
    if max > 0:
        percentage = (pos * 100) / max
        if percentage % rate < 100/max:
            print("{0}%".format(round(percentage)))


def trouver_lien(text, page_dict, page_title):
    #text: text de la page
    #page_dict: dictionnaire clé:titre élément: liste des titres
    #page title: nom de la page 
    place = 0
    liste = []
    #print(text)
    while True:
        x = text.find("[[", place) # rcherche balise ouvrante lien 
        if x == -1:
            break
        y = text.find("]]", x) # rcherche balise fermante lien 
        if y == -1:
            break
        z = text.find("|", x + 2, y) #cas ou il y a un allias (changement de nom)
        #print(text[x+2:y])
        if z != -1:
            mot = text[x+2:z]
        else:
            mot = text[x+2:y]
        mot=mot.lower()
        if mot.endswith("\n"):
            # Supprime les deux derniers caractères
            mot= mot[:-2]
        if mot.startswith("\n"):
            mot=mot[2:]
        liste.append(mot)
        place = y + 2  # mise à jour de `place` pour commencer la recherche après ce lien
    page_dict[page_title] = liste

    #print(page_title,liste)
    
def parse_lxml(fichier,link_res,lim):
    #link_res: lien pour le fichier avec les résultats dans le format: nom_de_la_page$lien1#lien2...
    #lim: nombre de lien avant de s'aretter (-1: pas de limite)

    #écrit le titre et chacun des liens de chaque page dans le fichier res dans le format titre$lien1#lien2#lien3
    #return le nombre de page

    if not os.path.exists(link_res):
        res=open(link_res,"w",encoding="utf-8")
        count=0
        page_dict={}
        for event,elem in etree.iterparse(fichier,events=('start','end','start-ns','end-ns')):
            if lim>=0: #limite du nombre de lien si supérieur à 0 
                if count>lim:
                    break
            if event=="start":
                if elem.tag[-4::]=="page": #slection des pages
                    model=""
                    text=""
                    title=""
                    id=""
                    verif=0
                    valid=True
                elif elem.tag[-5::]=='title': #slection du titre
                    titre=elem.text
                    if titre==None:
                        valid=False
                    #print(titre)
                elif elem.tag[-2::]=='id': #slection du titre
                    id=elem.text
                elif elem.tag[-5::]=="model":
                    model=elem.text
                    if model!="wikitext":
                        valid=False
                    #print(model)
                elif elem.tag[-2::]=="ns" and elem.text!="0" :
                    #print("-----------------------------------------------------")
                    #print(elem.text)
                    valid=False
                elif elem.tag[-4::]=="text" and elem.text!=None and valid==True:
                    #print(elem.text)
                    trouver_lien(elem.text,page_dict,titre) #fonction pour ajouter la liste de tout les liens d'une page dans le dictionnaire
                    verif=+1
                
            elif event=="end":
                if elem.tag[-4::]=="page" and verif==1 and id!=None: #fin d'une page
                    #affichage de l'évolution dans le terminal
                    load(count,lim,10)
                    #écriture du titre et lin dans le format id@titre$lien1#lien2
                    res.write(id)
                    res.write("§")
                    res.write(titre.lower())
                    res.write("$")
                    for i in page_dict[titre]:
                        res.write(i.lower())
                        res.write("#")
                    res.write("\n")
                    page_dict.clear()
                    #print(titre,page_dict[titre])
                    elem.clear()
                    count+=1
        return count
    return lim

def make_graph(file,lim,creation_gexf):
    res=open(file,'r',encoding='utf-8')
    dico={} #dictionnaire: clée:titre élément: numéro de noeud 
    graphe=nx.DiGraph() #création graphe
    lignes=res.readlines() #lecture ligne par ligne du fichier res
    num_ligne=0
    for ligne in lignes: 
        try:
            ligne=ligne.split("§")
            id=ligne[0]
            ligne=ligne[1]
            ligne=ligne.split("$")  
            titre=ligne[0].lower()
            graphe.add_node(num_ligne,titres=titre, Community = -1,id=id)# création de tous les noeuds
            dico[titre]=num_ligne #ajout du titre et du numéro du noeud dans le dico
            #print(graphe.nodes())
            num_ligne+=1
        except:
            ()
    num_ligne=-1
    count_error_connexion=0
    count_error_liste=0
    count_lien_cree=0
    for ligne in lignes:
        num_ligne+=1
        try:
            ligne=ligne.split("§")
            id=ligne[0]
            ligne=ligne[1]
            ligne=ligne.split("$")
            titre=ligne[0]  
            liste=ligne[1].split("\n")[0]
            liste=liste.split("#") #liste de tous les liens d'une page
            load(num_ligne,lim,10)
            if lim>=0:
                if num_ligne>=lim:
                    break
            for elem in liste:
                try :
                    graphe.add_edge(num_ligne, dico[elem.lower()]) #ajout d'un lien entre les pages
                    count_lien_cree+=1
                except:
                    count_error_connexion+=1
        except:
            count_error_liste+=1
    if creation_gexf:
        start_time_gexf = time.time()  # Enregistre le temps de début
        nx.write_gexf(graphe, "res/graphe_wikipedia.gexf")
        end_time_gexf = time.time()  # Enregistre le temps de fin
        elapsed_time = end_time_gexf - start_time_gexf  # Calcule la durée d'exécution
        print(f"Temps d'exécution de la création du fichier gexf: {elapsed_time} secondes")
    
    return graphe
