import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree, objectify
import time
from igraph import *


def trouver_lien(text, page_dict, page_title):
    place = 0
    liste = []
    while True:
        x = text.find("[[", place)
        if x == -1:
            break
        y = text.find("]]", x)
        if y == -1:
            break
        z = text.find("|", x + 2, y)
        if z != -1:
            mot = text[x+2:z]
        else:
            mot = text[x+2:y]
        liste.append(mot)
        place = y + 2  # mise à jour de `place` pour commencer la recherche après ce lien
    page_dict[page_title] = liste

    #print(page_title,liste)
    


def parse_lxml(fichier,link_res,lim):
    res=open(link_res,"w",encoding="utf-8")
    count=0
    page_dict={}
    for event,elem in etree.iterparse(fichier,events=('start','end','start-ns','end-ns')):
        #print(event,elem)

        if lim>0:
            if count>lim:
                break
        if event=="start":
            if elem.tag[-4::]=="page":
                model=""
                text=""
                title=""
                verif=False
                valid=True
            elif elem.tag[-5::]=='title':
                titre=elem.text
                if titre==None:
                    valid=False
                #print(titre)
            elif elem.tag[-5::]=="model":
                model=elem.text
                if model!="wikitext":
                    valid=False
                #print(model)
            elif elem.tag[-4::]=="text" and elem.text!=None and valid==True:
                #print(elem.text)
                trouver_lien(elem.text,page_dict,titre)
                verif=1
            
        elif event=="end":
            
            if elem.tag[-4::]=="page" and verif==1:
                if count%10000==0:
                    print(count)
                res.write(titre)
                res.write("$")
                for i in page_dict[titre]:
                    res.write(i)
                    res.write("#")
                res.write("\n")
                page_dict.clear()
                #print(titre,page_dict[titre])
                elem.clear()
                count+=1
    return count

def make_graph(file,lim):
    res=open(file,'r',encoding='utf-8')
    dico={}
    graphe=nx.Graph()
    lignes=res.readlines()
    num_ligne=0
    for ligne in lignes:
        ligne=ligne.split("$")
        titre=ligne[0]
        graphe.add_node(num_ligne,titres=titre)
        dico[titre]=num_ligne
        #print(graphe.nodes())
        num_ligne+=1
    num_ligne=0
    print("fin création noeud")
    count_error=0
    for ligne in lignes:
        ligne=ligne.split("$")
        titre=ligne[0]
        try:
            liste=ligne[1].split("\n")[0]
            liste=liste.split("#")
            if num_ligne%100000==0:
                print(num_ligne)
            if lim>0:
                if num_ligne>lim:
                    break
            for elem in liste:
                try :
                    graphe.add_edge(num_ligne, dico[elem])
                except:
                    count_error+=1
                    #print("erreur nom")
            num_ligne+=1
            
        except:
            count_error+=1
    print(num_ligne)
    print("fin création lien")
    # Dessiner le graphe
   # pos = nx.spring_layout(graphe)  # positions for all nodes

    # nodes
    #nx.draw_networkx_nodes(graphe, pos, node_size=200)

    # edges
    #nx.draw_networkx_edges(graphe, pos, width=1)

    # labels
    #node_labels = nx.get_node_attributes(graphe, 'titres')
    #nx.draw_networkx_labels(graphe, pos, labels=node_labels)

    #plt.title("Graphe avec Titres de Nœuds")
    #plt.axis('off')  # Cacher les axes pour une meilleure clarté
    #plt.show()
    print("début conversion")
    start_time = time.time()  # Enregistre le temps de début
    
    nx.write_gexf(graphe, "graphe_wikipedia_10000.gexf")

    end_time = time.time()  # Enregistre le temps de fin
    elapsed_time = end_time - start_time  # Calcule la durée d'exécution
    print(f"Temps d'exécution: {elapsed_time} secondes")
            
#open simple english 
link="C:\\Users\\virel\\Desktop\\prépa\\TIPE\\simplewiki-20240401-pages-meta-current.xml\\simplewiki-20240401-pages-meta-current.xml"
#dump=open("C:\\Users\\virel\\Desktop\\prépa\\TIPE\\tiwiki-20240320-pages-articles-multistream.xml\\tiwiki-20240320-pages-articles-multistream.xml","r",encoding="utf-8")

res="C:\\Users\\virel\\Desktop\\prépa\\TIPE\\res.txt"
res2="C:\\Users\\virel\\Desktop\\prépa\\TIPE\\res2.txt"
res3="res4.txt"

lim=10000
count=parse_lxml(link,res3,lim)
print(count)
make_graph(res3,lim)


#4,22 min pour 10000 noeuds
