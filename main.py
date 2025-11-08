import graph_function as gf
import cluster_detection as cd


#open simple english 
titre="simple_wiki"
lim_page=-1#nombre de page visitée (-1 pour ne pas avoir de limite)
creation_gexf= False#défini la création d'un fichier gexf pour le graph (fichier visuel)
home_made=False
load=True #permet de charger les fichiers précédent (matrice embeddings + dict et save)

nombre_phrases=80 #paramètre phrase marche aléatoire
nombre_mot=60

res="save/save_{0}_{1}.txt".format(titre,lim_page)
link="doc\\dump_{0}.xml".format(titre)
if lim_page == -1:
    print("Analyse de toutes les pages de ", titre)
else:
    print( "analyse de ",lim_page, "pages de ",titre)
print("--------début détection lien------------------------")
nb_noeud=gf.parse_lxml(link,res,lim_page) 
print(nb_noeud, " détectés")
print("--------fin détection lien------------------------")
print("--------début création graphe------------------------")
graphe=gf.make_graph(res,nb_noeud,creation_gexf)
print("--------fin création graphe------------------------")


cd.main(graphe,nombre_phrases,nombre_mot,home_made, load)
    