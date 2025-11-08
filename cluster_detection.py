import random
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import umap
import networkx as nx
import graph_function as gf
from kneed import KneeLocator
import time

import random
import numpy as np
import word2vec as w2

def random_walk_snd_ordre(graphe, nb_phrase, nb_iterations, p, q,nb_noeud):
    corpus = []
    sum=0
    for i in graphe.nodes():
        sum+=1
        gf.load(sum,nb_noeud,10)
        for _ in range(nb_phrase):
            phrase = aux_random_walk_snd_ordre(graphe, nb_iterations, i, p, q, -1)
            if phrase:  # Vérifiez que la phrase n'est pas vide
                corpus.append(phrase)
    return corpus

def aux_random_walk_snd_ordre(graphe, nb_iterations, pos, p, q, pre):
    phrase = []
    for _ in range(nb_iterations):
        voisins = list(graphe.adj[pos])  # Liste des voisins
        if voisins:
            # Calcul des probabilités pondérées
            proba = []
            for voisin in voisins:
                if voisin == pre:  # Retour vers le précédent nœud
                    proba.append(1 / p)
                elif pre in graphe.adj[voisin]:  # Connecté au précédent nœud
                    proba.append(1)
                else:  # Autre voisin
                    proba.append(1 / q)

            tot=sum(proba)
            alea=random.random() * tot
            tot=0
            for i in range(len(proba)):
                tot+=proba[i]
                if tot>=alea:
                    res=i
                    break

            titre_voisin = graphe.nodes[voisins[res]].get('id', 'Aucun id')
            # Ajout du titre à la phrase et mise à jour des positions
            phrase.append(titre_voisin)
            pre = pos
            pos = voisins[res]
        else:
            break  # Arrêtez si le nœud n'a pas de voisins
    return phrase

def random_walk(graphe, nb_phrase, nb_iterations, nb_noeud):
    corpus = []
    for i in graphe.nodes():
        gf.load(i,nb_noeud,10)
        for _ in range(nb_phrase):
            phrase = aux_random_walk(graphe, nb_iterations, i)
            if phrase:  # Vérifiez que la phrase n'est pas vide
                corpus.append(phrase)
    return corpus

def aux_random_walk(graphe, nb_iterations, pos):
    phrase = []
    for _ in range(nb_iterations):
        voisins = list(graphe.adj[pos])
        if voisins:
            voisin = random.choice(voisins)
            titre_voisin = graphe.nodes[voisin].get('titres', 'Aucun id')
            phrase.append(titre_voisin)
            pos = voisin  # Continuez la marche
        else:
            break  # Si le nœud n'a pas de voisins, arrêtez
    return phrase

def b_elbow_method(min_k, max_k, vectors, iter_max=100):
    
    distortions = []
    K=range(min_k, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, max_iter=iter_max, n_init='auto')
        kmeans.fit(vectors)

        # Inertie = somme des distances au carré aux centroïdes
        # On la normalise ici pour la rendre comparable
        distortion_moyenne = kmeans.inertia_ / vectors.shape[0]
        distortions.append(distortion_moyenne)

        # Libère les ressources associées au modèle pour économiser la RAM
        del kmeans
        gf.load(k-min_k,max_k-min_k,10)
    
    # Affichage du graphe de la méthode du coude
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-', markersize=8)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Distorsion moyenne')
    plt.title('Méthode du coude pour déterminer K')
    plt.savefig("res/Elbow_method_b.png")
    
    # Détermination du nombre optimal de clusters
    secur = K[np.argmin(np.gradient(distortions))]

    kneedle = KneeLocator(K, distortions, curve="convex", direction="decreasing")
    nb_commu = kneedle.knee if kneedle.knee else secur  # Sécurisation en cas d'échec
    print("nb_commu calculé: ",nb_commu)
    
    # Application de K-Means avec le nombre optimal de clusters
    kmeans = KMeans(n_clusters=nb_commu, max_iter=iter_max)
    kmeans.fit(vectors)
    
    # Création des groupes d'indices
    clusters_index = [[] for _ in range(nb_commu)]
    for i, label in enumerate(kmeans.labels_):
        clusters_index[label].append(i)
    
    return clusters_index, nb_commu

def elbow_method(min,max,matrice_embeddings,iter_max):
    list_distance=[]
    communaute=[]
    for k in range(min,max):
        clusters_index,distance=k_means(matrice_embeddings,k,iter_max)
        list_distance.append(distance)
        communaute.append(clusters_index)
        gf.load(k-min,max-min,10)

    plt.plot(range(min,max),list_distance,'bx-', markersize=8)
    plt.show()
    nb_commu=int(input("Où est le coude ?"))
    plt.savefig("res/Elbow_method.png")
    return communaute[nb_commu-min],nb_commu

def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

def k_means(W,k,iters):
   
    
    # Initialisation des centroïdes
    centroids = W[np.random.choice(W.shape[0], k, replace=False)]

    for _ in range(iters):
        #création des clusters
        distance=0
        clusters=[]
        clusters_index=[]
        for i in range(k):
            clusters.append([])
            clusters_index.append([])
        
        #parcourt de tous les embeddings
        for index,h in enumerate(W):
            distances=[0]*k
            for i in range(k):
                distances[i]=euclidean_distance(h,centroids[i])
            #détermination du centroid le plus proche
            plus_proche=np.argmin(distances)
            distance+=np.min(distances)
            #ajout du vecteur à la commuanuté la plus proche
            clusters[plus_proche].append(h)
            clusters_index[plus_proche].append(index)
        # Mise à jour des centroïdes
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroid for centroid, cluster in zip(centroids, clusters)])

        # Vérification de la convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return clusters_index,distance


def pca(X, n_components=2, verbose=True):
    # Centrage des données
    X_centered = X - np.mean(X, axis=0)

    # Matrice de covariance
    cov_matrix = np.dot(X_centered.T, X_centered)

    # Décomposition en valeurs et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Tri décroissant des valeurs propres
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]

    # Calcul du pourcentage d'information conservée
    variance_total = np.sum(eigenvalues)
    variance_conservee = np.sum(eigenvalues[:n_components])
    pourcentage_conservee = (variance_conservee / variance_total) * 100

    if verbose:
        print(f"Variance expliquée par les {n_components} premières composantes : {pourcentage_conservee:.2f}%")
        print(f"Information perdue : {100 - pourcentage_conservee:.2f}%")

    # Réduction
    components = eigenvectors[:, :n_components]
    return np.dot(X_centered, components)



def graphe_communaute(graphe, W1, clusters, reduced_embeddings):
    error=0
    for cluster_id, cluster in enumerate(clusters):
        for id in cluster:
            # Trouver l'index de l'embedding dans W1
            graphe.nodes[id]["Community"] = cluster_id
            graphe.nodes[id]["X"] = reduced_embeddings[id, 0]  # Coordonnées X fictives
            graphe.nodes[id]["Y"] = reduced_embeddings[id, 1]  # Coordonnées Y fictives
            graphe.nodes[id]["degré entrant"]=graphe.in_degree(id)
    for node_id in list(graphe.nodes()):
        try:
            if graphe.nodes[node_id]["Community"] == -1:
                graphe.remove_node(node_id)
        except:
            error+=1 
    nx.write_gexf(graphe, "res/graph_wikipedia_commu.gexf")
   

        
from sklearn.metrics.pairwise import cosine_similarity
def mots_proches(mot_cible, embeddings_matrix, word_to_index, top_n=3):
    """
    Affiche les mots les plus proches d’un mot donné selon les vecteurs d'embedding.
    
    Args:
        mot_cible (str): mot à rechercher
        embeddings_matrix (np.ndarray): matrice des vecteurs (shape: nb_mots x dim)
        word_to_index (dict): dictionnaire {mot: index}
        top_n (int): nombre de voisins à afficher
    """
    if mot_cible not in word_to_index:
        print(f" Mot introuvable : '{mot_cible}'")
        print(" Exemples de mots valides :", list(word_to_index.keys())[:10])
        return

    idx = word_to_index[mot_cible]
    vecteur = embeddings_matrix[idx].reshape(1, -1)

    # Calcul des similarités cosinus avec tous les autres vecteurs
    similarites = cosine_similarity(vecteur, embeddings_matrix)[0]

    # Indices des mots les plus similaires (on ignore le mot lui-même)
    indices_tries = np.argsort(similarites)[::-1]
    voisins_proches = [i for i in indices_tries if i != idx][:top_n]

    # Inverser le dictionnaire : index → mot
    index_to_word = {i: w for w, i in word_to_index.items()}

    print(f"\n Mots les plus proches de '{mot_cible}' :")
    for i in voisins_proches:
        mot_voisin = index_to_word[i]
        score = similarites[i]
        print(f"  - {mot_voisin} (score: {score:.4f})")

def main(graphe,nombre_phrase,nombre_mot,home_made, load):
    #nombre de phrase: nombre de phrase dans le corpus 
    #nombre de mot par phrase dans le corpus
    p=1 #facteur pour favoriser les voisins hors de la commuanuté
    q=2 #facteur pour favoriser les voisins dans la communauté
    marche_aleatoire=1 #1: 1er ordre, #2: second ordre


    if load==False:
        #paramètre Word2vec
        min_count=5
        embedding_dim = 300  # Dimension des embeddings
        window_size = 5
        learning_rate = 0.025
        epochs = 30

        nombre_noeud=len(graphe.nodes())
        print("----------début marche aléatoire------------")

        #marche aléatoire
        start_time_gexf = time.time()  # Enregistre le temps de début
        
        if marche_aleatoire==1:
            corpus=random_walk(graphe,nombre_phrase,nombre_mot,nombre_noeud)
        else:
            corpus=random_walk_snd_ordre(graphe,nombre_phrase,nombre_mot,p,q,nombre_noeud)
        
        end_time_gexf = time.time()  # Enregistre le temps de fin
        elapsed_time = end_time_gexf - start_time_gexf  # Calcule la durée d'exécution
        print(f"Temps d'exécution de la marche aléatoire: {elapsed_time} secondes")
        print("----------fin marche aléatoire------------")


        print("----------début algo word2vec------------")
        #entrainement du réseau de neuronnes

        start_time_gexf = time.time()  # Enregistre le temps de début
        if home_made==True:
            id2word,matrice_embeddings,word2id=w2.f_word2vec_self_made(graphe,corpus,window_size,min_count,embedding_dim,learning_rate,epochs)
            words_vectors={}
            for i in range(len(id2word)):
                words_vectors[id2word[i]]=w2.embedding_mot(len(id2word),i,matrice_embeddings)

        else:
            # Entraînement du modèle Word2Vec
            model = Word2Vec(sentences=corpus, vector_size=embedding_dim, window=window_size, min_count=min_count, sg=1)
            
            # Liste des mots appris par le modèle
            words = list(model.wv.key_to_index.keys())
            
            # Matrice des embeddings
            matrice_embeddings = np.array([model.wv[word] for word in words])
            
            # Dictionnaire des vecteurs de mots
            words_vectors = {word: model.wv[word] for word in words}
        
        end_time_gexf = time.time()  # Enregistre le temps de fin
        elapsed_time = end_time_gexf - start_time_gexf  # Calcule la durée d'exécution
        print(f"Temps d'exécution de Word2Vec: {elapsed_time} secondes")
        #enregistrement dans un fichier
        with open('save\Embeddings_dict.npy', 'wb') as save_dic:
                np.save(save_dic, words_vectors)
        with open('save\Embeddings_matrix.npy', 'wb') as save_embeddings:
                np.save(save_embeddings, matrice_embeddings)
    else:
        print("Chargement des résultats précédents ...")
        words_vectors= np.load('save\Embeddings_dict.npy', allow_pickle="TRUE").item()
        matrice_embeddings= np.load('save\Embeddings_matrix.npy', allow_pickle="TRUE")
        word_to_index = {mot: i for i, mot in enumerate(words_vectors.keys())}
        index_to_word = {i: mot for mot, i in word_to_index.items()}

        # On garde les vecteurs dans le bon ordre (liste de vecteurs)   
        embedding_matrix = np.array([words_vectors[mot] for mot in word_to_index])
    
    mot="history"   
    print("détermination voisin de ", mot)
    mots_proches(mot,embedding_matrix,word_to_index, 5)
    print("----------début clusterisation-----------")   
    #algorithme de clusterisation
    iter_max=100
    min_k=5
    max_k=40

    start_time_gexf = time.time()  # Enregistre le temps de début
    
    if home_made==True:
        clusters_index,nb_commu=elbow_method(min_k,max_k,matrice_embeddings,iter_max)
    else:
        clusters_index,nb_commu=b_elbow_method(min_k, max_k, matrice_embeddings, iter_max)
    
    print("division du graphe en ", nb_commu, " commuanutés")

    end_time_gexf = time.time()  # Enregistre le temps de fin
    elapsed_time = end_time_gexf - start_time_gexf  # Calcule la durée d'exécution
    print(f"Temps d'exécution de la création du fichier gexf: {elapsed_time} secondes")
    print("----------fin clusterisation-----------")   


    print("----------début réduction en dimension 2-----------")   
    # Réduction de dimensionnalité pour la visualisation
    reduced_embeddings = pca(matrice_embeddings, n_components=2)
    print("----------fin réduction en dimension 2-----------") 


    #visualisation des résultats
    print("----------création graphe de communauté-----------") 
    #plot_interactive_graph(word_vectors, words, clusters)
    graphe_communaute(graphe,reduced_embeddings,clusters_index,reduced_embeddings)

    print("le fichier graph_wikipedia_commu.gexf à été crée")