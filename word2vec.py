import numpy as np

import graph_function as gf
import random
import timeit

from gensim.models import Word2Vec
def f_Word2vec(graphe,corpus):

    # Défini le modèle 
    print(corpus)
    model = Word2Vec(min_count=0, vector_size=100, epochs=10)  
    
    # Construit le vocabulaire et entraine le model
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def embedding_mot(vocab_size,target_idx,W1):
    x = np.zeros(vocab_size)
    x[target_idx] = 1
    h = np.dot(W1.T, x)
    return h

def sigmoid(x):
    """Fonction sigmoïde numériquement stable"""
    return 1 / (1 + np.exp(-x))

def create_pairs(corpus, window_size):
    """Génère des paires (mot cible, mot contexte)"""
    pairs = []
    for phrase in corpus:
        for i, target_word in enumerate(phrase):
            for j in range(-window_size, window_size + 1):
                if j != 0 and 0 <= i + j < len(phrase):
                    pairs.append((target_word, phrase[i + j]))
    return pairs

def negative_sampling(vocab_size, num_samples, context_idx):
    """Sélectionne des mots négatifs aléatoires pour l'entraînement"""
    negatives = []
    while len(negatives) < num_samples:
        neg_word = random.randint(0, vocab_size - 1)
        if neg_word != context_idx:  # Éviter d'ajouter le vrai mot contexte
            negatives.append(neg_word)
    return negatives

def forward(target_idx, context_idx, neg_samples, W1, W2):
    """Effectue la passe avant pour l'entraînement avec negative sampling"""
    h = W1[target_idx]  # Représentation du mot cible
    
    # Score du mot contexte réel
    u_pos = np.dot(h, W2[:, context_idx])
    
    # Scores des mots négatifs
    u_neg = np.dot(h, W2[:, neg_samples])  # (embedding_dim, num_samples)

    return h, u_pos, u_neg

def backprop(target_idx, context_idx, neg_samples, h, u_pos, u_neg, W1, W2, learning_rate):
    """Rétropropagation avec Negative Sampling"""
    
    # Erreurs
    error_pos = sigmoid(u_pos) - 1  # Erreur pour le mot contexte réel
    error_neg = sigmoid(u_neg)      # Erreur pour les mots négatifs

    # Mise à jour de W2 (matrice de sortie)
    W2[:, context_idx] -= learning_rate * error_pos * h  # Mise à jour pour le vrai mot contexte
    for i, neg_word in enumerate(neg_samples):
        W2[:, neg_word] -= learning_rate * error_neg[i] * h  # Mise à jour pour chaque mot négatif

    # Mise à jour de W1 (matrice d'embedding d'entrée)
    grad_W1 = error_pos * W2[:, context_idx] + np.dot(W2[:, neg_samples], error_neg)
    W1[target_idx] -= learning_rate * grad_W1

def f_word2vec_self_made(graphe,corpus,window_size,min_count,embedding_dim,learning_rate,epochs):
    """Word2Vec optimisé avec Negative Sampling"""

    num_negative_samples = 5  # Nombre d'exemples négatifs par mise à jour
    # Création du vocabulaire
    vocabulaire = set(word for phrase in corpus for word in phrase)
    vocab_size = len(vocabulaire)
    
    word2id = {word: i for i, word in enumerate(vocabulaire)}
    id2word = {i: word for i, word in enumerate(vocabulaire)}

    # Création des paires de mots
    pairs = create_pairs(corpus, window_size)
    print(f"Création de {len(pairs)} paires de mots.")

    # Initialisation des poids
    W1 = np.random.rand(vocab_size, embedding_dim) - 0.5
    W2 = np.random.rand(embedding_dim, vocab_size) - 0.5

    print("Début de l'entraînement...")
    i=0
    for epoch in range(epochs):
        loss = 0
        random.shuffle(pairs)  # Mélanger les paires pour éviter un biais
        
        start = timeit.default_timer()
        for target_word, context_word in pairs:
            
            target_idx = word2id[target_word]
            context_idx = word2id[context_word]
            i+=1
            gf.load(i,len(pairs)*epochs,10)

            # Échantillonnage négatif
            neg_samples = negative_sampling(vocab_size, num_negative_samples, context_idx)
            # Forward pass
            h, u_pos, u_neg = forward(target_idx, context_idx, neg_samples, W1, W2)
            # Calcul de la perte
            loss -= np.log(sigmoid(u_pos)) + np.sum(np.log(sigmoid(-u_neg)))
            # Backpropagation avec mise à jour des poids
            backprop(target_idx, context_idx, neg_samples, h, u_pos, u_neg, W1, W2, learning_rate)
            if i==1000:
                print("Temps estimé:", (timeit.default_timer() - start)*len(pairs)/1000)

    print("temps total de l'apprantissage: ",timeit.default_timer() - start)

    print("Fin de l'entraînement, retour des embeddings.")
    return id2word, W1, word2id

#timeit