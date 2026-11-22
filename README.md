# Reconnaissance de chiffres manuscrits par moindres carrés

Projet réalisé à l'IPSA. Le but est de reconnaître des chiffres écrits à la main
(base MNIST) sans réseau de neurones, seulement avec de l'algèbre linéaire.

## Principe

Chaque image fait 28 × 28 pixels. On la transforme en un vecteur de 784 valeurs,
auquel on ajoute un 1 pour le biais. Les étiquettes sont codées en one-hot dans
une matrice B (un 1 dans la colonne du bon chiffre).

On cherche la matrice W qui envoie au mieux les images d'entraînement sur leur
étiquette, au sens des moindres carrés. Elle se calcule directement avec la
pseudo-inverse :

```
W = pinv(E~) @ B
```

Pour une nouvelle image, le chiffre prédit est celui qui a le plus grand score.

## Résultat

Environ **86 %** de bonnes réponses sur les 10 000 images de test.

Le script ouvre aussi une petite fenêtre (Tkinter) où l'on peut dessiner un chiffre
à la souris. Le dessin est recadré, réduit dans une boîte de 20 × 20 et centré
comme les images de MNIST, puis le script affiche le chiffre prédit et le
diagramme des scores.

## Lancer le projet

1. Installer les bibliothèques :
   ```
   pip install -r requirements.txt
   ```
   La fenêtre de dessin utilise Tkinter, fourni avec Python sur Windows et macOS.
   Sur Linux, il faut parfois l'installer à part (`sudo apt install python3-tk`).
2. Lancer :
   ```
   python mnist_moindres_carres.py
   ```

Les données MNIST sont incluses, compressées (`mnist_train.csv.gz` et
`mnist_test.csv.gz`) pour respecter la limite de taille de GitHub. `np.genfromtxt`
les lit directement, il n'y a rien à décompresser. Le chargement prend une à deux
minutes.

## Fichiers

- `mnist_moindres_carres.py` : chargement, calcul de W, précision sur le test et
  fenêtre de dessin.
- `mnist_train.csv.gz` (60 000 images) et `mnist_test.csv.gz` (10 000 images).

Une version interactive est disponible sur mon site.
