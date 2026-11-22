import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement des données MNIST
# Les CSV sont compressés (.gz) pour tenir sur GitHub, genfromtxt les lit directement
dossier_script = os.path.dirname(os.path.abspath(__file__))
train_data = np.genfromtxt(os.path.join(dossier_script, "mnist_train.csv.gz"), delimiter=",", skip_header=1)
test_data = np.genfromtxt(os.path.join(dossier_script, "mnist_test.csv.gz"), delimiter=",", skip_header=1)

#%% 2. Visualisation d'un exemple
ligne = 205
img = train_data[ligne, 1:].reshape(28, 28)
plt.figure()
plt.imshow(img, cmap='gray')
plt.title(f"Label: {train_data[ligne, 0]}")
plt.show()

# %% 3. Préparation des données d'entraînement
data = train_data[:, 1:]
labels = train_data[:, 0]
m, n = np.shape(data)

# Ajout du biais (colonne de 1) pour obtenir E_tilde
E_tilde = np.hstack((data, np.ones((m, 1))))

# 4. Encodage One-Hot des étiquettes (Matrice B)
B = np.zeros((m, 10))
for i in range(m):
    B[i, int(labels[i])] = 1

# 5. Apprentissage : Calcul de la matrice de poids W
# Utilisation de la pseudo-inverse pour résoudre E_tilde * W = B
W = np.linalg.pinv(E_tilde) @ B

# 6. Fonctions de prédiction
def f(x, W):
    """Calcule le score pour chaque classe."""
    # Ajout du 1 pour le biais au vecteur d'entrée
    xtilde = np.hstack((x, 1))
    return xtilde @ W  # Résultat de taille (10,)

def f_prediction(x, W):
    """Renvoie le chiffre prédit (indice du score maximum)."""
    return np.argmax(f(x, W))


# %% 7. Précision sur le jeu de test
X_test = test_data[:, 1:]
y_test = test_data[:, 0]
scores_test = np.hstack((X_test, np.ones((len(X_test), 1)))) @ W
precision = np.mean(np.argmax(scores_test, axis=1) == y_test)
print(f"Précision sur les {len(y_test)} images de test : {precision * 100:.2f} %")

# %% 9. Interface de dessin pour tester le modèle à la main
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

class DessinChiffre:
    def __init__(self, master, W):
        self.W = W
        self.master = master
        master.title("Dessine un chiffre")

        self.taille_canvas = 280  # 28*10, pour dessiner confortablement
        self.canvas = tk.Canvas(master, width=self.taille_canvas, height=self.taille_canvas, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Image PIL dessinée en parallèle du canvas (même contenu)
        self.image = Image.new("L", (self.taille_canvas, self.taille_canvas), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.peindre)

        self.rayon = 10

        bouton_predire = tk.Button(master, text="Prédire", command=self.predire)
        bouton_predire.grid(row=1, column=0, sticky="we")

        bouton_effacer = tk.Button(master, text="Effacer", command=self.effacer)
        bouton_effacer.grid(row=1, column=1, sticky="we")

        self.label_resultat = tk.Label(master, text="Dessine un chiffre puis clique sur Prédire", font=("Arial", 14))
        self.label_resultat.grid(row=2, column=0, columnspan=2)

    def peindre(self, event):
        x, y = event.x, event.y
        r = self.rayon
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def pretraiter(self, image):
        """Reproduit le format des images MNIST : chiffre recadré, redimensionné
        à l'intérieur d'une boîte de 20x20, puis centré par centre de masse
        dans un cadre de 28x28."""
        arr = np.array(image, dtype=np.float64)

        # Recadrage sur la boîte englobante du dessin (pixels non nuls)
        lignes = np.any(arr > 0, axis=1)
        colonnes = np.any(arr > 0, axis=0)
        y0, y1 = np.where(lignes)[0][[0, -1]]
        x0, x1 = np.where(colonnes)[0][[0, -1]]
        recadre = image.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))

        # Redimensionnement en conservant le ratio, dans une boîte de 20x20
        largeur, hauteur = recadre.size
        if largeur > hauteur:
            nouvelle_largeur = 20
            nouvelle_hauteur = max(1, round(hauteur * 20 / largeur))
        else:
            nouvelle_hauteur = 20
            nouvelle_largeur = max(1, round(largeur * 20 / hauteur))
        redim = recadre.resize((nouvelle_largeur, nouvelle_hauteur), Image.LANCZOS)

        # Centrage par centre de masse dans un cadre de 28x28
        cadre = Image.new("L", (28, 28), color=0)
        arr_redim = np.array(redim, dtype=np.float64)
        ys, xs = np.indices(arr_redim.shape)
        masse_totale = arr_redim.sum()
        cy = (ys * arr_redim).sum() / masse_totale
        cx = (xs * arr_redim).sum() / masse_totale
        decalage_x = round(14 - cx)
        decalage_y = round(14 - cy)
        cadre.paste(redim, (decalage_x, decalage_y))

        return np.array(cadre, dtype=np.float64).flatten()

    def effacer(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.taille_canvas, self.taille_canvas], fill=0)
        self.label_resultat.config(text="Dessine un chiffre puis clique sur Prédire")

    def predire(self):
        x = self.pretraiter(self.image)

        pred = f_prediction(x, self.W)
        scores = f(x, self.W)
        self.label_resultat.config(text=f"Chiffre prédit : {pred}")

        plt.figure()
        plt.bar(np.arange(10), scores)
        plt.xticks(range(0, 10))
        plt.title(f"Scores de confiance - Prédiction : {pred}")
        plt.xlabel("Chiffres")
        plt.ylabel("Score de confiance")
        plt.show()


root = tk.Tk()
app = DessinChiffre(root, W)
root.mainloop()