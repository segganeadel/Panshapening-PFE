# Donnees
Pour l'entrainement et les tests, nous avons opté pour le dataset PanCollection, Ce dataset comporte 4 satellites (QuickBird, WorldView-3, GaoFen-2 et WorldView-2). 

L'absence de donnees d'entrainement pour le satelite WorldView-2 le rend inutilisable pour l'entrainement, et le manque de poids pre-entrane pour le satellite GaoFen-2 pour les autres architechtures dans DL-Pan nous a pousse a ne pas l'utiliser, car on ne pourra pas comparer les resultats de maniere juste. 

Nous avons donc opté pour les satellites QuickBird et WorldView-3. Les donnees sont disponibles sur le drive de PanCollection, et sont telechargeables en format (.h5py) qu'on de suite transforme en format (.mat) pour pouvoir charger des parties des donnees en memoire, contrairement a h5py qui charge tout en memoire et retourne des donnees incohérentes en cas de manque de memoire.

Les deux satellites comportent des donnes d'entrainement, validation et test, ceux d'entrainement et de validation ont une taille de 16x16 pixels pour l'image MS et 64x64 pixels pour l'image PAN, pour les tests deux types sont applicables, un a resolution reduite ou l'image de reference est disponible, et une a pleine resolution ou l'image de reference n'est pas disponible, appelees respectivement RR et FR, pour RR les tailles sont de 64x64 pixels pour l'image MS et 256x256 pixels pour l'image PAN, et pour FR les tailles sont de 128x128 pixels pour l'image MS et 512x512 pixels pour l'image PAN.

Avant d'obternir ces donnees, il est necessaire de passer par le pipeline de traitement illustre dans la figure 2.3, dans l'etape de downsampling de l'image MS (HRMS) on applique un filtre de degradation specifique a chaque satellite genere par la fonction MTF qui prend en parametre le nom du satellite et recupere d'un dictionnaire les valeurs Nyquist specifiques a chaque chaine spectrale suivi d'une interpolation pour reduire la taille de l'image.

# Entrainement
Pour faire une etude comparative equitable des architectures bassee sur deep learning mentionnees dans l'etat de l'art on a utilise les poids pre-entraines sur les memes donnees d'entrainement introduit dans le titre precedent, l'entraiment a ete fait ainsi.

Pour notre model mambafuse on entraine le model pour 100 epochs avec un batch de 32 , un taux d'apprentissage de 2e-4 en utilisant l'optimiseur Adam.