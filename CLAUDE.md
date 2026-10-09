# Projet : challenge Humanoid (stage Robot Learning Research)

## Objectif
Utiliser mes propres vidéos de téléphone (pick-and-place avec la main) pour piloter un
Panda dans le simulateur LIBERO. Question de recherche : une démo humaine filmée au
téléphone vaut-elle une démo robot, à nombre de démos égal ?

## Pipeline
1. Calibration de la table avec un marqueur ArUco (DICT_4X4_50, id 0).
2. Pose 3D de la pince pouce-index : MediaPipe Hands + solvePnP, dans le repère de la table.
3. Position initiale de l'objet (couleur vive, homographie sur le plan de la table).
4. Retargeting vers l'effecteur du Panda, lissage, état de la pince.
5. Replay dans LIBERO (Colab), on garde uniquement les épisodes réussis.
6. Fine-tuning de SmolVLA avec LoRA via LeRobot (Colab).
7. Évaluation sur des positions d'objet jamais vues (20 épisodes par configuration).
Bonus seulement si le reste est fini : petit world model pour classer des actions candidates.

## Organisation
- Mac : traitement vidéo (CPU). Colab : LIBERO + entraînement (LIBERO ne tourne que sous Linux).
- data/raw/train : pickplace_posXX_repY.mp4 (XX = 01..08, Y = 1..5)
- data/raw/heldout : heldout_posXX_repY.mp4 (test, positions différentes de train)
- data/raw/calib : calib_marker.mp4, calib_gripper.mp4
- data/processed : un .npz par vidéo, même nom que la vidéo
- src/ : scripts numérotés (01_..., 02_...), results/ : GIFs, csv, courbes
- Ne jamais renommer ni modifier les vidéos brutes.

## Règles de travail
- Code simple, court, commenté. Pas de dépendances inutiles, pas de sur-ingénierie.
- Je dois comprendre chaque ligne : explique tes choix avant de coder.
- Ne lance une commande que dans ce dossier. Demande avant d'installer un paquet.
- Environnement : .venv à la racine (python3 -m venv), opencv-contrib-python uniquement.
- Résultats : toujours des chiffres mesurés, jamais d'affirmation sans mesure.
- README final sobre, sans emojis ni superlatifs, avec une section "ce qui n'a pas marché".