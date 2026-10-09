# Learning an action-conditioned video model from phone videos of my own hand

Application to the Humanoid Robot Learning Research internship challenge. Everything here comes from
data I recorded myself with a phone: 40 videos of a right hand picking up a small object from 8 marked
positions and dropping it into a bowl. I used Claude as a coding assistant; I ran every script and
checked every number reported below.

## What it does
1. Calibrates the table from the phone camera (8 crosses drawn 10 cm apart, plus an ArUco marker).
2. Tracks the hand with MediaPipe and turns each video into a thumb-index pinch trajectory.
3. Trains a small U-Net world model (64x64 images, 10 fps) that predicts the next frame from the current
   frame and the hand state (pinch position and opening).
4. Evaluates it on start positions never seen in training, against simple baselines.

## Data
- 40 videos (4K, 30 fps), 8 start positions x 5 repetitions, fixed phone camera, one bowl.
- 38 are usable: `pickplace_pos02_rep2` (no pinch gesture detected) and `pickplace_pos08_rep1` (gripper
  not open before the grasp) were rejected by the gesture detector.
- The camera moved slightly between some recordings (up to about 115 px), so every video is registered
  to a reference view with the ArUco marker.
- Raw videos are not in the repo (size). `data/processed/wm_data.npz` contains the frames and hand
  states used for training (crop zone `[264, 59, 2160, 2608]`).

## Accuracy of the hand pipeline (measured)
- Pinch position on the table, tested by touching the 8 crosses with the pinch: 2.0 cm mean error
  (1.6 to 2.8 cm) after registration. Without registration it was 7.5 cm.
- Grasp point vs the cross the object was placed on: 1.3 cm mean, 1.1 cm median, 3.0 cm max, over 38 videos.
- Height is not measured. Monocular depth (MediaPipe + PnP) placed the hand about 31 % too close to the
  camera in the contact test, so I do not use it. Heights follow an assumed profile (grasp at 1.0 cm).
  The release height (13.8 cm, std 2.0 cm across videos) is inferred by assuming the hand is above the
  centre of the bowl; the bowl centre comes from two ruler measurements.

## World model
- Small convolutional U-Net that predicts the change from the current frame. Inputs: the current frame
  and four maps built from the hand state (pinch position now and next as Gaussian blobs, opening now
  and next). The last layer starts at zero, so the model starts as "copy the last frame".
- Loss: L1, with pixels that change weighted 6x. Trained on 4-step rollouts for 4000 steps, best
  validation checkpoint kept. Light brightness augmentation.
- Split by start position: positions 03 and 06 are never seen in training (10 videos, test). Repetition 5
  of the other positions is validation (6 videos). 22 videos are used for training.
- Baselines: copy the last frame; the same model fed the hand of another video; the same model without
  the hand.
- The mask of changing pixels used in some metrics comes from the real future frame; it is used for
  measurement only.

### Results
Validation (positions seen in training, repetition 5, 4-step rollouts = 0.4 s, whole image, mean
absolute error on 512 fixed windows), model with hand: 0.0024, against 0.0043 for copying the last frame.
The whole-image error is dominated by pixels that do not move.

Test (positions 03 and 06, never seen in training): TODO (error by horizon, `results/wm_curves.png`)
Does the box follow the hand? TODO (position error of the green box, real vs predicted)
Rollouts: TODO (`results/rollout.gif`: real / prediction / prediction with another video's hand)

## What did not work
- Replaying my trajectories in LIBERO (task: alphabet soup into the basket, run on Colab). A scripted
  controller reaches 7 of 8 grid positions after shifting the grid 4 cm towards the robot. Replaying the
  human trajectory failed on both positions tested (the arm stalls, or the object is dropped). I stopped
  there and did not train a policy.
- PnP-based 3D hand height (see above).

## Limitations
- 38 videos, one scene, one hand. 64x64 images: the object is about 7 px wide.
- The "action" is the hand position in the image, not robot commands.

## Reproduce
Mac: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Download the
MediaPipe hand landmarker model into `models/hand_landmarker.task`. Then, in order: `src/01_calibrate.py`
(click the 8 crosses), `src/02_extract_hand.py`, `src/03_trajectory.py`, `src/04_release_height.py`,
`src/05_prepare_wm.py`.
Colab (GPU): open `notebooks/world_model.ipynb`, upload `data/processed/wm_data.npz` to `/content`, run the
cells in order. Trained weights are in `checkpoints/`.