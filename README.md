# Vision-language-Model-Based-Construction-Robot-System-as-Demonstrators-for-Imitation-Learning

- The placement height remained constant.
- Orientation changes occurred only along the yaw axis.
- Isaac Sim’s IK solver was used for intermediate motion planning.

With this formulation, **100% placement success** was achieved using only 50 demonstrations.

---

## Image-Based Policy Learning
State-based observations were replaced with the RGB image from the **180th frame**. A lightweight TinyCNN was trained to predict the target end-effector pose directly from visual input.

The image-based policy demonstrated strong performance across all placement conditions.

| Pose 1 | Pose 2 |
|------|------|
| ![](Images/frame1.png) | ![](Images/frame2.png) |

---

## Multimodal Vision-Language Policy
To incorporate placement orientation instructions, a multimodal architecture was introduced.

### Architecture
- **Image branch**: CNN → predicts `(x, y)`
- **Text branch**: Frozen CLIP text encoder + MLP → predicts `yaw`
- **Multi-head output** for decoupled spatial and rotational reasoning

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/528ceb98-f455-4ebf-8aca-473664bd1c3c" />


### Results
- Consistent success across all test locations (up to 5 cm variation)
- Generalization beyond demonstrated strategies
- Successful execution even in regions where only one strategy was present in the training data

---

## Extension: RNN with End-Effector Pose Actions
The action space was further extended to predict the end-effector pose at every time step using a BC-RNN.

While final placement accuracy was comparable, invoking the articulation controller at every step caused oscillations of the grasped truss, leading to unstable placements. In contrast, joint-space actions produced smoother and more stable motion trajectories.

---

## Key Takeaways
- Distilled VLM-based behavior cloning significantly improves robustness and inference speed.
- Decoupling spatial placement and rotational intent enables better generalization.
- Image + language supervision allows the model to exceed the limitations of the original demonstrations.

---

## Acknowledgements
- Isaac Sim
- Robomimic
- CLIP (OpenAI)



