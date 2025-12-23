# Vision-Language-Model-Based Construction Robot System  
## Demonstrators for Imitation Learning

This repository presents a vision-language-model-based robotic manipulation system designed for truss placement in a construction scenario.  
The system distills a SAM-enabled VLM pipeline into a lightweight, task-oriented behavior cloning framework that achieves fast inference and strong generalization.

---

## Action Representation Design

To reduce error accumulation and improve robustness, the policy predicts **task-level end-effector goals** instead of low-level joint commands.

The action formulation follows three key assumptions:

- **The placement height remains constant**, as both the truss and wooden frame lie on the same planar surface.
- **Orientation variation occurs only along the yaw axis**, corresponding to the two diagonal placement strategies.
- **Isaac Sim’s inverse kinematics (IK) solver** is used to generate intermediate joint trajectories.

Under this formulation, the learning problem is simplified to predicting a low-dimensional target pose, allowing the model to focus on spatial reasoning rather than kinematic control.

With this design, the system achieves **100% placement success using only 50 demonstrations**, significantly outperforming joint-space behavior cloning.

---

## Image-Based Policy Learning

To further reduce dependency on state variables, state-based observations were replaced with a **single RGB image captured at the 180th simulation frame**.  
This frame corresponds to a stable viewpoint immediately before the execution of the placement motion.

A lightweight **TinyCNN** architecture was trained to regress the target end-effector pose directly from the image.  
Despite its simplicity, the image-based policy demonstrates strong performance across all placement conditions and spatial variations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/14f743af-554d-4114-bd2c-3e34e7632fe7" width="45%" />
  <img src="https://github.com/user-attachments/assets/b48d2571-4e4b-4c7d-a27c-2cc56077d767" width="45%" />
</p>

<p align="center">
  <b>Pose 1</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Pose 2</b>
</p>

These results indicate that a **single-frame visual observation** is sufficient for reliable spatial inference in this structured manipulation task.

---

## Multimodal Vision-Language Policy

To incorporate explicit **instruction-level guidance** for placement orientation, a multimodal vision-language policy was introduced.

The key motivation behind this design is that **spatial localization and rotational intent are inherently decoupled**:
- The target placement position depends on the visual configuration of the wooden frame.
- The desired placement orientation is determined entirely by the language instruction.

---

### Architecture Overview

The final model adopts a **multi-head, multi-modal architecture**:

- **Image branch**
  - CNN-based encoder
  - Predicts target translation `(x, y)`

- **Text branch**
  - Frozen CLIP tokenizer and text encoder
  - Lightweight MLP head
  - Predicts yaw orientation

- **Multi-head output**
  - Translation and rotation are inferred independently
  - Aligns with the physical structure of the task

<img src="https://github.com/user-attachments/assets/2b45d3df-b740-4a58-9285-8896507fb3ab" width="100%" />

---

### Evaluation Results

The multimodal policy exhibits several desirable properties:

- **Consistent success across all test locations**
- **Robust generalization beyond demonstrated strategies**
- **Successful execution even in regions where only one placement strategy was observed during training**

These results demonstrate that the distilled vision-language model does not merely replicate demonstrations, but instead learns a structured representation of spatial and linguistic constraints.

---

## Extension: RNN with End-Effector Pose Actions

As an extension, the action space was further expanded to predict the **end-effector pose at every time step** using a BC-RNN formulation.  
The motivation was to reduce reliance on repeated IK calls and encourage temporally coherent trajectory generation.

Although the final placement accuracy was comparable, invoking the articulation controller at every simulation step introduced high-frequency oscillations of the grasped truss, preventing stable insertion into the target region.

In contrast, joint-space action formulations produced smoother and more stable trajectories that closely matched IK-generated references.

---

## Key Takeaways

- Reformulating the action space at the **task level** significantly improves learning efficiency.
- Decoupling translation and rotation aligns naturally with the physical structure of the task.
- Multimodal behavior cloning enables **generalization beyond demonstrated trajectories**.
- Lightweight distilled VLMs can outperform large segmentation-based pipelines in both speed and robustness.

---
