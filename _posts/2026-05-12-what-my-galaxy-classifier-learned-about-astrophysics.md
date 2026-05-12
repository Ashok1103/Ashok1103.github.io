---
title: "My Neural Network Rediscovered a 50-Year-Old Astronomy Problem. Here is How."
date: 2026-05-12
categories: [Machine Learning, Astrophysics]
tags: [deep-learning, computer-vision, grad-cam, pytorch, galaxy-morphology, resnet]
---

I trained a neural network to classify galaxy images into three morphological 
types. It reached 90% test accuracy across 17,736 images. Then I asked it to 
show me *where* it was looking when it made each decision. What it showed me 
turned out to be a well-known problem in observational astronomy, one that has 
frustrated researchers for decades. The model had no domain knowledge encoded. 
No textbook. No paper. It found this degeneracy on its own, purely from 
patterns in pixel data. That surprised me, and I think it is worth writing 
about.

---

## The Problem: Classifying Galaxies at Scale

Galaxies are not all the same shape. Broadly, they fall into a few 
morphological categories: **smooth ellipticals**, which look like featureless 
blobs of light; **spirals**, which have distinct disk structure and winding 
arms; and **edge-on or disturbed systems**, which include galaxies tilted 
toward us, merging pairs, and irregular shapes. These categories are not just 
aesthetic. Morphology traces a galaxy's formation history, its star formation 
rate, and how it has evolved over billions of years.

For small samples, astronomers classify galaxies by eye. Projects like Galaxy 
Zoo crowdsourced this to hundreds of thousands of volunteers. But next-generation 
surveys like the **DECam Legacy Survey**, **UNIONS**, and **Euclid** are 
imaging hundreds of millions of galaxies. At that scale, human classification 
is impossible. Automated pipelines are not optional anymore. They are the only 
path forward, and the accuracy of those pipelines directly affects the science 
that comes out of them.

This project is a small step toward understanding what those pipelines get 
right, and more importantly, where they fail and why.

---

## What Grad-CAM Is

Training a neural network to classify images is one thing. Understanding *why* 
it makes each prediction is another problem entirely. This is where 
**Grad-CAM** (Gradient-weighted Class Activation Mapping) comes in.

The analogy I find most useful: imagine you are reading a research paper and 
someone asks you to justify your conclusion. You would go back and highlight 
the sentences that led you there. Grad-CAM does something similar for a 
convolutional neural network. After the model makes a prediction, Grad-CAM 
traces which regions of the input image had the strongest influence on that 
specific prediction, then overlays a heatmap on the original image. Red means 
the model weighted that region heavily. Blue means it largely ignored it.

This is not just a visualization trick. It is a diagnostic tool. If a model 
classifies a spiral galaxy correctly but the heatmap lights up on a background 
star instead of the disk, something is wrong with what it has learned, even if 
the accuracy number looks fine.

![Grad-CAM heatmap on correctly classified spiral galaxies](/assets/img/posts/gradcam_spirals.png)
  Grad-CAM activation map for correctly classified spirals. 
  The model attends to the disk and spiral arms, not the central bulge.*
-->

---

## The Finding: A Known Degeneracy, Rediscovered

My model was trained progressively. A NumPy MLP built from scratch came first, 
with no PyTorch, just raw matrix operations and manually derived gradients. 
Then a custom 4-block CNN in PyTorch. Finally, a fine-tuned ResNet-18 using 
two-stage transfer learning.

| Model | Parameters | Test Accuracy |
|---|---|---|
| NumPy MLP (from scratch) | 1,082,115 | 75.8% |
| Custom 4-block CNN | 422,179 | 87.8% |
| ResNet-18 (fine-tuned) | 11,178,051 | 90.0% |

At 90% accuracy, the ResNet-18 was misclassifying roughly 1 in 10 images. 
When I applied Grad-CAM to the misclassified cases, a clear pattern emerged. 
The model was not confused at random. It was systematically misclassifying a 
specific type of galaxy: **edge-on disks** predicted as **smooth ellipticals**.

Looking at the heatmaps for these cases, the model was doing something 
reasonable. It was attending to the elongated central profile and the light 
concentration along the major axis. These are genuine features of edge-on disk 
galaxies. The problem is that at the resolution of the DECam Legacy Survey, 
those features look almost identical to a smooth elliptical galaxy viewed 
face-on. Both show a concentrated oval of light with no visible substructure. 
The difference is geometric: one is a disk seen edge-on, the other is a 
pressure-supported sphere. But in a 256x256 pixel image, that distinction can 
disappear entirely.

![Edge-on disk vs smooth elliptical at survey resolution](/assets/img/posts/gradcam_wrong.png)
     *Left: an edge-on disk galaxy misclassified as smooth elliptical. Right: 
     a true smooth elliptical. The Grad-CAM maps are nearly identical.*
-->

This is not a model failure. It is a **projection effect**, and astronomers 
know it well. Without additional data such as spectroscopy (which reveals 
rotation velocity and confirms disk structure) or multi-wavelength imaging 
(which separates dust lanes from stellar light), the ambiguity is real and 
physical. My model, trained on RGB images alone, hit exactly this wall. It 
rediscovered a degeneracy that took astronomers decades to characterize, with 
no prior knowledge beyond 17,736 labeled images.

---

## What This Means

Automated morphology pipelines for next-generation surveys will face this 
degeneracy at a scale of hundreds of millions of galaxies. A 10% misclassification 
rate that is systematically biased toward one class does not average out. It 
propagates into downstream science: galaxy evolution statistics, merger rate 
estimates, stellar mass functions.

The fix is not a better CNN. It is better data. Incorporating near-infrared 
imaging would reveal dust lanes invisible in optical RGB. Spectroscopic redshift 
surveys provide velocity dispersion data that distinguishes disks from 
ellipticals directly. The model is showing us what information is missing, 
which is arguably more useful than a higher accuracy number on a clean benchmark.

Here is the question I keep coming back to: if a model trained on images alone 
can recover a known astrophysical degeneracy as a systematic error pattern, 
what other physical limits are encoded in misclassification patterns we have 
not looked at carefully yet?

The live demo is available here if you want to try classifying a galaxy 
yourself: [Galaxy Morphology Classifier on Hugging Face](https://ashok1103-galaxy-morphology-classifier.hf.space)

The full project code, notebooks, and Grad-CAM analysis are on 
[GitHub](https://github.com/Ashok1103/Machine-Learning-Portfolio/blob/main/Galaxy%20Classifier).
