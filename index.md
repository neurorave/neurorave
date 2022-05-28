# Neurorave

**This website is still under construction. We keep adding new results, so please come back later if you want more.**

This website presents additional material and experiments around the paper *Neurorave: Embedded deep audio synthesis with expressive control*.

Despite the significant advances in deep models for music generation, the use of these techniques is still restricted to expert users. Before being democratized among musicians, generative models have to face their long-standing challenge: providing *expressive control* over the music generation. In the context of instrument design, this renders the integration of deep generative models in creative environments an arduous task. In this paper, we tackle both of these issues by introducing a new deep generative model-based controllable hardware synthesizer: the NeuroRave. We enforce the controllability of a real-time audio generative model by explicitly disentangling salient musical features in the latent space by using an adversarial confusion criterion. User-specified features are then reintroduced as model conditioning, allowing for continuous generation *control*, akin to a synthesizer knob. We further integrate this lightweight model into embedded hardware, resulting in a flexible Eurorack format interacting with other classical modules. 

In short, NeuroRave combines cutting-edge neural audio synthesis as well as customizable expressive control, allowing the musician to access neural synthesis through a flexible interface. 

**Examples contents**
  * [Audio reconstruction](#audio-reconstruction)
  * [Single attribute control](#single-attribute-control)
  * [Multiple attributes control](#multiple-attributes-control)
  * [Datasets comparison](#datasets-comparison)
  * [Latent space analysis](#latent-space-analysis)
  * [Timbre and attribute transfers](#timbre-and-attribute-transfers)
  * [Joint prior generation](#joint-prior-generation)

**Code and implementation**
  * [Real-time implementation](#real-time-implementation)
  * [Hardware embedding](#hardware-embedding)
  * [Source code](#code)

**Additional details**
  * [Mathematical appendix](#mathematical-appendix)
  * [Models architecture](#models-details)

## Audio reconstruction

First, we compare the quality of various models (*VAE*, *RAVE*) in reconstructing an input from the test set, depending on whether it uses *conditioning* (*C-\**) or *faders* (\textit*F-\**). Then, for those two categories, we also evaluate how the control behaves by changing the attributes of the input sound with those of an out-of-distribution examples. We compute this for mono-attribute training (swapping only the *RMS*) or for multi-attribute training (swapping all atributes) cases. 

## Single attribute control

In this section, we further analyze how different methods behave in terms of control quality. To do so, we trained a separate model for each of the 6 descriptors, and a model for all descriptors at once (termed C-RAVE (m.) and F-RAVE (m.)). We analyze the correlation between target and output attributes when changing a single descriptor.

## Multiple attributes control

In this section, we further analyze how different methods behave in terms of control quality, by changing random sets of 2, 3, or 4 attributes at once.

## Datasets comparison

Here, we evaluate how our proposed F-RAVE model can be used on any type of sounds, by training on \textit{harmonic} (NSynth), \textit{percussive} (darbouka) and \textit{speech} (SC09) datasets in the multi-attribute setup. We display the reconstruction (\textit{Rec.}) and control (\textit{Ctr.}) results

## Latent space analysis

## Timbre and attribute transfers

## Joint prior generation

## Real-time implementation

Not available yet.

## Hardware embedding

Not available yet.


## Code

The full open-source code is currently available on the corresponding [GitHub repository](https://github.com/neurorave/neurorave). Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries that can be found in the README.

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on this current page.

## Models details

### Baseline models.

### RAVE
