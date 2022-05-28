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

First, we compare the quality of various models (*VAE*, *RAVE*) in reconstructing an input from the test set, depending on whether it uses *conditioning* (*C-*) or *faders* (*F-*). Then, for those two categories, we also evaluate how the control behaves by changing the attributes of the input sound with those of an out-of-distribution examples. We compute this for mono-attribute training (swapping only the *RMS*) or for multi-attribute training (swapping all atributes) cases. 

## Single attribute control

In this section, we further analyze how different methods behave in terms of control quality. To do so, we trained a separate model for each of the 6 descriptors, and a model for all descriptors at once (termed C-RAVE (m.) and F-RAVE (m.)). We analyze the correlation between target and output attributes when changing a single descriptor.

## Multiple attributes control

In this section, we further analyze how different methods behave in terms of control quality, by changing random sets of 2, 3, or 4 attributes at once.

## Datasets comparison

Here, we evaluate how our proposed F-RAVE model can be used on any type of sounds, by training on *harmonic* (NSynth), *percussive* (darbouka) and *speech* (SC09) datasets in the multi-attribute setup. We display the reconstruction (*Rec.*) and control (*Ctr.*) results

## Latent space analysis

## Timbre and attribute transfers

## Joint prior generation

## Real-time implementation

Not available yet.

## Hardware embedding

Finally, in order to evaluate the creative quality of our model as a musical instrument, we introduce the *NeuroRave*, a prototype hardware synthesizer that generates music using our F-RAVE model. The interface is a module following the Eurorack specifications in order to allow for CV and gate interactions with other classical Eurorack modules. More precisely, alongside with the OLED screen and a RGB LED encoder button, our module features four CVs and two Gates. The software computation is handled by a [Jetson Nano](https://developer.nvidia.com/embedded), a mini-computer connected to our front board, which provides a 128-core GPU alongside with a Quad-core CPU.

### User interaction
The main *gate* input triggers the generation of the sound, while one of the CV handle the first latent dimension of the prior. The left CVs handle the sliding of the attributes, one by one. The last gate offers a more experimental control as it triggers the modification of all attributes at the same time, similarly to a macro-control. 


## Code

The full open-source code is currently available on the corresponding [GitHub repository](https://github.com/neurorave/neurorave). Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries that can be found in the README.

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on this current page.

## Models details

### Baseline models.

### RAVE
