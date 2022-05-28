
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

<style>
.page {
  width: calc(100%);
}
</style>

# Neurorave

**This website is still under construction. We keep adding new results, so please come back later if you want more.**

This website presents additional material and experiments around the paper *Neurorave: Embedded deep audio synthesis with expressive control*.

Despite the significant advances in deep models for music generation, the use of these techniques is still restricted to expert users. Before being democratized among musicians, generative models have to face their long-standing challenge: providing *expressive control* over the music generation. In the context of instrument design, this renders the integration of deep generative models in creative environments an arduous task. In this paper, we tackle both of these issues by introducing a new deep generative model-based controllable hardware synthesizer: the NeuroRave. We enforce the controllability of a real-time audio generative model by explicitly disentangling salient musical features in the latent space by using an adversarial confusion criterion. User-specified features are then reintroduced as model conditioning, allowing for continuous generation *control*, akin to a synthesizer knob. We further integrate this lightweight model into embedded hardware, resulting in a flexible Eurorack format interacting with other classical modules. 

In short, NeuroRave combines cutting-edge neural audio synthesis as well as customizable expressive control, allowing the musician to access neural synthesis through a flexible interface. 

## Audio reconstruction

First, we compare the quality of various models (*VAE*, *RAVE*) in reconstructing an input from the test set, depending on whether it uses *conditioning* (*C-\**) or *faders* (\textit*F-\**). Then, for those two categories, we also evaluate how the control behaves by changing the attributes of the input sound with those of an out-of-distribution examples. We compute this for mono-attribute training (swapping only the *RMS*) or for multi-attribute training (swapping all atributes) cases. 

## Single-attribute control

In this section, we further analyze how different methods behave in terms of control quality. To do so, we trained a separate model for each of the 6 descriptors, and a model for all descriptors at once (termed C-RAVE (m.) and F-RAVE (m.)). We analyze the correlation between target and output attributes when changing a single descriptor.

## Multi-attribute control

In this section, we further analyze how different methods behave in terms of control quality, by changing random sets of 2, 3, or 4 attributes at once.

## Datasets comparison

Here, we evaluate how our proposed F-RAVE model can be used on any type of sounds, by training on \textit{harmonic} (NSynth), \textit{percussive} (darbouka) and \textit{speech} (SC09) datasets in the multi-attribute setup. We display the reconstruction (\textit{Rec.}) and control (\textit{Ctr.}) results

## Latent space analysis

## Timbre and attribute transfers

## Joint prior generation

## Real-time implementation

Not available yet.


## Code

The full open-source code is currently available on the corresponding [GitHub repository](https://github.com/acids-ircam/flow_synthesizer). Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries that can be found in the README.

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on this current page.

## Models details

### Baseline models. 
In order to evaluate our proposal, we implemented several feed-forward deep models. In our context, it means that all the baseline models take the full spectrogram of one sample $$\mathbf{x}_i$$ as input and try to infer the corresponding synthesis parameters $$\mathbf{v}_i$$. All these models are trained with a Mean-Squared Error (MSE) loss computed between the output of the model and the ground-truth parameters vector.

#### Multi-Layer Perceptron (MLP)
First, we implement a 5-layers `MLP` with 2048 hidden units per layer, Exponential Linear Unit (ELU) activation, batch normalization and dropout with $$p=.3$$. This model is applied on a flattened version of the spectrogram and the final layer is a sigmoid activation.


### Auto-encoding models
We implemented various AE architectures, which are slightly more complex in terms of training as it involves two training signals. 

#### Families of auto-encoders (AE, VAE, WAE, VAEFlows)
First, we implement a simple deterministic `AE` without regularization. We implement the `VAE` by adding a KL regularization to the latent space and the `WAE` by replacing the KL by the MMD. Finally, we implement `VAEFlow` by adding a normalizing flow of 16 successive IAF transforms to the `VAE` posterior. 

### Optimization
We train all models for 500 epochs with ADAM, initial rate 2e-4, Xavier initialization and a scheduler that halves the rate if validation loss stalls for 20 epochs. With this setup, the most complex model only takes $\sim$5 hours to train on a NVIDIA Titan Xp GPU.

