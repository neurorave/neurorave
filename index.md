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

First, we compare the quality of our models to perform pure reconstruction of an input from the test set, depending on whether it uses *conditioning* (*C-*) or *faders* (*F-*). Those models were computed on the *Darbuka* and *NSynth* datasets.


<div class="figure">
    <table style="width:100%;">
        <tr>
            <th style="width:50px"><b>Darbuka</b></th>
            <th>Original</th>
            <th>Reconstruction</th>
        </tr>
        <tr>
            <td>C-Rave</td>
            <td>
                <audio controls> 
                    <source src="audio/darbouka_c_darbouka_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/darbouka_c_darbouka_reconstruct.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td>F-Rave</td>
            <td>
                <audio controls> 
                    <source src="audio/darbouka_darbouka_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/darbouka_darbouka_reconstruct.wav">
                </audio>
            </td>
        </tr>
    </table>
</div>



<div class="figure">
    <table style="width:100%;">
        <tr>
            <th style="width:50px"><b>NSynth</b></th>
            <th>Original</th>
            <th>Reconstruction</th>
        </tr>
        <tr>
            <td>C-Rave</td>
            <td>
                <audio controls> 
                    <source src="audio/nsynth_c_nsynth_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/nsynth_c_nsynth_reconstruct.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td>F-Rave</td>
            <td>
                <audio controls> 
                    <source src="audio/nsynth_nsynth_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/nsynth_nsynth_reconstruct.wav">
                </audio>
            </td>
        </tr>
    </table>
</div>

<img src="audio/nsynth_nsynth_reconstruct.png" width="100%">
<img src="audio/nsynth_nsynth_control.png" width="100%">

<br/>

For the *Japanese* and *Violin* datasets, we only display results for our proposed *F-Rave* model.


<div class="figure">
    <table style="width:100%;">
        <tr>
            <th style="width:50px"><b>Japan</b></th>
            <th>Original</th>
            <th>Reconstruction</th>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls> 
                    <source src="audio/japanese_japanese_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/japanese_japanese_reconstruct.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls> 
                    <source src="audio/japanese_japanese_2_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/japanese_japanese_2_reconstruct.wav">
                </audio>
            </td>
        </tr>
    </table>
</div>



<div class="figure">
    <table style="width:100%;">
        <tr>
            <th style="width:50px"><b>Violin</b></th>
            <th>Original</th>
            <th>Reconstruction</th>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls> 
                    <source src="audio/violin_violin_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/violin_violin_reconstruct.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls> 
                    <source src="audio/violin_violin_2_original.wav">
                </audio>
            </td>
            <td>
                <audio controls> 
                    <source src="audio/violin_violin_2_reconstruct.wav">
                </audio>
            </td>
        </tr>
    </table>
</div>

## Single attribute control

In this section, we further analyze how different methods behave in terms of control quality. To do so, we trained a separate model for each of the 6 descriptors, and a model for all descriptors at once (termed C-RAVE (m.) and F-RAVE (m.)). We analyze the correlation between target and output attributes when changing a single descriptor.

As experiments, we first simulates the behavior faders sliding by taking rampes up, rampes down, sinusoides and sawtooth as modulators. Then, we select as modulators the attributes of other samples coming from the same dataset.

**Attributes coming from classical synthesizers control signals**

Original sound:
<img src="audio/eurorack_single/darbouka_darbouka.png" width="100%">
<audio controls> 
      <source src="audio/eurorack_single/darbouka_darbouka_reconstruction.wav">
</audio>

Modulated sound, we perform a sliding of attribute one by one:

<div class="figure">
    <table style="width:100%;">
        <tr>
            <th><img src="audio/eurorack_single/darbouka_darbouka_eurorack_0.png" width="19%"></th>
            <th><img src="audio/eurorack_single/darbouka_darbouka_eurorack_1.png" width="19%"></th>
            <th><img src="audio/eurorack_single/darbouka_darbouka_eurorack_2.png" width="19%"></th>
            <th><img src="audio/eurorack_single/darbouka_darbouka_eurorack_3.png" width="19%"></th>
            <th><img src="audio/eurorack_single/darbouka_darbouka_eurorack_4.png" width="19%"></th>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/eurorack_single/darbouka_darbouka_eurorack_0.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/eurorack_single/darbouka_darbouka_eurorack_1.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/eurorack_single/darbouka_darbouka_eurorack_2.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/eurorack_single/darbouka_darbouka_eurorack_3.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/eurorack_single/darbouka_darbouka_eurorack_4.wav">
                </audio>
            </td>
        </tr>  
    </table>
</div>

<img src="audio/eurorack_single/darbouka_darbouka_eurorack_0.png" width="19%">
<img src="audio/eurorack_single/darbouka_darbouka_eurorack_1.png" width="19%">
<img src="audio/eurorack_single/darbouka_darbouka_eurorack_2.png" width="19%">
<img src="audio/eurorack_single/darbouka_darbouka_eurorack_3.png" width="19%">
<img src="audio/eurorack_single/darbouka_darbouka_eurorack_4.png" width="19%">
<table style="width:100%;" border="0">
      <tr>
            <th> <audio controls style="width: 150px; display: block; margin:20px;"> 
                  <source src="audio/eurorack_single/darbouka_darbouka_eurorack_0.wav">
                  </audio> </th>
<th> <audio controls style="width: 150px; display: block; margin:20px;"> 
      <source src="audio/eurorack_single/darbouka_darbouka_eurorack_1.wav">
</audio> </th>
<th> <audio controls style="width: 150px; display: block; margin:20px;"> 
      <source src="audio/eurorack_single/darbouka_darbouka_eurorack_2.wav">
</audio> </th>
<th> <audio controls style="width: 150px; display: block; margin:20px;"> 
      <source src="audio/eurorack_single/darbouka_darbouka_eurorack_3.wav">
</audio> </th>
<th> <audio controls style="width: 150px; display: block; margin:20px;"> 
      <source src="audio/eurorack_single/darbouka_darbouka_eurorack_4.wav">
</audio> </th>
      </tr>
</table>

**Attributes coming from an other sample of the dataset**

<div class="figure">
    <table style="width:100%;">
        <tr>
            <th style="width:50px"><b>Dataset</b></th>
            <th>Darbouka</th>
            <th>Japanese</th>
            <th>Violin</th>
        </tr>
        <tr>
            <td>Original</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_audio.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_audio.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_audio.wav">
                </audio>
            </td>
        </tr>
        <tr>
            <td>Modulator</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_feats.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_feats.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_feats.wav">
                </audio>
            </td>
        </tr>
         <tr>
            <td>RMS</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_change_0.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_change_0.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_change_0.wav">
                </audio>
            </td>
        </tr>
         <tr>
            <td>Centroid</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_change_1.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_change_1.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_change_1.wav">
                </audio>
            </td>
        </tr>
         <tr> 
            <td>Bandwidth</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_change_2.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_change_2.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_change_2.wav">
                </audio>
            </td>
        </tr>
        <tr> 
            <td>Sharpness</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_change_3.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_change_3.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_change_3.wav">
                </audio>
            </td>
        </tr>
        <tr> 
            <td>Boominess</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/darbouka_darbouka_darbouka_0_763_1017_change_4.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/japanese_japanese_japanese_0_4637_2062_change_4.wav">
                </audio>
            </td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/single_attribute_sample_attr/violin_violin_violin_0_74_90_change_4.wav">
                </audio>
            </td>
        </tr>          
    </table>
</div>

The modification of attributes are clearly heard and it appears that the RMS and the centroid have a strong influence on the sound generation whereas the sharpness and the boominess have a more subtle effect. However, taking the attributes from an other sample of the dataset quickly degrades the quality on the Japanese dataset and on the violin. This is due to the abrupt change of attributes which can easily be outside of the range of the original attributes.

## Multiple attributes control

In this section, we further analyze how different methods behave in terms of control quality, by changing random sets of 2, 3, or 4 attributes at once. Similarly to the section above, we first take as modulator classical eurorack signals, then we select an other sample of the dataset as modulator.

## Timbre transfers

<div class="figure">
    <table style="width:100%;">
        <tr>
            <th>Source</th>
            <th>Audio</th>
            <th>Target</th>
            <th>Audio</th>
        </tr>
        <tr>
            <td>Violin</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_darbouka_violin_original.wav">
                </audio>
            </td>
            <td>Darbouka</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_darbouka_violin_reconstruct.wav">
                </audio>
            </td>
        </tr>
          
        <tr>
            <td>Darbouka</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_japanese_darbouka_original.wav">
                </audio>
            </td>
            <td>Japanese</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_japanese_darbouka_reconstruct.wav">
                </audio>
            </td>
        </tr>
          
        <tr>
            <td>Violin</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_japanese_violin_original.wav">
                </audio>
            </td>
            <td>Japanese</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_japanese_violin_reconstruct.wav">
                </audio>
            </td>
        </tr>
          
        <tr>
            <td>Darbouka</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_violin_darbouka_original.wav">
                </audio>
            </td>
            <td>Violin</td>
            <td>
                <audio controls style="width: 150px; display: block; margin:20px;"> 
                    <source src="audio/timbre_transfer/transfer_violin_darbouka_reconstruct.wav">
                </audio>
            </td>
        </tr>
      </table>
</div>

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
