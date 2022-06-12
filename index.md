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

As experiments, we first take as modulators the attributes of other samples coming from the same dataset. Then, we simulates the behavior faders sliding by taking rampes up, rampes down, sinusoides and sawtooth as modulators. For both of the configurations, we also simulate a smoothed attenuation of the modulation effects by computing the mean of attributes between the original sample and the modulator. 

1. Attributes coming from an other sample of the dataset:

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

2. Smoothing by taking the mean of the modulator and the original sample:

3. Attributes coming from classical synthesizers modulators:

4. Smoothing by taking the mean of the modulator and the original sample:

<div class="figure">

    <p style="text-align: center; font-size: 20px">Metaparameter \(z_{5}\)</p>
    <img src="audio/meta_parameters/z5/z5.png" width="100%">

    <div align="middle">
        <audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_0.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_1.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_2.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_3.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_4.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_5.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_6.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_7.mp3">
        </audio>
    </div>
    <br/>

    <br/>
    <p align="middle"><b>Click <a href="javascript:look('divMetaParams');" title="More comparisons">here</a> to see more examples</b></p>
    
    <div id="divMetaParams" style="display: none;">
        
    <p style="text-align: center; font-size: 20px">Metaparameter \(z_{6}\)</p>
    <img src="audio/meta_parameters/z6/z6.png" width="100%">

    <div align="middle">
        <audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_0.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_1.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_2.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_3.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_4.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_5.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_6.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_7.mp3">
        </audio>
    </div>
    <br/>

    </div>

    
</div>
<br/>

## Multiple attributes control

In this section, we further analyze how different methods behave in terms of control quality, by changing random sets of 2, 3, or 4 attributes at once.


<div class="figure">

    <p style="text-align: center; font-size: 20px">Metaparameter \(z_{5}\)</p>
    <img src="audio/meta_parameters/z5/z5.png" width="100%">

    <div align="middle">
        <audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_0.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_1.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_2.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_3.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_4.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_5.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_6.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z5/dim_7.mp3">
        </audio>
    </div>
    <br/>

    <br/>
    <p align="middle"><b>Click <a href="javascript:look('divMetaParams');" title="More comparisons">here</a> to see more examples</b></p>
    
    <div id="divMetaParams" style="display: none;">
        
    <p style="text-align: center; font-size: 20px">Metaparameter \(z_{6}\)</p>
    <img src="audio/meta_parameters/z6/z6.png" width="100%">

    <div align="middle">
        <audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_0.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_1.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_2.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_3.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_4.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_5.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_6.mp3">
        </audio><!--
        --><audio controls style="width: 10.5%; padding: 0.5%">
            <source src="audio/meta_parameters/z6/dim_7.mp3">
        </audio>
    </div>
    <br/>

    </div>

    
</div>
<br/>

## Datasets comparison

Here, we evaluate how our proposed F-RAVE model can be used on any type of sounds, by training on *harmonic* (NSynth), *percussive* (darbouka) and *speech* (SC09) datasets in the multi-attribute setup. We display the reconstruction (*Rec.*) and control (*Ctr.*) results


<div class="figure">
    <table class="noRowLine neighborhood audioTable">
        <tr>
            <th rowspan="2">Audio</th>
            <th colspan="2">\(\mathbf{z}_0 + \mathcal{N}(0, 0.1)\)</th>
            <th rowspan="2">Audio space</th>
            <th colspan="2">\(\mathbf{z}_1 + \mathcal{N}(0, 0.1)\)</th>
            <th rowspan="2">Audio</th>
        </tr>
        <tr>
            <th>Parameters</th>
            <th>Spectrogram</th>
            <th>Spectrogram</th>
            <th>Parameters</th>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_0.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_0.png"></td>
            <td rowspan="8">AUDIO SPACE IMG</td>
            <td><img src="audio/neighborhood/n0/p1_dim_0.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_0.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_1.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_1.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_1.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_1.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_2.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_2.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_2.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_2.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_3.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_3.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_3.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_3.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_4.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_4.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_4.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_4.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_5.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_5.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_5.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_5.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_6.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_6.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_6.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_6.mp3"></audio></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/p0_dim_7.mp3"></audio></td>
            <td>PARAMS IMG</td>
            <td><img src="audio/neighborhood/n0/p0_dim_7.png"></td>
            <td><img src="audio/neighborhood/n0/p1_dim_7.png"></td>
            <td>PARAMS IMG</td>
            <td><audio controls><source src="audio/neighborhood/n0/p1_dim_7.mp3"></audio></td>
        </tr>
    </table>
</div>
<div class="figure">
    <table class="noRowLine neighborhood interpolation">
        <tr>
            <td><img src="audio/neighborhood/n0/interpolate_dim_0.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_1.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_2.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_3.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_4.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_5.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_6.png"></td>
            <td><img src="audio/neighborhood/n0/interpolate_dim_7.png"></td>
        </tr>
        <tr>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_0.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_1.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_2.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_3.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_4.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_5.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_6.mp3"></audio></td>
            <td><audio controls><source src="audio/neighborhood/n0/interpolate_dim_7.mp3"></audio></td>
        </tr>
    </table>
</div>

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
