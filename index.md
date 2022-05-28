<!--
<script src="http://vjs.zencdn.net/4.0/video.js"></script>
-->

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

<div class="figure">
    <table>
        <tr>
            <th>Model</th>
            <th>Sample</th>
            <th>Spectrogram</th>
            <th>Parameters</th>
        </tr>
        <tr>
            <td><b>Original preset</b></td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/original.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>VAE-Flow-post</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/vae_flow_mel_mse_cnn_flow_kl_f_iaf_1.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>VAE-Flow</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/vae_flow_mel_mse_cnn_mlp_iaf_2.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/cnn_mel_mse_1.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>MLP</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/mlp_mel_mse_1.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>VAE</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/vae_mel_mse_cnn_mlp_2.mp3">
                </audio>
            </td>
        </tr>
        <tr>
            <td>WAE</td>
            <td>
                <audio controls> 
                    <source src="audio/reconstruction/1_0/wae_mel_mse_cnn_mlp_2.mp3">
                </audio>
            </td>
        </tr>
    </table>
</div>

<br/>

## Single-attribute control

In this section, we further analyze how different methods behave in terms of control quality. To do so, we trained a separate model for each of the 6 descriptors, and a model for all descriptors at once (termed C-RAVE (m.) and F-RAVE (m.)). We analyze the correlation between target and output attributes when changing a single descriptor.

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

## Multi-attribute control

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

Here, we evaluate how our proposed F-RAVE model can be used on any type of sounds, by training on \textit{harmonic} (NSynth), \textit{percussive} (darbouka) and \textit{speech} (SC09) datasets in the multi-attribute setup. We display the reconstruction (\textit{Rec.}) and control (\textit{Ctr.}) results

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


## Code

The full open-source code is currently available on the corresponding [GitHub repository](https://github.com/acids-ircam/flow_synthesizer). Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries that can be found in the README.

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on this current page.

## Models details

### Baseline models. 
In order to evaluate our proposal, we implemented several feed-forward deep models. In our context, it means that all the baseline models take the full spectrogram of one sample $$\mathbf{x}_i$$ as input and try to infer the corresponding synthesis parameters $$\mathbf{v}_i$$. All these models are trained with a Mean-Squared Error (MSE) loss computed between the output of the model and the ground-truth parameters vector.

#### Multi-Layer Perceptron (MLP)
First, we implement a 5-layers `MLP` with 2048 hidden units per layer, Exponential Linear Unit (ELU) activation, batch normalization and dropout with $$p=.3$$. This model is applied on a flattened version of the spectrogram and the final layer is a sigmoid activation.

#### Convolutional Neural Network (CNN)
We implement a convolutional model composed of 5 layers with 128 channels of strided dilated 2-D convolutions with kernel size 7, stride 2 and an exponential dilation factor of $$2^{l}$$ (starting at $$l=0$$) with batch normalization and ELU activation. The convolutions are followed by a 3-layers MLP of 2048 hidden units with the same properties as the previous model.

#### Residual Networks (ResNet)
Finally, we implemented a *Residual Network*, with parameters settings identical to `CNN`. The normal path is defined as a convolution (similar to the previous model), Batch Normalization and ELU activation, while the residual paths are defined as a simple 1x1 convolution that maps to the size of the normal path. Both paths are then added.


### Auto-encoding models
We implemented various AE architectures, which are slightly more complex in terms of training as it involves two training signals. First, the traditional AE training is performed by using a MSE reconstruction loss between the input spectrogram $$\mathbf{x}_i$$ and reconstructed version $$\tilde{\mathbf{x}}_i$$. We use the previously described `CNN` setup for both encoders and decoders. However, we halve their number of parameters (by dividing the number of units and channels by 2) to perform a fair comparison by obtaining roughly the same capacity as the baselines.

All AEs map to latent spaces of dimensionality equal to the number of synthesis parameters (16 or 32). This also implies that the different normalizing flows will have a dimensionality equal to the numbers of parameters. We perform *warmup* by linearly increasing the latent regularization $$\beta$$ from 0 to 1 for 100 epochs. 

For all AE architectures, a second network is used to try to infer the parameters $$\mathbf{v}_i$$ based on the latent code $$\mathbf{z}_i$$ obtained by encoding a specific spectrogram $$\mathbf{x}_i$$. For this part, we train all simple AE models with a 2-layers MLP of 1024 units to predict the parameters based on the latent space, with a MSE loss. 

#### Families of auto-encoders (AE, VAE, WAE, VAEFlows)
First, we implement a simple deterministic `AE` without regularization. We implement the `VAE` by adding a KL regularization to the latent space and the `WAE` by replacing the KL by the MMD. Finally, we implement `VAEFlow` by adding a normalizing flow of 16 successive IAF transforms to the `VAE` posterior. 

### Our proposal
Finally, we evaluate *regression flows* ($$Flow_{reg}$$) by adding them to $$VAE_{flow}$$, with an IAF of length 16 without using semantic tags. Finally, we add the *disentangling flows* ($$Flow_{dis}$$) by introducing our objective defined in the paper.

### Optimization
We train all models for 500 epochs with ADAM, initial rate 2e-4, Xavier initialization and a scheduler that halves the rate if validation loss stalls for 20 epochs. With this setup, the most complex model only takes $\sim$5 hours to train on a NVIDIA Titan Xp GPU.

