# CIRC - Cognitive Imaging Reconstruction and Captioning

## Authors
  - Mohamed Gallai
  - Oam Khatavkar
  - Pranavi Kolouju
  - Ramez Masad

## Introduction

The human brain is complex, and it's activity can provide key insight into how we perceive and process the visuals of our natural world. However, it is an impossible feat to know what visual cues the human brain is interpreting with only medical imaging data. In this project, we will utalize fMRI data to reconstruct high resolution images accompanied by rich text captions. By coupling a diffusion model with the CLIP image encoder, we will be able to produce these results. Through this approach, aim to build a tool that can help "explain" the connection between our neural and visual systems using bimodal representations.  

## Datasets

- **Natural Scenes Dataset (NSD)  - Reference Dataset**
  - *Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle, L.T., Nau, M., Caron, B., Pestilli, F., Charest, I., ... & Kay, K. (2021)*
  - high-resolution fMRI data from 8 subjects viewing thousands of natural scenes over the course of 30-40 sessions.
  - [Read here](https://naturalscenesdataset.org/)

- **Deep Image Reconstruction**  
  - *Shen, G., Horikawa, T., Majima, K., & Kamitani, Y. (2020)*
  - fMRI scans of 3 subjects responding to various visual stimuli, including natural images, geometric shapes, alphabet letters, and imagination cues.
  - [Read here](https://openneuro.org/datasets/ds001506/versions/1.3.1)

- **Generic Object Decoding**  
  - *Horikawa, T., & Kamitani, Y. (2019)*
  - fMRI scans from 5 subjects viewing images across multiple categories from ImageNet, performing a one-back image repetition task.
  - [Read here](https://openneuro.org/datasets/ds001246/versions/1.2.1)

## Related Literature

1. **Identifying Natural Images from Human Brain Activity**  
   - *Kay, K., Naselaris, T., Prenger, R., & Gallant, J. L. (2008)*  
   - Discusses how receptive-field models and fMRI data can be used to decode natural images viewed by individuals.
   - [Read here](https://www.nature.com/articles/nature06713#Sec2)  

2. **High-resolution Image Reconstruction with Latent Diffusion Models from Human Brain Activity**  
   - *Takagi, Y., & Nishimoto, S. (2024)*  
   - High-resolution image reconstruction from fMRI images using latent diffusion models without training or fine-tuning.
   - [Read here](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)  

3. **Improving Visual Image Reconstruction using Diffusion Models**  
   - *Takagi, Y., & Nishimoto, S. (2024)*  
   - Proposes the DiffMSR model with a Prior-Guide Large Window Transformer decoder.
   - [Read here](https://arxiv.org/abs/2306.11536)  

4. **Controllable Mind Visual Diffusion Model**  
   - *Zeng, B., Li, S., Liu, X., et al. (2024)*  
   - Introduces the Controllable Mind Visual Diffusion Model for extracting semantic and silhouette information from fMRI images.
   - [Read here](https://arxiv.org/abs/2305.10135)  


