# SADIM
An unofficial implementation and experimentation on the model proposed in "Swapping Autoencoder for Deep Image Manipulation" by Park et al. (2020)

A considerable portion of this project is performed thanks to [rosinality](https://github.com/rosinality/swapping-autoencoder-pytorch). The official implementation is also available. Details were provided by Park et al. in the [official GitHub repository](https://github.com/taesungp/swapping-autoencoder-pytorch).

For further details of the architecture and training, please refer to the [paper](https://arxiv.org/abs/2007.00653).
```
@article{DBLP:journals/corr/abs-2007-00653,
  author    = {Taesung Park and
               Jun{-}Yan Zhu and
               Oliver Wang and
               Jingwan Lu and
               Eli Shechtman and
               Alexei A. Efros and
               Richard Zhang},
  title     = {Swapping Autoencoder for Deep Image Manipulation},
  journal   = {CoRR},
  volume    = {abs/2007.00653},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.00653},
  eprinttype = {arXiv},
  eprint    = {2007.00653},
  timestamp = {Mon, 06 Jul 2020 15:26:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2007-00653.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

For further ablation studies, I tested the qualitative effects of patch size that is utilized by the Patch Cooccurrence Discriminator. The qualitative results can be found in the repository.
