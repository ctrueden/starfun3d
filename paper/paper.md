## Abstract

Nuclei segmentation is an important task in cell biology analysis that requires accurate and reliable methods, especially within complex low signal to noise ratio images with crowded cells populations. In this context, deep learning-based methods such as Stardist have emerged as the best performing solutions for segmenting nucleus. Unfortunately, the performances of such methods rely on the availability of vast libraries of ground truth hand-annotated data-sets, which become especially tedious to create for 3D cell cultures in which nuclei tend to overlap. In this work, we present a workflow to segment nuclei in 3D in such conditions when no specific ground truth exists. It combines the use of a robust 2D segmentation method, Stardist 2D, which have been trained on thousands of already available ground truth datasets, with the generation of pair of 3D masks and synthetic fluorescence volumes through a conditional GAN. It allows to train a Stardist 3D model with 3D ground truth masks and synthetic volumes that mimic our fluorescence ones. This strategy allows to segment 3D data that have no available ground truth, alleviating the need to perform manual annotations, and improving the results obtained by training Stardist with the original ground truth data.

## 1 INTRODUCTION

The popularity of 3D cell cultures, such as organoids or spheroids, has recently exploded due to their ability to offer valuable models to study human biology, far more physiologically relevant than 2D cultures ([Jensen and Teng, 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-14); [Kapałczyńska et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-15)). Nowadays, automatically acquiring hundreds of organoids in 3D has become a reality thanks to the advances in microscopy systems ([Beghin et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-2)). Life scientists have therefore access to distributions of nuclei in 3D for a wide diversity of cell types and growing conditions, a key feature forming the basis of advanced quantitative analysis of important cell functions. However, 3D cellular cultures inherently display a large diversity of nuclei features, shapes and arrangements. And 3D microscopy of such complex samples most often leads to images with lower contrast and/or signal to noise ratio as compared to 2D microscopy of single cellular layer, with overlapping nuclei according to the achieved optical sectioning. Hence, accurate and automated nuclei segmentation in these conditions has turned out to be highly complex. The bottleneck has therefore shifted from the acquisition to the downstream analysis and quantification steps.

Many computational solutions have been proposed over the years that uses traditional image processing methods to tackle this segmentation problem ([Caicedo et al., 2019](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-3); [Malpica et al., 1997](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-20); [Li et al., 2007](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-17)), in 2D and 3D. However, they are usually tailored for a specific application and do not generalize well, resulting in the necessity to adapt their parameters and ultimately preventing an automatic and bias-free analysis. In parallel, the last few years have witnessed the rapid emergence of convolutional neural networks (CNNs) as a method of choice for microscopy image segmentation, achieving an accuracy unheard of for a wide range of segmentation tasks. Among those, deep learning-based nuclei segmentation has been particularly active, with more than one hundred methods being developed since 2019 ([Mougeot et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-21)).

Supervised approaches, in which paired of images and masks are used for training, are the current state of the art for nuclei segmentation, with Stardist ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23); [Weigert et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-28)) and Cellpose ([Stringer et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-25)) having reached a prominent position. Their adoption has been facilitated by their integration into several open-source platforms ([von Chamier et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-26); [Gómez-de Mariscal et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-9); [Sofroniew et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-24)), in which life scientists are able to directly use several pre-trained models ([Caicedo et al., 2019](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-3)) to segment their images. Unfortunately, these pre-trained models are limited to the segmentation of nuclei in 2D, a fact directly related to the lack of available ground truth (GT) datasets in 3D. Determining the organization of nuclei in 3D therefore requires life scientists to annotate themselves their data, a daunting and timeconsuming task which turns out to be much more challenging than in 2D due to the lower image quality and more crowded cell populations.

Generating synthetic training datasets has therefore emerged as a potential solution to this problem through the use of conditional Generative Adversarial Networks (cGANs) ([Goodfellow et al., 2014](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-11); [Isola et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-13); [Zhu et al., 2017](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-30)). cGANs learn a mapping from an observed source image *x* and a random noise vector *z*, to a “transformed” target image *y, G* : *x, z* → *y* ([Fig. 1](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F1)). For nuclei segmentation, the use of such generative networks aims to generate realistic microscopy images (background, signal to noise ratio, inhomogeneity, …) of nuclei (target style) from binary masks (source style) by training the cGAN with paired or unpaired datasets ([Baniukiewicz et al., 2019](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-1); [Fu et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-7); [Wang et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-27)). Nevertheless, GANs are notoriously known to be difficult to train, with a large number of hyperparameters required to be tuned.

[![Figure 1:|440x254](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F1.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F1.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F1.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F1.large.jpg)

Figure 1:

Architecture of FCGAN. In addition to an image in the source style, the generator takes a noise image as input. The discriminator meant to determine if the generated synthetic image *G*(*x*) is real or fake is multi-scale. Please note that the FCGAN was trained on the whole images, and not on crops as shown in this illustration.

In this work, we present a workflow to segment nuclei in 3D when no specific GT exists by leveraging on cGANs to generate fluorescence synthetic volumes of nuclei from 3D masks. While life scientists start to be accustomed to using packaged deep learning methods for segmentation or classification, image synthesis is still far from being accessible to regular users. We therefore made the choice to specifically design our workflow to be accessible to nonexpert life-scientists, leveraging on already packaged methods, both for the segmentation and the image generation, with the aim to avoid any tedious handannotating steps. With that in mind, our workflow first relies on the segmentation of the individualized 2D planes of our acquired volumes using the already pre-trained, well established and now robust 2D segmentation model Stardist 2D *db*2018 ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23)). After modification of the instance masks by applying a distance transform and a Gaussian filter, we pair these transformed ‘GT’ masks with the fluorescent nuclei images to train a cGAN specifically designed for biological images, that do not require complex hyper-parameters tuning or specific cropping data selection ([Han et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-12)). It allows to generate 3D volumes of nuclei, that resemble the fluorescent volume we acquired, from existing 3D GT masks that were typically created for a different cell type or microscopy modality. Finally, those pairs of GT mask volumes and synthetic fluorescent volumes of nuclei are used to train a Stardist model in 3D dedicated to our 3D samples type and imaging modality.

Through this workflow, we demonstrate that using existing works to their full extent can (i) circumvent the requirement of tedious hand-annotating steps for the segmentation of nuclei in 3D in complex and crowded environment, and (ii) alleviate the need to develop new deep learning architectures in an already crowded field ([Mougeot et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-21)), while also facilitating their usage by life scientists. We applied this workflow on a couple of cell lines and microscopy modalities and show that we managed to have good qualitative segmentation results without having to specifically hand annotate new volumes. This approach greatly widens the scope of use of 3D segmentation methods in the rapidly growing and diversifying field of 3D cell culture analysis, where many imaging modalities are explored and cell types with their own morphology characteristics used.

## 2 MOTIVATION

We recently acquired several oncospheres in 3D of the colorectal cancer cell line HCT-116 ([Goodarzi et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-10)) expressing the nucleus fluorescent label Fucci with a confocal microscope, denoted as *I* *c*, that we aim to segment while not having any GT mask for that cell culture type and imaging modality. In parallel, we were kindly provided a pre-trained Stardist 3D model ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-1.gif.backup.1702613768.2508) with the corresponding GT dataset by the authors of ([Beghin et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-2)). The GT dataset was composed of 7 volumes that were manually annotated and that we will denote as ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-2.gif.backup.1702613768.3077) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-3.gif) for the images and masks, respectively. These 7 volumes were composed of different 3D cell culture types (oncospheres and neuroectoderms), nuclear staining (DAPI and SOX) and z-steps (500 nm and 1 μm). In addition, they have been acquired with the soSPIM imaging technology ([Galland et al., 2015](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-8)), a single-objective light-sheet microscope different from the confocal imaging modality used to acquire the oncospheres we aimed to segment. With this ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-4.gif) model, the auhtors of ([Beghin et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-2)) managed to automatically segment more than one hundred oncospheres and neuroectoderms acquired with the soSPIM imaging modality. However, applying ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-5.gif) to *I* *c* gave poor results, as it only managed to identify a portion of the nuclei with an overall oversegmentation, most probably due to the different cell morphology and imaging modality of *I* *c* as compared to the GT datasets used to train the model.

We therefore tried to generate synthetic volume having the same style than *I* *c* with SpCycleGAN ([Fu et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-7)), with the final objective of being able to train a new Stardist 3D model more adapted to our data. SpCycleGAN allows to generate synthetic volumes from binary masks without GT by being built on top of the unpaired image-to-image translation model CycleGAN ([Zhu et al., 2017](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-30)). Being unpaired, the network can use *I* *c* as target style without requiring the corresponding nuclei segmentation as source. Originally, the authors of ([Fu et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-7); [Wu et al., 2023](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-29)) generated 3D nuclei masks by filling volumes with binarized (deformed) ellipsoids even though it could be limiting as it may not faithfully represent the nuclei morphology. Having already a GT nuclei masks, we therefore decided to train SpCycleGAN with ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-6.gif)and *I* *c* to avoid this pitfall.

In our hand, SpCycleGAN failed to generate realistic microscopy images ([Fig. 2](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F2)). Scrutinizing the toy datasets available with the network, we tested different scaling and cropping of the data, as well as removing any crops having void regions during training. Unfortunately, none of our tests were conclusive. We hypothesize that this may be related to two reasons. First, volumes composing *I* *c* exhibit a high level of noise and an overall crowded and overlapping nuclei population. SpCycleGAN may fail to transfer all these features from binary masks. Second, SpCycleGAN may require parameters fine-tuning to work properly, but this goes against our objective of having a more generic workflow that could be used by nonexperts.

[![Figure 2:|293x440](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F2.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F2.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F2.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F2.large.jpg)

Figure 2:

SpCycleGAN fails to generate realistic microscopy images even with different scaling and cropping.

## 3 METHOD

The proposed method is based upon the use of well established and more robust deep-learning methods for nuclei segmentation (Stardist ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23); [Weigert et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-28))) and image synthesis (conditional GANs in a slightly modified version ([Han et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-12))). Ours choices are motivated by the fact that we wanted an as-simple-as-possible backbone. In this manner, we intend to show that a simple approach focusing on data manipulation combined with existing networks can perform qualitatively well and propose a workflow that could become generic to enable the segmentation of a large variety of 3D biological samples acquired upon different imaging modalities.

### 3.1 Fully-Conditional GAN

Contrary to natural images in which every local region contains relevant information, biological images are composed of a mix of informative and void regions. They are also inherently multiscale with both the large-scale spatial organization of cells and their individual morphology and texture being essential. Since GANs have been originally developed for natural images, they tend to struggle to capture these intertwined features and to fail to generate realistic void regions.

We therefore based our synthesis method in the fully-conditional GAN (FCGAN) architecture ([Han et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-12)), an improvement of GAN focused on modifications that allow to synthesize multi-scale biological images ([Fig. 1](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F1)). This architecture consists of a generator built upon the cascaded refinement network ([Chen and Koltun, 2017](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-4)) instead of the more traditional U-Net ([Ronneberger et al., 2015](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-22); [Isola et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-13)) architecture as it is less prone to mode collapse. They also added two modifications to the traditional GAN architecture. First, they modified the input noise vector *z* to a noise “image”, i.e. a 3D tensor with the first two dimensions corresponding to the spatial positions. Instead of a noise vector that limits the size of output images, modifying the noise image size allows to output synthetized images of arbitrary sizes. Second, they used a multi-scale discriminator. Having a fixed size discriminator limits the quality and coherency of synthetized images to a micro (object) or macro (global organization) level. Using a multi-scale discriminator ensure the generator to produce images both globally and locally accurate, a desired feature when dealing with biological images that exhibit several levels of organization. All these modifications allow to directly train the FCGAN with the acquired images, alleviating the need to fine-tune the parameters and resize or crop the void regions.

### 3.2 Image Synthesis Based 3D Stardist Training

In spite of its numerous advantages, FCGAN requires however paired images for training, therefore compelling us to provide corresponding mask and image pairs. Fortunately, 2D pre-trained models of well-established nuclei segmentation methods such as Stardist ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23)) or Cellpose ([Stringer et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-25)) have been trained with thousands of 2D GT masks and achieve now a high degree of robustness over a large variety of sample types. Consequently, we can directly determine the nuclei spatial distribution by segmenting each of the 2D frame (1024 1024) of the volumes of *I* *c* with the 2D pretrained *db*2018 ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-7.gif) model ([Fig. 3](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F3)) instead of simulating them. These 2D segmentation will be denoted ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-8.gif) .

[![Figure 3:|440x240](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F3.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F3.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F3.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F3.large.jpg)

Figure 3:

Nuclei segmentation with the 2D *db*2018 pretrained Stardist model results in sufficiently good results to train a conditional GAN, scale bar = 20 μm.

For image generation through the FCGAN model, binary and instance masks can lead to fuzzy results because their gradient can be too rough during backpropagation ([Long et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-19)) ([Fig. 4(a)](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F4)). We therefore applied a distance transform to our instance masks ([Long et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-19)) followed by an intensity normalization and a Gaussian filter (see Source code 1), which allows to remove intensity stiff jumps and dependency to the nuclei length, ultimately facilitating the style transfer performed by the FCGAN model ([Fig. 4(b)](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F4)).

[![Figure 4:|440x278](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F4.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F4.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F4.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F4.large.jpg)

Figure 4:

Manipulating instance masks to generate the FCGAN’s source style, scale bar = 5 μm. (a) After applying a distance transform on the binarized instance masks, the FCGAN generates plausible images but with visible artefacts related to the rough gradient of the transform (red arrows). (b) Applying a Gaussian filter after the mask binarization helps the FCGAN to learn a proper mapping between the source and target style.

[![Source code 1:|383x440](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F5.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F5.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F5.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F5.large.jpg)

Source code 1:

Transforming instance masks to the FCGAN’s source style.

The function described in Source code 1 was applied on each mask image of ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-9.gif.backup.1702613768.9636), to obtain a set of transformed mask images ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-10.gif) . Consequently, pairing ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-11.gif) (source style) with the corresponding images from *I* *c* (target style), allowed us to train the *FCGAN* *c* model.

Finally, we generated ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-12.gif), synthetic confocalstyled volumes ([Fig 5](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F6)) obtained by applying *FCGAN* to the transformed masks ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-13.gif.backup.1702613769.3241) of the 3D GT dataset ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-14.gif.backup.1702613769.3877). This was carried out after ensuring that the sizes of the masks composing ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-15.gif) were similar to the sizes of the nuclei present in *I* *c* . We then trained a 3D Stardist model called ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-16.gif) with the paired ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-17.gif) volumes and the corresponding 3D masks ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-18.gif) .

[![Figure 5:|440x273](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F6.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F6.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F6.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F6.large.jpg)

Figure 5:

Comparison of experimental confocal images (scale bar = 20 μm) and synthetic images generated with the FCGAN style transfer (scale bar = 10 μm).

## 4 RESULTS

To assess both qualitatively and quantitatively the segmentation accuracy of our workflow, we compared it to two baselines and used two different cell cultures acquired with different microscopy modalities. In addition to *I* *c*, we thus acquired with a spinning-disc confocal microscope 4 suspended S180-E-cad–GFP cells ([Chu et al., 2004](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-5)) spheroids which nuclei were stained with DAPI (denoted as *I* *SD*). Indeed, the volumes composing *I* *c* are highly challenging because of their crowded overlapping nuclei population and the overall level of noise. This makes them unsuitable to evaluate the segmentation accuracy of our workflow other than qualitatively, since it is even difficult by eye to determine the limits of each nucleus. On the contrary, *I* *SD* were composed of a lower number of cells over a shorter thickness (5-10 μm) ensuring to have a sparser distribution of nuclei per volume (90 nuclei for the 4 volumes) easier to asses. On the side of the segmentation, and in addition to ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-19.gif), the second baseline was obtained by training a 3D Stardist model ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-20.gif) to learn how to reconstruct 3D instance masks from binarized masks, i.e. by pairing ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-21.gif) with its binarization. 3D segmentation with ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-22.gif) were therefore done by applying the model directly on the binarized 2D masks obtained with the pre-trained ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-23.gif) model on *I* *c* and *I* *SD*.

### 4.1 Qualitative Assessment with *I* *c*

Applying ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-24.gif) ([Beghin et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-2)) gave poor results with the volumes composing *I* *c* for two reasons ([Fig. 6](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F7)). First, this model was trained with masks smaller than the nuclei present in *I* *c*, resulting in oversegmentation. Second, due to the different imaging modality, *I* *c* exhibited more noise with its nuclei having a different texture than the ones from ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-25.gif) . It resulted with ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-26.gif) missing a lot of nuclei when applied to *I* *c*. On the other hand, ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-27.gif) managed to segment more nuclei than ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-28.gif) .since the 2D segmentation provided by ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-29.gif) managed to identify more nuclei on a per frame basis. Nevertheless, results shows that ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-30.gif) struggles to handle isolated 2D masks, resulting in over-segmented small nuclei ([Fig. 7](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F8) and [Fig. 6](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F7)). On the contrary, the segmentation provided by ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-31.gif) gave better qualitative results than ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-32.gif) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-33.gif) ([Fig. 6](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F7)). In particular, ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-34.gif) managed to segment nuclei exhibiting a wide range of intensity values, from dim to intense, while preventing over-segmentation at the same time.

[![Figure 6:|440x232](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F7.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F7.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F7.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F7.large.jpg)

Figure 6:

Comparison of segmentation obtained with ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F7/embed/inline-graphic-35.gif) ([Beghin et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-2)), ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F7/embed/inline-graphic-36.gif)and our ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F7/embed/inline-graphic-37.gif) model trained with synthetic volumes.

[![Figure 7:|440x326](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F8.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F8.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F8.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F8.large.jpg)

Figure 7:

![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F8/embed/inline-graphic-38.gif) fails to properly handle isolated 2D masks by over-segmenting them.

### 4.2 Quantitative Assessment with *I* *SD*

Contrary to ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-39.gif) and *I* *c* that exhibited cells having a bright and homogeneous texture, *I* *SD* is composed of cells having a bright region surrounded by a dimer region which corresponds to the cell cytoplasm ([Fig. 8](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F9)left). Consequently, even if ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-40.gif) managed to identify more than 90% of the nuclei ([Table 1](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#T1)), the overall segmentation is imperfect. While the model identified almost all the brightest nuclei ([Fig. 8](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F9)), explaining the high number of True Positives (TP), False Positives (FP) results from the model separating the bright and dim regions of a cell as 2 objects. False Negatives (FN), on the other hand, mostly originates from some nuclei exhibiting a different texture, for which the model only identified a small part of the membrane. ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-41.gif) identified almost 99% of the nuclei present in *I* *SD* ([Table 1](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#T1)), thanks to ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-42.gif) performing well in segmenting the nuclei in this low density nuclei distribution. Nevertheless, the segmentation being done in a per-frame basis, it led to the wrong identification of some cytoplasm as nucleus, explaining the large number of FP. Finally, we used our workflow to train a new Stardist 3D model ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-43.gif) with paired of synthetic volumes ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-44.gif) (generated by transferring the style of *I* *SD* to the transformed masks ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-45.gif)) and masks ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-46.gif). Similarly to ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-47.gif) successfully detected 98% of the nuclei ([Table 1](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#T1)). Nevertheless, directly accounting the axial direction for segmentation, in comparison to merging 2D masks to 3D as in ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-48.gif), ended up with a lower number of errors with only 3 FP resulting from over-segmented cells.

* [View inline](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full)
* [View popup](https://www.biorxiv.org/highwire/markup/3543254/expansion?width=1000&height=500&iframe=true&postprocessors=highwire_tables%2Chighwire_reclass%2Chighwire_figures%2Chighwire_math%2Chighwire_inline_linked_media%2Chighwire_embed)

Table 1: Comparison of the segmentation accuracy between ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/T1/embed/inline-graphic-49.gif) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/T1/embed/inline-graphic-50.gif).

[![Figure 8:|440x227](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F9.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F9.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F9.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F9.large.jpg)

Figure 8:

Comparison of the segmentation provided by ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F9/embed/inline-graphic-51.gif.backup.1702613770.9231) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/F9/embed/inline-graphic-52.gif.backup.1702613770.9778), scale bar = 5 μm. Red arrows pinpoint regions with segmentation errors.

### 4.3 Discussion

Our workflow relies heavily on the FCGAN image synthesis capabilities which are, in our context, much more reliable than a CycleGAN thanks to the FCGAN’s multi-scale feature and paired training dataset. We have shown that using the pre-trained ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-53.gif.backup.1702613771.0376) model directly on the acquired microscopy images and modifying the obtained instance masks was sufficient to properly train a FCGAN model. On the contrary, the segmentation provided by ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-54.gif.backup.1702613771.0908) was not accurate enough to directly train a Stardist model in 3D such as ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-55.gif). This discrepancy is related to the fact that ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-56.gif) does not account for the axial direction, with several nuclei being therefore only segmented over one or two frames. It leads to a nuclei over-segmentation by ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-57.gif), and therefore a possibly larger number of FP and FN. In our workflow, ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-58.gif) is only used to learn the style transfer. Creation of the synthetic volumes is done by applying the new style to ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-59.gif), a dataset composed of hand annotated masks in which all the nucleus are accurately identified. Since our models ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-60.gif) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-61.gif) are trained with these synthetic volumes, they are mostly insensitive to the segmentation accuracy of ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-62.gif) and are tailored to segment nuclei having the same texture as the acquisitions they have been trained on, therefore outperforming both ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-63.gif) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-64.gif).

Importantly, our synthetic volumes ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-65.gif) and ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-66.gif.backup.1702613771.537) are composed of stacked generated 2D frames that does not guarantee intensity coherency in the axial direction ([Fig. 9](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F10)). They also lack smooth transitions between some appearing and disappearing nuclei as one would expect from a real acquisition. Nevertheless, our results confirm precedent findings ([Baniukiewicz et al., 2019](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-1); [Wu et al., 2023](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-29)) for which training segmentation networks with synthetic volumes allows to achieve good performance.

[![Figure 9:|420x440](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F10.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F10.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F10.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F10.large.jpg)

Figure 9:

As synthetic volumes are composed of stack generated frames, a lack of smooth transition between appearing and disappearing nuclei is visible (red arrows), scale bar = 10 μm.

As already explained by the authors of Stardist ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23); [Weigert et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-28)) and Cellpose ([Stringer et al., 2021](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-25)), we want to reinforce the fact that it is critical to train these models with data that have objects of similar sizes and features than the images we want to segment. Training Stardist with synthetic volumes as proposed in our workflow facilitates this process. It becomes indeed sufficient to resize [image] to generate images with nuclei exhibiting the same sizes and textures than the ones composing the acquired volumes we want to segment.

Finally, the border detection of the cells composing *I* *SD* by [image] was not perfect and can be explained by two reasons ([Fig. 8](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F9)). First, the 3D segmentation provided by Stardist is highly convex while *I* *SD* exhibits cells with irregular and possibly concave borders. Second, *I* *SD* is composed of volumes with saturated intensity, resulting in a compressed dynamics. This combined with the small number of volumes prevented FCGAN to learn a perfect style mapping.

## 5 CONCLUSION

In this work, our objective was not to develop a method achieving state of the art accuracy for segmenting nuclei in 3D. In the past years ([Mougeot et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-21)), nuclei segmentation has seen the development of hundreds of methods competing for the leading position in term of segmentation accuracy. Unfortunately, most of these techniques are out of reach for life science labs and imaging facilities because they can be limited to one operating system or do not provide source code, tutorial or toy datasets ([Mougeot et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-21)). We therefore wanted to show that it is also possible to achieve good qualitative segmentation by using already established methods such as Stardist ([Schmidt et al., 2018](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-23); [Weigert et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-28)) that are widely used in labs and facilities. Image synthesis with GANs is still, however, far from being easily accessible to life scientists. We therefore focused on a GAN architecture designed for the multi-scale organization of biological images. The FCGAN of Han ([Han et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-12)) was therefore an ideal choice as it allowed us to directly use our acquired microscopy images without cropping or finetuning parameters of the model.

All things considered, our workflow resulted in qualitatively good segmentation of microscopy volumes for which no GT existed. We were able to generate synthetic volumes having the style of different cell types and microscopy modalities from the same set of 3D GT masks ([Fig. 10](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#F11)). Pairing them together, we trained several Stardist models tailored for each acquisition, managing to segment datasets without spending weeks in annotating volumes. We also quantitatively demonstrated that our workflow performed better than a pre-trained Stardist 3D model on a limited set of 4 volumes. In the future, we plan to push further this quantification as wells as further test its generalization capability by acquiring new datasets having a higher complexity than the *I* *SD* dataset. We also plan to test Omnipose ([Cutler et al., 2022](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-6)) in our workflow, with the aim to better identify irregular cell borders.

[![Figure 10:|437x440](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F11.medium.gif)](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F11.large.jpg?width=800&height=600&carousel=1)

* [Download figure](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F11.large.jpg?download=true)
* [Open in new tab](https://www.biorxiv.org/content/biorxiv/early/2023/12/12/2023.12.06.570366/F11.large.jpg)

Figure 10:

Generation of synthetic images from masks. From the same mask image (up), two synthetic images were generated with a different style (bottom).

Another point to consider is that the image synthesis provided by FCGAN will always be heavily dependent on the quality of the nuclei 2D segmentation. In our case, the pre-trained ![Embedded Image](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2023/12/12/2023.12.06.570366/embed/inline-graphic-69.gif) model gave satisfactory segmentation for feeding the FCGAN model. Otherwise, it will be necessary to generate new 2D manual annotations, a task still a lot easier than 3D annotations that can be accelerated by techniques such as SAM ([Kirillov et al., 2023](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-16)).

Finally, we think that this workflow could be improved by further manipulating the existing GT masks. GANs require similar distributions of the objects morphology and spatial organization between the source and target styles to generate realistic images ([Liu et al., 2020](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#ref-18)). Similarly, Stardist would also benefit from training with pairs having the same distributions that the volumes to segment. This requires the ability to quantify the differences between these distributions, as it would allow to modify the GT masks to make them similar to the objects one would want to segment.

## REFERENCES

1. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-1-1)

Baniukiewicz, P., Lutton, E. J., Collier, S., and Bretschneider, T. (2019). Generative adversarial networks for augmenting training data of microscopic cell images. Frontiers in Computer Science, 1.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=P.+Baniukiewicz&author[1]=E. J.+Lutton&author[2]=S.+Collier&author[3]=T.+Bretschneider&title=Generative+adversarial+networks+for+augmenting+training+data+of+microscopic+cell+images&publication_year=2019&journal=Frontiers+in+Computer+Science&volume=1)

2. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-2-1)

Beghin, A., Grenci, G., Sahni, G., Guo, S., Rajendiran, H., Delaire, T., Mohamad Raffi, S. B., Blanc, D., de Mets, R., Ong, H. T., Galindo, X., Monet, A., Acharya, V., Racine, V., Levet, F., Galland, R., Sibarita, J.-B., and Viasnoff, V. (2022). Automated high-speed 3d imaging of organoid cultures with multi-scale phenotypic quantification. Nature Methods, 19(7):881–892.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=A.+Beghin&author[1]=G.+Grenci&author[2]=G.+Sahni&author[3]=S.+Guo&author[4]=H.+Rajendiran&author[5]=T.+Delaire&author[6]=S. B.+Mohamad Raffi&author[7]=D.+Blanc&author[8]=R.+de Mets&author[9]=H. T.+Ong&author[10]=X.+Galindo&author[11]=A.+Monet&author[12]=V.+Acharya&author[13]=V.+Racine&author[14]=F.+Levet&author[15]=R.+Galland&author[16]=J.-B.+Sibarita&author[17]=V.+Viasnoff&title=Automated+high-speed+3d+imaging+of+organoid+cultures+with+multi-scale+phenotypic+quantification&publication_year=2022&journal=Nature+Methods&volume=19&pages=881-892)

3. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-3-1)

Caicedo, J. C., Goodman, A., Karhohs, K. W., Cimini, B. A., Ackerman, J., Haghighi, M., Heng, C., Becker, T., Doan, M., McQuin, C., Rohban, M., Singh, S., and Carpenter, A. E. (2019). Nucleus segmentation across imaging experiments: the 2018 data science bowl. Nature Methods, 16(12):1247–1253.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=J. C.+Caicedo&author[1]=A.+Goodman&author[2]=K. W.+Karhohs&author[3]=B. A.+Cimini&author[4]=J.+Ackerman&author[5]=M.+Haghighi&author[6]=C.+Heng&author[7]=T.+Becker&author[8]=M.+Doan&author[9]=C.+McQuin&author[10]=M.+Rohban&author[11]=S.+Singh&author[12]=A. E.+Carpenter&title=Nucleus+segmentation+across+imaging+experiments:+the+2018+data+science+bowl&publication_year=2019&journal=Nature+Methods&volume=16&pages=1247-1253)

4. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-4-1)

Chen, Q. and Koltun, V. (2017). Photographic image synthesis with cascaded refinement networks.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Chen%2C+Q.+and+Koltun%2C+V.+(2017).+Photographic+image+synthesis+with+cascaded+refinement+networks.)

5. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-5-1)

Chu, Y.-S., Thomas, W. A., Eder, O., Pincet, F., Perez, E., Thiery, J. P., and Dufour, S. (2004). Force measurements in E-cadherin–mediated cell doublets reveal rapid adhesion strengthened by actin cytoskele-ton remodeling through Rac and Cdc42 . Journal of Cell Biology, 167(6):1183–1194.

[Abstract/FREE Full Text](https://www.biorxiv.org/lookup/ijlink/YTozOntzOjQ6InBhdGgiO3M6MTQ6Ii9sb29rdXAvaWpsaW5rIjtzOjU6InF1ZXJ5IjthOjQ6e3M6ODoibGlua1R5cGUiO3M6NDoiQUJTVCI7czoxMToiam91cm5hbENvZGUiO3M6MzoiamNiIjtzOjU6InJlc2lkIjtzOjEwOiIxNjcvNi8xMTgzIjtzOjQ6ImF0b20iO3M6NDg6Ii9iaW9yeGl2L2Vhcmx5LzIwMjMvMTIvMTIvMjAyMy4xMi4wNi41NzAzNjYuYXRvbSI7fXM6ODoiZnJhZ21lbnQiO3M6MDoiIjt9)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=Y.-S.+Chu&author[1]=W. A.+Thomas&author[2]=O.+Eder&author[3]=F.+Pincet&author[4]=E.+Perez&author[5]=J. P.+Thiery&author[6]=S.+Dufour&title=Force+measurements+in+E-cadherin–mediated+cell+doublets+reveal+rapid+adhesion+strengthened+by+actin+cytoskele-ton+remodeling+through+Rac+and+Cdc42&publication_year=2004&journal=Journal+of+Cell+Biology&volume=167&pages=1183-1194)

6. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-6-1)

Cutler, K. J., Stringer, C., Lo, T. W., Rappez, L., Stroustrup, N., Brook Peterson, S., Wiggins, P. A., and Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. Nature Methods, 19(11):1438–1448.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=K. J.+Cutler&author[1]=C.+Stringer&author[2]=T. W.+Lo&author[3]=L.+Rappez&author[4]=N.+Stroustrup&author[5]=S.+Brook Peterson&author[6]=P. A.+Wiggins&author[7]=J. D.+Mougous&title=Omnipose:+a+high-precision+morphology-independent+solution+for+bacterial+cell+segmentation&publication_year=2022&journal=Nature+Methods&volume=19&pages=1438-1448)

7. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-7-1)

Fu, C., Lee, S., Ho, D. J., Han, S., Salama, P., Dunn, K. W., and Delp, E. J. (2018). Three dimensional fluorescence microscopy image synthesis and segmentation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2302–23028.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=C.+Fu&author[1]=S.+Lee&author[2]=D. J.+Ho&author[3]=S.+Han&author[4]=P.+Salama&author[5]=K. W.+Dunn&author[6]=E. J.+Delp&title=Three+dimensional+fluorescence+microscopy+image+synthesis+and+segmentation&publication_year=2018&journal=In+2018+IEEE/CVF+Conference+on+Computer+Vision+and+Pattern+Recognition+Workshops+(CVPRW)&pages=2302-23028)

8. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-8-1)

Galland, R., Grenci, G., Aravind, A., Viasnoff, V., Studer, V., and Sibarita, J.-B. (2015). 3d high- and super-resolution imaging using single-objective spim. Nature Methods, 12(7):641–644.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=R.+Galland&author[1]=G.+Grenci&author[2]=A.+Aravind&author[3]=V.+Viasnoff&author[4]=V.+Studer&author[5]=J.-B.+Sibarita&title=3d+high-+and+super-resolution+imaging+using+single-objective+spim&publication_year=2015&journal=Nature+Methods&volume=12&pages=641-644)

9. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-9-1)

Gómez-de Mariscal, E., García-López-de Haro, C., Ouyang, W., Donati, L., Lundberg, E., Unser, M., Muñoz-Barrutia, A., and Sage, D. (2021). Deepim-agej: A user-friendly environment to run deep learning models in imagej. Nature Methods, 18(10):1192–1195.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=E.+Gómez-de Mariscal&author[1]=C.+García-López-de Haro&author[2]=W.+Ouyang&author[3]=L.+Donati&author[4]=E.+Lundberg&author[5]=M.+Unser&author[6]=A.+Muñoz-Barrutia&author[7]=D.+Sage&title=Deepim-agej:+A+user-friendly+environment+to+run+deep+learning+models+in+imagej&publication_year=2021&journal=Nature+Methods&volume=18&pages=1192-1195)

10. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-10-1)

Goodarzi, S., Prunet, A., Rossetti, F., Bort, G., Tillement, O., Porcel, E., Lacombe, S., Wu, T.-D., Guerquin-Kern, J.-L., Delanoë-Ayari, H., Lux, F., and Rivière, C. (2021). Quantifying nanotherapeutic penetration using a hydrogel-based microsystem as a new 3d in vitro platform. Lab Chip, 21:2495–2510.

[CrossRef](https://www.biorxiv.org/lookup/external-ref?access_num=10.1039/D1LC00192B&link_type=DOI)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=S.+Goodarzi&author[1]=A.+Prunet&author[2]=F.+Rossetti&author[3]=G.+Bort&author[4]=O.+Tillement&author[5]=E.+Porcel&author[6]=S.+Lacombe&author[7]=T.-D.+Wu&author[8]=J.-L.+Guerquin-Kern&author[9]=H.+Delanoë-Ayari&author[10]=F.+Lux&author[11]=C.+Rivière&title=Quantifying+nanotherapeutic+penetration+using+a+hydrogel-based+microsystem+as+a+new+3d+in+vitro+platform&publication_year=2021&journal=Lab+Chip&volume=21&pages=2495-2510)

11. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-11-1)

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial networks.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Goodfellow%2C+I.+J.%2C+Pouget-Abadie%2C+J.%2C+Mirza%2C+M.%2C+Xu%2C+B.%2C+Warde-Farley%2C+D.%2C+Ozair%2C+S.%2C+Courville%2C+A.%2C+and+Bengio%2C+Y.+(2014).+Generative+adversarial+networks.)

12. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-12-1)

Han, L., Murphy, R. F., and Ramanan, D. (2020). Learning generative models of tissue organization with supervised gans.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Han%2C+L.%2C+Murphy%2C+R.+F.%2C+and+Ramanan%2C+D.+(2020).+Learning+generative+models+of+tissue+organization+with+supervised+gans.)

13. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-13-1)

Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. (2018). Image-to-image translation with conditional adversarial networks.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Isola%2C+P.%2C+Zhu%2C+J.-Y.%2C+Zhou%2C+T.%2C+and+Efros%2C+A.+A.+(2018).+Image-to-image+translation+with+conditional+adversarial+networks.)

14. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-14-1)

Jensen, C. and Teng, Y. (2020). Is it time to start transitioning from 2d to 3d cell culture? Frontiers in Molecular Biosciences, 7.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=C.+Jensen&author[1]=Y.+Teng&title=Is+it+time+to+start+transitioning+from+2d+to+3d+cell+culture?&publication_year=2020&journal=Frontiers+in+Molecular+Biosciences&volume=7)

15. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-15-1)

Kapałczyńska, M., Kolenda, T., Przybyła, W., Zajaczkowska, M., Teresiak, A., Filas, V., Ibbs, M., Blizńiak, R. łuczewski,, and Lamperska, K. (2018). 2d and 3d cell cultures – a comparison of different types of cancer cell cultures. Archives of Medical Science, 14(4):910–919.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=M.+Kapałczyńska&author[1]=T.+Kolenda&author[2]=W.+Przybyła&author[3]=M.+Zajaczkowska&author[4]=A.+Teresiak&author[5]=V.+Filas&author[6]=M.+Ibbs&author[7]=R. łuczewski+Blizńiak&author[8]=K.+Lamperska&title=2d+and+3d+cell+cultures+–+a+comparison+of+different+types+of+cancer+cell+cultures&publication_year=2018&journal=Archives+of+Medical+Science&volume=14&pages=910-919)

16. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-16-1)

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., and Girshick, R. (2023). Segment anything.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Kirillov%2C+A.%2C+Mintun%2C+E.%2C+Ravi%2C+N.%2C+Mao%2C+H.%2C+Rolland%2C+C.%2C+Gustafson%2C+L.%2C+Xiao%2C+T.%2C+Whitehead%2C+S.%2C+Berg%2C+A.+C.%2C+Lo%2C+W.-Y.%2C+Doll%C3%A1r%2C+P.%2C+and+Girshick%2C+R.+(2023).+Segment+anything.)

17. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-17-1)

Li, G., Liu, T., Tarokh, A., Nie, J., Guo, L., Mara, A., Holley, S., and Wong, S. T. (2007). 3d cell nuclei segmentation based on gradient flow tracking. BMC Cell Biology, 8(1):40.

[CrossRef](https://www.biorxiv.org/lookup/external-ref?access_num=10.1186/1471-2121-8-40&link_type=DOI)[PubMed](https://www.biorxiv.org/lookup/external-ref?access_num=17784958&link_type=MED&atom=%2Fbiorxiv%2Fearly%2F2023%2F12%2F12%2F2023.12.06.570366.atom)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=G.+Li&author[1]=T.+Liu&author[2]=A.+Tarokh&author[3]=J.+Nie&author[4]=L.+Guo&author[5]=A.+Mara&author[6]=S.+Holley&author[7]=S. T.+Wong&title=3d+cell+nuclei+segmentation+based+on+gradient+flow+tracking&publication_year=2007&journal=BMC+Cell+Biology&volume=8)

18. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-18-1)

Liu, Q., Gaeta, I. M., Millis, B. A., Tyska, M. J., and Huo, Y. (2020). Gan based unsupervised segmentation: Should we match the exact number of objects.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Liu%2C+Q.%2C+Gaeta%2C+I.+M.%2C+Millis%2C+B.+A.%2C+Tyska%2C+M.+J.%2C+and+Huo%2C+Y.+(2020).+Gan+based+unsupervised+segmentation%3A+Should+we+match+the+exact+number+of+objects.)

19. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-19-1)

Long, J., Yan, Z., Peng, L., and Li, T. (2021). The geometric attention-aware network for lane detection in complex road scenes. PLOS ONE, 16(7):1–15.

[CrossRef](https://www.biorxiv.org/lookup/external-ref?access_num=10.1371/journal.pone.0249644&link_type=DOI)[PubMed](https://www.biorxiv.org/lookup/external-ref?access_num=http://www.n&link_type=MED&atom=%2Fbiorxiv%2Fearly%2F2023%2F12%2F12%2F2023.12.06.570366.atom)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=J.+Long&author[1]=Z.+Yan&author[2]=L.+Peng&author[3]=T.+Li&title=The+geometric+attention-aware+network+for+lane+detection+in+complex+road+scenes&publication_year=2021&journal=PLOS+ONE&volume=16&pages=1-15)

20. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-20-1)

Malpica, N., de Solórzano, C. O., Vaquero, J. J., Santos, A., Vallcorba, I., García-Sagredo, J. M., and del Pozo, F. (1997). Applying watershed algorithms to the segmentation of clustered nuclei. Cytometry, 28(4):289–297.

[CrossRef](https://www.biorxiv.org/lookup/external-ref?access_num=10.1002/(SICI)1097-0320(19970801)28:4<289::AID-CYTO3>3.0.CO;2-7&link_type=DOI)[PubMed](https://www.biorxiv.org/lookup/external-ref?access_num=9266748&link_type=MED&atom=%2Fbiorxiv%2Fearly%2F2023%2F12%2F12%2F2023.12.06.570366.atom)[Web of Science](https://www.biorxiv.org/lookup/external-ref?access_num=A1997XQ11000003&link_type=ISI)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=N.+Malpica&author[1]=C. O.+de Solórzano&author[2]=J. J.+Vaquero&author[3]=A.+Santos&author[4]=I.+Vallcorba&author[5]=J. M.+García-Sagredo&author[6]=F.+del Pozo&title=Applying+watershed+algorithms+to+the+segmentation+of+clustered+nuclei&publication_year=1997&journal=Cytometry&volume=28&pages=289-297)

21. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-21-1)

Mougeot, G., Dubos, T., Chausse, F., Péry, E., Graumann, K., Tatout, C., Evans, D. E., and Desset, S. (2022). Deep learning – promises for 3D nuclear imaging: a guide for biologists. Journal of Cell Science, 135(7):jcs258986.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=G.+Mougeot&author[1]=T.+Dubos&author[2]=F.+Chausse&author[3]=E.+Péry&author[4]=K.+Graumann&author[5]=C.+Tatout&author[6]=D. E.+Evans&author[7]=S.+Desset&title=Deep+learning+–+promises+for+3D+nuclear+imaging:+a+guide+for+biologists&publication_year=2022&journal=Journal+of+Cell+Science&volume=135)

22. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-22-1)

Ronneberger, O., Fischer, P., and Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Navab, N., Hornegger, J., Wells, W. M., and Frangi, A. F., editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241, Cham. Springer International Publishing.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Ronneberger%2C+O.%2C+Fischer%2C+P.%2C+and+Brox%2C+T.+(2015).+U-net%3A+Convolutional+networks+for+biomedical+image+segmentation.+In+Navab%2C+N.%2C+Hornegger%2C+J.%2C+Wells%2C+W.+M.%2C+and+Frangi%2C+A.+F.%2C+editors%2C+Medical+Image+Computing+and+Computer-Assisted+Intervention+%E2%80%93+MICCAI+2015%2C+pages+234%E2%80%93241%2C+Cham.+Springer+International+Publishing.)

23. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-23-1)

Schmidt, U., Weigert, M., Broaddus, C., and Myers, G. (2018). Cell detection with star-convex polygons. In Frangi, A. F., Schnabel, J. A., Davatzikos, C., Alberola-López, C., and Fichtinger, G., editors, Medical Image Computing and Computer Assisted Intervention – MICCAI 2018, pages 265–273, Cham. Springer International Publishing.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Schmidt%2C+U.%2C+Weigert%2C+M.%2C+Broaddus%2C+C.%2C+and+Myers%2C+G.+(2018).+Cell+detection+with+star-convex+polygons.+In+Frangi%2C+A.+F.%2C+Schnabel%2C+J.+A.%2C+Davatzikos%2C+C.%2C+Alberola-L%C3%B3pez%2C+C.%2C+and+Fichtinger%2C+G.%2C+editors%2C+Medical+Image+Computing+and+Computer+Assisted+Intervention+%E2%80%93+MICCAI+2018%2C+pages+265%E2%80%93273%2C+Cham.+Springer+International+Publishing.)

24. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-24-1)

Sofroniew, N., Lambert, T., Evans, K., Nunez-Iglesias, J., Bokota, G., Bussonnier, M., Peña-Castellanos, G., Winston, P., Yamauchi, K., Pop, D. D., Pam Liu, Z., Solak, A. C., alisterburt Buckley, G., Sweet, A., Gaifas, L., Lee, G., Rodríguez-Guerra, J., Clack, N., Bragantini, J., Migas, L., Hilsenstein, V., Mendonça, M. W., Haase, R., Hector Freeman, J., Boone, P., Lowe, A. R., and Gohlke, C. (2022). napari/napari:0.4.13rc0.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Sofroniew%2C+N.%2C+Lambert%2C+T.%2C+Evans%2C+K.%2C+Nunez-Iglesias%2C+J.%2C+Bokota%2C+G.%2C+Bussonnier%2C+M.%2C+Pe%C3%B1a-Castellanos%2C+G.%2C+Winston%2C+P.%2C+Yamauchi%2C+K.%2C+Pop%2C+D.+D.%2C+Pam+Liu%2C+Z.%2C+Solak%2C+A.+C.%2C+alisterburt+Buckley%2C+G.%2C+Sweet%2C+A.%2C+Gaifas%2C+L.%2C+Lee%2C+G.%2C+Rodr%C3%ADguez-Guerra%2C+J.%2C+Clack%2C+N.%2C+Bragantini%2C+J.%2C+Migas%2C+L.%2C+Hilsenstein%2C+V.%2C+Mendon%C3%A7a%2C+M.+W.%2C+Haase%2C+R.%2C+Hector+Freeman%2C+J.%2C+Boone%2C+P.%2C+Lowe%2C+A.+R.%2C+and+Gohlke%2C+C.+(2022).+napari%2Fnapari%3A0.4.13rc0.)

25. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-25-1)

Stringer, C., Wang, T., Michaelos, M., and Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1):100–106.

[CrossRef](https://www.biorxiv.org/lookup/external-ref?access_num=10.1038/s41592-020-01018-x&link_type=DOI)[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=C.+Stringer&author[1]=T.+Wang&author[2]=M.+Michaelos&author[3]=M.+Pachitariu&title=Cellpose:+a+generalist+algorithm+for+cellular+segmentation&publication_year=2021&journal=Nature+Methods&volume=18&pages=100-106)

26. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-26-1)

von Chamier, L., Laine, R. F., Jukkala, J., Spahn, C., Krentzel, D., Nehme, E., Lerche, M., Hernández-Pérez, S., Mattila, P. K., Karinou, E., Holden, S., Solak, A. C., Krull, A., Buchholz, T.-O., Jones, M. L., Royer, L. A., Leterrier, C., Shechtman, Y., Jug, F., Heilemann, M., Jacquemet, G., and Henriques, R. (2021). Democratising deep learning for microscopy with zerocostdl4mic. Nature Communications, 12(1):2276.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=L.+von Chamier&author[1]=R. F.+Laine&author[2]=J.+Jukkala&author[3]=C.+Spahn&author[4]=D.+Krentzel&author[5]=E.+Nehme&author[6]=M.+Lerche&author[7]=S.+Hernández-Pérez&author[8]=P. K.+Mattila&author[9]=E.+Karinou&author[10]=S.+Holden&author[11]=A. C.+Solak&author[12]=A.+Krull&author[13]=T.-O.+Buchholz&author[14]=M. L.+Jones&author[15]=L. A.+Royer&author[16]=C.+Leterrier&author[17]=Y.+Shechtman&author[18]=F.+Jug&author[19]=M.+Heilemann&author[20]=G.+Jacquemet&author[21]=R.+Henriques&title=Democratising+deep+learning+for+microscopy+with+zerocostdl4mic&publication_year=2021&journal=Nature+Communications&volume=12)

27. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-27-1)

Wang, J., Tabassum, N., Toma, T. T., Wang, Y., Gahlmann, A., and Acton, S. T. (2022). 3D GAN image synthesis and dataset quality assessment for bacterial biofilm. Bioinformatics, 38(19):4598–4604.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=J.+Wang&author[1]=N.+Tabassum&author[2]=T. T.+Toma&author[3]=Y.+Wang&author[4]=A.+Gahlmann&author[5]=S. T.+Acton&title=3D+GAN+image+synthesis+and+dataset+quality+assessment+for+bacterial+biofilm&publication_year=2022&journal=Bioinformatics&volume=38&pages=4598-4604)

28. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-28-1)

Weigert, M., Schmidt, U., Haase, R., Sugawara, K., and Myers, G. (2020). Star-convex polyhedra for 3d object detection and segmentation in microscopy. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 3655–3662, Los Alamitos, CA, USA. IEEE Computer Society.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Weigert%2C+M.%2C+Schmidt%2C+U.%2C+Haase%2C+R.%2C+Sugawara%2C+K.%2C+and+Myers%2C+G.+(2020).+Star-convex+polyhedra+for+3d+object+detection+and+segmentation+in+microscopy.+In+2020+IEEE+Winter+Conference+on+Applications+of+Computer+Vision+(WACV)%2C+pages+3655%E2%80%933662%2C+Los+Alamitos%2C+CA%2C+USA.+IEEE+Computer+Society.)

29. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-29-1)

Wu, L., Chen, A., Salama, P., Winfree, S., Dunn, K. W., and Delp, E. J. (2023). Nisnet3d: three-dimensional nuclear synthesis and instance segmentation for fluorescence microscopy images. Scientific Reports, 13(1):9533.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=L.+Wu&author[1]=A.+Chen&author[2]=P.+Salama&author[3]=S.+Winfree&author[4]=K. W.+Dunn&author[5]=E. J.+Delp&title=Nisnet3d:+three-dimensional+nuclear+synthesis+and+instance+segmentation+for+fluorescence+microscopy+images&publication_year=2023&journal=Scientific+Reports&volume=13)

30. [↵](https://www.biorxiv.org/content/10.1101/2023.12.06.570366v1.full#xref-ref-30-1)

Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Computer Vision (ICCV), 2017 IEEE International Conference on.

[Google Scholar](https://www.biorxiv.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&author[0]=J.-Y.+Zhu&author[1]=T.+Park&author[2]=P.+Isola&author[3]=A. A.+Efros&title=Unpaired+image-to-image+translation+using+cycle-consistent+adversarial+networks&publication_year=2017&journal=In+Computer+Vision+(ICCV),+2017+IEEE+International+Conference+on)
