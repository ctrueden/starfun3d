https://zenodo.org/records/10518151

This contains the 3 Stardist 3D models trained for the paper '3D Nuclei
Segmentation by Combining GAN Based Image Synthesis and Existing 3D
Manual Annotations'.

* model_sospim contains the weights of the model trained with the
  original sospim fluorescent images (stained with DAPI and SOX2 labels)
  and the manual annotations. The average size of one nucleus in pixel
  is [x=27, y=28, z=10].

* model_confocal contains the weights of the model trained with the
  synthethizes modules mimicking the confocal acquisitions (stained with
  the FUCCI label) and the manual annotations. The average size of one
  nucleus in pixel is [x=39, y=39, z=7].

* model_spinning contains the weights of the model trained with the
  synthethizes modules mimicking the spinning-disk acquisitions (stained
  with the DAPI label) and the manual annotations. The average size of
  one nucleus in pixel is [x=39, y=39, z=7].
