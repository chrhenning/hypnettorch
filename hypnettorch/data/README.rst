Datasets
********

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

This folder contains data loaders for common datasets. Note, the code in this folder is a derivative of the dataloaders developed in `this <https://github.com/chrhenning/ann_implementations/tree/master/src/data>`__ repo. For examples on how to use these data loaders with Tensorflow checkout the `original code <https://github.com/chrhenning/ann_implementations>`__.

All dataloaders are derived from the abstract base class :class:`data.dataset.Dataset` to provide a common API to the user.

Preparation of datasets
=======================

**Datasets not mentioned in this section will be automatically downloaded and processed.**

Here you can find instructions about how to prepare some of the datasets for automatic processing.

Large-scale CelebFaces Attributes (CelebA) Dataset
--------------------------------------------------

`CelebA <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ is a dataset with over 200K celebrity images. It can be downloaded from `here <https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8>`__.

Google Drive will split the dataset into multiple zip-files. In the following, we explain, how you can extract these files on Linux. To decompress the sharded zip files, simply open a terminal, move to the downloaded zip-files and enter:

.. code-block:: console

    $ unzip '*.zip'

This will create a local folder named ``CelebA``.

Afterwards move into the ``Img`` subfolder:

.. code-block:: console

    $ cd ./CelebA/Img/

You can now decide, whether you want to use the JPG or PNG encoded images.

For the jpeg images, you have to enter:

.. code-block:: console

    $ unzip img_align_celeba.zip

This will create a folder ``img_align_celeba``, containing all images in jpeg format.

To save space on your local machine, you may delete the zip file via ``rm img_align_celeba.zip``.

The same images are also available in png format. To extract these, you have to move in the corresponding subdirectory via ``cd img_align_celeba_png.7z``. You can now extract the sharded 7z files by entering:

.. code-block:: console

    $ 7z e img_align_celeba_png.7z.001

Again, you may now delete the archives to save space via ``rm img_align_celeba_png.7z.0*``.

You can proceed similarly if you want to work with the original images located in the folder ``img_celeba.7z``.

FYI, there are scripts available (e.g., `here <https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py>`__), that can be used to download the dataset.

Imagenet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)
-------------------------------------------------------------------

The ILSVRC2012 dataset is a subset of the ImageNet dataset and contains over 1.2 Mio. training images depicting natural image scenes of 1,000 object classes. The dataset can be downloaded here `here <http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads>`__.

For our program to be able to use the dataset, it has to be prepared as described `here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__.

In the following, we recapitulate the required steps (which are executed from the directory in which the dataset has been loaded to).

1. Download the training and validation images as well as the `development kit for task 1 & 2 <http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz>`__.

2. Extract the training data.

   .. code-block:: console

        mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
        find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        cd ..

   **Note, this step deletes the the downloaded tar-file. If this behavior is not desired replace the command** ``rm -f ILSVRC2012_img_train.tar`` **with** ``mv ILSVRC2012_img_train.tar ..``.

3. Extract the validation data and move images to subfolders.

   .. code-block:: console

      mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
      wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
      cd ..

   This step ensures that the validation samples are grouped in the same folder structure as the training samples, i.e., validation images are stored under their corresponding WordNet ID (*WNID*).

4. Extract the meta data:
  
   .. code-block:: console
   
      mkdir meta && mv ILSVRC2012_devkit_t12.tar.gz meta/ && cd meta && tar -xvzf ILSVRC2012_devkit_t12.tar.gz --strip 1
      cd ..

Udacity Steering Angle Prediction
---------------------------------

The CH2 steering angle prediction dataset from Udacity can be downloaded `here <https://github.com/udacity/self-driving-car/tree/master/datasets/CH2>`__. In the following, we quickly explain how we expect the downloads to be preprocessed for our datahandler to work.

You may first decompress the files, after which you should have two subfolders ``Ch2_001`` (for the test data) and ````Ch2_002`` (for the training data). You may replace the file ``Ch2_001/HMB_3_release.bag`` with the complete test set ``Ch2_001/HMB_3.bag``.

We use `this docker tool <https://github.com/rwightman/udacity-driving-reader>`__ to extract the information from the Bag files and align the steering information with the recorded images.

Simply clone the repository and execute the ``./build.sh``. `This issue <https://github.com/rwightman/udacity-driving-reader/issues/24>`__ helped us to overcome an error during the build.

Afterwards, the bagfiles can be extracted using (note, that in- and output directory must be specified using absolute paths), for instance

.. code-block:: console

    sudo ./run-bagdump.sh -i /data/udacity/Ch2_001/ -o /data/udacity/Ch2_001/

and

.. code-block:: console

    sudo ./run-bagdump.sh -i /data/udacity/Ch2_002/ -o /data/udacity/Ch2_002/  

The data handler only requires the ``center/`` folder and the file ``interpolated.csv``. All remaining extracted data (for instance, left and right camera images) can be deleted.

Alternatively, the dataset can be downloaded from `here <https://academictorrents.com/details/5ac7e6d434aade126696666417e3b9ed5d078f1c>`__. This dataset appears to contain images recorded a month before the official Challenge 2 dataset was recorded. We could not find any information whether the experimental conditions are identical (e.g., whether steering angles are directly comparable). Additionally, the dataset appears to contain situations like parking, where the vehicle doesn't move and there is no road ahead. Anyway, if desired, the dataset can be processed similarly to the above mentioned. One may first want to filter the bag file, to only keep information relevant for the task at hand, e.g.:

.. code-block:: console

    rosbag filter dataset-2-2.bag dataset-2-2_filtered.bag "topic == '/center_camera/image_color' or  topic == '/vehicle/steering_report'"

The bag file can be extracted in to ``center/`` folder and a file ``interpolated.csv`` as described above, using ``./run-bagdump.sh``.