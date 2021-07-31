Installing the package
=======================

If you are running a linux machine with pip installed, then you can install the EmotionStimuli package using pip.

.. code-block:: bash

  pip install .

In case if you have not installed pip or are facing trouble installing using git, you can install EmotionStimuli package using the provided setup.py

.. code-block:: bash

   python setup.py install

You will also be required to install the dependencies before running the project, so please install the from the `requirements.txt` file. 

.. code-block:: bash

  pip install -r requirements.txt
  
Also, you might be prompted for the installation of `en_core_web_sm` as it is the base tokenizer in the project, so please download the package using the following method before running it. 
  
.. code-block:: bash

  python -m spacy download en_core_web_sm
  

Setting up pretrained weights
------------------------------

The analysis module of the framework would require you to have the weights at a specific location. You can always change the location of the weights by editing the framework configuration in the config sub-module. If you are on linux machine, please download weights from https://github.com/Thanatoz-1/EmotionStimuli/releases/download/0.0.1/emotion_weights.zip and extract them in the folder called `emotion_weights` in your home folder. If you are facing issues with this, please update individual path in `emotion.config`. 
