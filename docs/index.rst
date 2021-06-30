.. EmotionStimuli documentation master file, created by
   sphinx-quickstart on Sun Jun 13 16:39:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EmotionStimuli's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   usage/installation
   usage/quickstart

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

What is Emotion Stimuli?
------------------------
Emotion classification in text has wide array of applications which include sentiment tracking targetted towards politicians, movies, products, companies, identifying the emotion behind a newspaper headline etc. With the rapid proliferation of text based information processing and a number of social-media websites, there has been a increasing amount of emotion analysis and information mining for researched on newly available datasets. However, historical works has only been focusing solely on detecting certain emotion ignoring questions such as ‘who feels the emotion? (Experiencer)’, 'towards whom the emotion is directed (Stimulus)?’, 'what provokes the emotion? (Cue)' and 'what is the cause of emotion? (Cause)'. This project has been targeted around the same task using a number of cues for identification of the stated spans. Have a look at the stated sentence below:

.. image:: _static/image/sentence_exmaple.jpg

Installing the package
=======================
If you are running a linux machine with pip installed, then you can install the EmotionStimuli package using pip.

.. code-block:: bash

  pip install .

In case if you have not installed pip or are facing trouble installing using git, you can install EmotionStimuli package using the provided setup.py

.. code-block:: bash

   python setup.py install

Using the package
=================

Once the package has been installed, you can call the package as follows for one particular emotion role. In this case, the role is "cause":

.. code-block:: python

  from emotion import HMM

  hmm = HMM("cause")

The model heavily relies on the structure of `Brown dataset`_ and so, we have also added the required respective classes for appropriate conversion which could be imported as follows:

.. code-block:: python

  from emotion.utils import Data

  # Using the Data class
  gne_all = Data(
      filename="data/dataset.json",
      roles=["experiencer", "target", "cue", "cause"],
      corpora=["gne"],
      splits=[0.8, 0.2],
  )

  gne_all.conv2brown()


The `Data` class can also be used to create dataset splits which could further be used to create various splits in the datasets for training, validation and testing.

One end to end training of a dataset is provided below for reference.

.. code-block:: python

  from emotion import Data, Dataset, HMM, Evaluation

  intended_corpus = "gne"
  intended_role = "experiencer"

  # Load data from input file.
  # Only load annotations for the emotion role "experiencer" from the "GNE" corpus.
  # Split the data into two subsets, containing 80% and 20% of the total instances respectively (randomly selected).
  gne_exp = Data(
      filename="data/dataset.json",
      roles=[intended_role],
      corpora=[intended_corpus],
      splits=[0.8, 0.2],
  )

  # Convert the loaded annoations to Brown-format.
  gne_exp.conv2brown()

  # Store the instances of a specific subset in the Dataset object.
  # Each instance is stored as an Instance object, featuring attributes for 
  # tokens as well as gold and predicted annotations.
  train_gne = Dataset(data=gne_exp, splt=0)
  test_gne = Dataset(data=gne_exp, splt=1)

  # Train the HMM model for the intended emotion role.
  model_exp_gne = HMM(intended_role)
  model_exp_gne.train(dataset=train_gne)

  # Predict the annotations for the intended role using the previously trained model.
  model_exp_gne.predictDataset(dataset=test_gne)

  # Evaluate the model and save the results.
  # A prediction evaluates only to a TP if the Jaccard score of the predicted and
  # the gold span is above the threshold of 0.8.
  # The parameter beta for calculating the f-score is set to 1.0.
  results = Evaluation(dataset=test, role=intended_role, threshold=0.8, beta=1.0)
  results.save_eval(eval_name="gne_exp", filename="output.json")

  # Additionally, save a detailed documentation of how
  # the evaluation was calculated.
  results.save_doc(filename="documentation.json")

.. _Brown dataset: https://en.wikipedia.org/wiki/Brown_Corpus