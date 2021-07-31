Using the package
==================

The package is targetted to be used for detailed emotional analysis. So here is a detailed use-case of the package:
Lets say we want analysis on a piece of text saying "I like ice cream because it reminds me of summer", we call use the following method to call the package and the weights and use the modules.

.. code-block:: python

  from emotion.application import EmotionRoleLabeller
  
  er = EmotionRoleLabeller()

Now that you have initialized the model, you can perform analysis using the `analyse` function to get detailed analysis of the text:

.. code-block:: python

  output = er.analyse("I like ice cream because it reminds me of summer")
  print(output)

The output will be printed as follows:

.. code-block:: bash

  {
    "text": "I like ice cream because it reminds me of summer",
    "emotion": "joy",
    "roles":
      {
        "cue": "like",
        "cause": "it reminds me of summer",
        "experiencer": "I",
        "target": "ice cream"
      }
  }
  
Training the modules
--------------------

The framework contains modules for training your own models too. In order to train your own models, please make changes to the config file and then install the package (for detailed instructions on installing the package, refer :ref:`Installing the package`). After you are done installing the package, you can import the modules and start training as follows:

.. code-block:: python

  from emotion.trainer import bilstm_trainer

Once the trainer is imported, it needs to be initialized with suitable location of the dataset and the dataset name. For our training, we called the dataset as follows:

.. code-block:: python

  train_obj = bilstm_trainer.Methods("path/to/dataset.json",["reman"], "target")

This initializes all the required files and file paths. After which, you can start trianing by calling the following line of code

.. code-block:: python

  train_obj.trainer()



Using the HMM Model
-------------------

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