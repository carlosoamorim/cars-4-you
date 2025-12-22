
# For running the notebook without collab:

1) `pip install -r requirements.txt`

2) working directory MUST contain "train.csv" and "test.csv

project/
│
├── requirements.txt
├── train.csv
├── test.csv
├── your_notebook.ipynb
└── README.md



# If you Run the notebook in Google Colab (recommended):

1) Open in Colab
https://colab.research.google.com/github/rmmandrade/ml2526/blob/main/Group57.ipynb



2) Execution Requirements and Notes (Important)



This project was developed and executed in Google Colab due to its computational demands.



3) Hardware Requirements



Minimum RAM required: ~52 MB of available memory in Google Colab



Execution on local machines may fail or be significantly slower depending on available resources.



4) Required Input Files



To successfully run the full pipeline, the following datasets must be provided:



train.csv — loaded in Section 1 (Initial Setup and Data Loading)



test.csv — loaded in Section 8 (Final Model Evaluation and Inference)



requirements.txt



The datasets (train.csv and test.csv) are included in the repository and should be loaded by the pipeline

in Section 1 (training data) and Section 8 (test data), respectively.



5) Execution Guidance



The notebook is fully sequential and should be run top to bottom.



All preprocessing, feature engineering, model training, and inference steps are self-contained in the notebook.



The final section includes an interactive analytics interface that allows predictions for new input data, satisfying the project requirement:



Create an analytics interface that returns a prediction when new input data is provided.



6) Notes for Evaluation



The pipeline was executed end-to-end in Colab using the configuration above.



Any errors encountered when running locally are likely due to insufficient memory or missing input files.


