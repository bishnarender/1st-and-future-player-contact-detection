# google-ai4code-understand-code-in-python-notebooks
## 20th place score achieved.
![ai4code_submission](https://user-images.githubusercontent.com/49610834/235819407-27ed1b6b-286a-454c-9f6d-b04f17d36bc0.png)

-----

### Start 
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. preprocess.ipynb
3. train_type_1.ipynb
4. inference.ipynb

-----

### preprocess.ipynb
![data_processing](https://user-images.githubusercontent.com/49610834/235819585-2eee62fc-772d-47e9-9e79-8415d74d00ee.jpg)

-----

### train_type_1.ipynb
<b>deberta-v3-base has been used. It has only 86M backbone parameters.</b>

<b>No pre MLM has been used.</b>

![model](https://user-images.githubusercontent.com/49610834/235819991-709b6b47-d45b-4a10-9076-60983629c183.jpg)

<b>How i trained the model over kaggle notebook:</b>

First, train the model with lr=1849e-7, parameter.seq_length=512, batch_size=3, n_accumulate=5 and the whole fold data (approx 1.25 lakh data points). Train till the train_loss=0.02 is not achieved. Overall the theme is to train with the maximum possible batch_size at start. 

In the second round, train the model with lr=1849e-10, parameter.seq_length=2048, batch_size=2, n_accumulate=5 and the fold data (minimum 20K data points). Train till the best is not achieved.
