This notebook is meant to help you run the codebase:
-----------------------------------------------------

1. Although all of the processes might not require tensorflow, it is recommended that you keep tensorflow running in background:
In the command prompt, just type, "activate tensorflow"
2. Folder: Inferscent/encoder/ run "demo.ipynb". Make sure to not move the "inferscent.pkl" files out of this location.
3. To run the SquAD dataset results, go to this path : ~\code\SQuAD-master; and run the "model.py" file. The model has been pre-trained already, no need to pre-train it any further, as the results are not going to improve.
4. ~\code\bnp paribas data: Run the only ".py" file in this folder. This is run on the BNP Paribas Closed-context dataset for highly technical question-answering.
5. Please be sure to download the Glove and FastText embeddings from the internet. The files are huge, and I am not being able to upload the zipped folder.
	i. Link to Glove embeddings(glove.840B.300d) downloads page: https://www.kaggle.com/takuok/glove840b300dtxt
	ii. Link to Crawl/FastText embeddings(crawl-300d-2M.vec) downloads page: https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m
6. Be sure to place the embeddings jsons in the correct path,  or define the path in the code accordingly.
7. ~\code\InferSent-master\InferSent-master: Run "demo.py" to run an implementation of the Inferscent sentence embedding.
8. These are all the essential outputs. In case you want to see some other analyses, you could run the "demo.py" or "main.py" files in the appropriately named folders.