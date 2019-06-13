# HDSF
This is the implementation of  Learning Hierarchical Discourse-level Structure for Fake News Detection paper 
http://cse.msu.edu/~karimiha/publications/NAACL2019Discourse.pdf

# Required packages:
    pytorch
    sklearn
    numpy
    pandas
    pickle
    gensim

# Intsrutions to run the project
1. Please download  HDSF folder from https://www.dropbox.com/s/shwgf52qlqwoo1n/HDSF.7z?dl=0 (it includes the data, splits, etc)
2. Unzip the file and copy it in an arbitrary location, say /PATH/ 
3. Download word embeddings from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and copy it to /PATH/HDSF
4. To train, run 'python train.py --project_dir /PATH/HDSF/' (hyperparameters are included in config.py)
5. To test, 'python test.py --project_dir /PATH/HDSF/' 


