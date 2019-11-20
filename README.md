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


# Citation
If you are using this code please cite the following paper

@inproceedings{karimi-tang-2019-learning,
    title = "Learning Hierarchical Discourse-level Structure for Fake News Detection",
    author = "Karimi, Hamid  and
      Tang, Jiliang",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational   Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1347",
    doi = "10.18653/v1/N19-1347",
    pages = "3432--3442",
}
