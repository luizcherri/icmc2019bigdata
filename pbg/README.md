## PBG (Propagation in Bipartite Graph)

**About**

This is an alternative Python implementation of PBG (Propagation in Bipartite Graph), used in Faleiros et al. (2016) [1]. PBG is a framework based on label propagation algorithm using the bipartite graph representation.

**Download**

- You can download the PBG software in http://www.alanvalejo.com.br/software?name=pbg

**Usage**

> Semi-supervised example

    $ python pbg-bnoc-semi-supervised.py
    $ python pbg-cstr-semi-supervised.py

> Unsupervised example in topic extraction

    $ python pbg-syskillwebert-unsupervised.py

**Install**

> Pip
    
    $ pip install -r requirements.txt

> Anaconda env

    $ conda env create -f environment.yml
    $ conda activate pbg

> Anaconda create

    $ conda create --name pbg python=3.7.2
    $ conda activate pbg
    $ conda install -c anaconda numpy
    $ conda install -c anaconda pandas
    $ conda install -c anaconda scipy
    $ conda install -c anaconda scikit-learn
    $ conda install -c anaconda pyyaml
    $ conda install -c conda-forge pypdf2
    $ conda install -c anaconda nltk 
    $ conda install -c anaconda unidecode 

**Known Bugs**

- Please contact the author for problems and bug report.

**Contact**

- Alan Valejo
- Ph.D. at University of São Paulo (USP), Brazil
- alanvalejo@gmail.com.br

**License and credits**

- The GNU General Public License v3.0
- Giving credit to the author by citing the papers [1]

**References**

> [1] Faleiros, Thiago De Paulo and Valejo, Alan and Lopes, A., Unsupervised learning of textual pattern based on propagation in bipartite graph, in Intelligent data analysis, accepted paper, 2019

~~~~~{.bib}
@article{faleiros2019unsupervised,
    author = {Faleiros, Thiago De Paulo and Valejo, Alan and Lopes, A.},
    title = {Unsupervised learning of textual pattern based on propagation in bipartite graph},
    journal = {Intelligent data analysis, accepted paper},
    year = {2019}
}
~~~~~

<div class="footer"> &copy; Copyright (C) 2016 Alan Valejo &lt;alanvalejo@gmail.com&gt; All rights reserved.</div>
