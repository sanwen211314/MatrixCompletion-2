# MatrixCompletion
Contains scripts for performing and comparing matrix completion on Netflix Prize data.

The codes included here are derived and altered from already existing codes, cited below.

FILES:
Create_Data_Subset.py : This script reads the Netflix prize data set https://www.kaggle.com/netflix-inc/netflix-prize-data/data and forms an incomplete matrix of a given size.  See the header of this code for a larger description. 

Rank1MatrixCompletion.py : This code performs the rank 1 propagation algorithm outlined at http://www.highdimensionality.com/2016/04/24/a-different-approach-to-low-rank-matrix-completion/

Netflix_Rank1_Completion.py : This script takes in the incomplete matrix resulting from "Create_Data_Subset.py" and performs the rank 1 propagation algorithm.

SVT.m + SVD_Utils : The original SVT algorithm [svt.stanford.edu](http://svt.stanford.edu) written by Stephen Becker and made to be compatible with Octave 5.1.0.  To make this compatible with Octave, several speed-ups that are available with MATLAB were removed. 

Netflix_SVT.m : A script that runs the SVT algorithm on a subset of the Netflix Prize data.

DMF_MC.m - The algorithm written for "Matrix completion by deep matrix factorization" Jicong Fan, Jieyu Cheng. Neural Networks, 2018(98):34-41 adjusted for compatibility with Octave 5.1.0.

DMF_NetflixPrize.m - This script utilizes "DMF_MC.m" to perform matrix completion on the incomplete matrices produced by "Create_Data_Subset.py"

Several other supporting Octave files are included such as tan_opt.m, sigm.m, rproptools, +Utils, etc. which are necessary for the DMF_MC.m code
