### Thomas Merkh, merkh.thomas@gmail.com
### 02.05.2020

"""
This code parses through the Netflix Prize dataset and creates a rank matrix of the following format:

Each row is indexed by a unique movie
Each column corresponds to a unique customer
Each entry is label 1-5 is how much each customer liked a given movie, and is labeled 0 if no rating.

First, a number of movies is fixed (lets say 50, meaning only data for the first 50 movies is considered). 
Second, a number of customers is fixed (lets say 60).
Then the code decides which 60 customers were "most active" in watching the first 50 movies, and forgets the rest. 

This is done by creating a dictionary whose keys are all of the uniqe customer ID numbers which appear in the first 50 movies, and the corresponding value is 
the number of movies that each customer watched, out of the first 50. 

Then a list of the most active customers is made from this dictionary.

Then loop over each of the movies, loop over each of the most active customers.  See if the given customer has watched the given movie.
If so, add their ranking to the matrix X.  If not, leave the value in X as a zero. 

################################################################

# The Netflix prize data is stored as

# 1:
# 1488844,3,2005-09-06
# 822109,5,2005-05-13
# 885013,4,2005-10-19
# 30878,4,2005-12-26
# 823519,3,2004-05-03
# 893988,3,2005-11-17
# 124105,4,2004-08-05
# 1248029,3,2004-04-22
# 1842128,4,2004-05-09
# 2238063,3,2005-05-11
# 1503895,4,2005-05-19
# 2207774,5,2005-06-06
...

The 1: line indicates that this is a movie ID.  Then all of the lines following it are customer IDs, rankings, and dates on which the ranking was given. 

################################################################
Strategy:
Read each line, if that line ends in a ":", then it is a new row of the matrix. 
If the line does not end in a ":", then split it using deliminer ","
The first element might be an index of the matrix.  If it is, input the value.
"""
import sys, os
import numpy as np
import scipy.io

############ User chosen parameters, how large should X be? ############
N_movies = 1000  # The movies only go up to 4499 in the first data_file "combined_data_1.txt"
N_cust = 1000
verbose = True
############ ############ ############ ############ ############ #######

X = np.zeros((N_movies, N_cust)) 
counter = 0
mylines = []

if(verbose):
    print("Parsing the data file...")                                                       
with open('combined_data_1.txt', 'rt') as myfile:      
    for line in myfile:
        line = line.strip()                            # Get rid of /n's at end of lines.  
        if(line[-1] == ":"):                           # If the line is the MovieID ending in a colon, add a list
            mylines.append([])
            movieID = int(line[:len(line)-1]) - 1      # This will be used as an index, so subtract 1
        else:
            mylines[movieID].append(line.split(',')[:2])  # Split each line at the commas and drop the dates, add to list
        
        if(movieID+1 == N_movies):
            break

# Now mylines is a list of lists.  Each inner list is a seperate movie and all of its ratings.  
# Now lets loop through each list, record the customers who have ranked movies, and count how many rankings they have given for the entire set of movies being considered. 

if(verbose):
    print("Counting how many movies each unique customer watched...")
customer_ledger = {} # empty dictionary,  Keys = CustomerIDs and Values = # of movies that customer ranked. 
for i in range(N_movies-1):
    for j in range(len(mylines[i])):
        cust_ID = mylines[i][j][0]
        if(cust_ID in customer_ledger.keys()):
            customer_ledger[cust_ID] += 1
        else:
            customer_ledger[cust_ID] = 1

# Now pick out the most active customers according to this dictionary
if(verbose):
    print("Determining the most active customers... This may take a few minutes")
Most_Active = []
Most_Active_IDs = []
for i in range(N_cust):
    MaxDictVal = max(customer_ledger, key=customer_ledger.get)
    MoviesWatched = max(customer_ledger.values())
    Most_Active.append([MaxDictVal, MoviesWatched])
    Most_Active_IDs.append(MaxDictVal)
    customer_ledger.pop(MaxDictVal)

if(verbose):
    print("The most active customers:")
    #print(Most_Active)
    print("Their specific customer ID numbers:")
    #print(Most_Active_IDs)

    print("Constructing X...  This may take several minutes") # This is the bottle neck step in the process

# Now form X, where each customerID is a column index, and each movie ID is a row index. 
# Loop over each movie first
for movieID in range(N_movies):
    cust_for_this_movie = [k[0] for k in mylines[movieID]] # Gathers the customer ID's for movie == movieID.

    # Loop over the relevant customers
    for i in range(N_cust):
        if(Most_Active_IDs[i] in cust_for_this_movie): 
            # find the data (list) in the list mylines[movieID] with first entry == Most_Active_IDs[i], this will contain the customer's ranking for the given movie 
            Customers_Ranking = [q for q in mylines[movieID] if q[0].startswith(Most_Active_IDs[i])]
            X[movieID][i] = Customers_Ranking[0][1]  # Input their ranking, which is stored in the 2nd spot of this list. 

if(verbose):
    #print(X)
    print("The number of nonzero entries:", np.sum((X > 0)*1))
    scipy.io.savemat('X_?by?.mat', {'X': X})
print("Program Finished, X.mat file saved.")