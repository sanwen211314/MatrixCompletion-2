import Rank1MatrixCompletion as mc
import numpy as np
import scipy.io
import scipy.linalg
import sys

# Read in data
X_loaded = scipy.io.loadmat('X_1000by1000.mat')
X = X_loaded['X'] #[0:10,0:10] # For slicing a subset for debugging
X0 = X
(n, m) = X.shape
X_recovered = np.zeros((n,m))

# Run this 100 times, record the error each run.
errrs = []
for N in range(100):
	# Create Mask
	sampleRate = 0.8  # Samplerate is the percentage of data still available. 
	M = np.less(np.random.rand(n,m), sampleRate)
	M_test = np.ones((n,m)) - M*1
	
	# Apply Mask
	X_Training = np.multiply(X,M*1)
	X_Test = np.multiply(X,M_test)
	
	# Perform Rank 1 Recovery
	[X_recovered, M_recovered] = mc.completeRank1Matrix(np.copy(X_Training), np.copy(M), False)
	np.nan_to_num(X_recovered, copy=False)
	# Measure Performance between entries of X_recovered and X_test.
	# This is done by using the test_mask on X_recovered and measuring the residual bwith X_test. 

	if(scipy.linalg.norm(X_Test, ord='fro') != 0):
		recovery_error = scipy.linalg.norm( (np.multiply(X_recovered, M_test) - X_Test), ord='fro')/scipy.linalg.norm(X_Test, ord='fro')
	else:
		continue

	print("Unadjusted Error:", recovery_error)

	## Basically if no care at all is taken, estimated ratings greater than 5 will arise.
	for i in range(n):
		for j in range(m):
			X_recovered[i][j] = min([X_recovered[i][j],5])
			X_recovered[i][j] = max([X_recovered[i][j],1])

	if(scipy.linalg.norm(X_Test, ord='fro') != 0):
		recovery_error2 = scipy.linalg.norm( (np.multiply(X_recovered, M_test) - X_Test), ord='fro')/scipy.linalg.norm(X_Test, ord='fro')
	else:
		continue

	print("Adjusted Error:", recovery_error2)
	errrs.append(recovery_error2)

# np.savetxt("errors_rank1_1000by100.csv", np.asarray(errrs), delimiter = ",")

# np.set_printoptions(threshold=sys.maxsize)
# print(X_Test)
# print(np.multiply(X_recovered, M_test))