# TODO Remember that for large datasets multi_dot might be best. Should test case to case.
# For most of the code we have commented the lines

#TODO If all you want is the likelihood there should be a simplified version of the KF
# With an overhead function and a parameter returnLikelihood = True

# TODO Steps:
# Create a KF that works - Check KF1
# Create a KF for nulls - Check Kf1
# Create a univariate KF that works - Check KF2
# Create a univariate KF that works for nulls - Check KF2

# Separate the KF for likelihood and for getting the states (or not ?)

# Cythonize