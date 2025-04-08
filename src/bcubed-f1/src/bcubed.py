import numpy as np
from pprint import pprint

class Clustering:
    def __init__(self, clusters, membership_subsets, path):
        self.clusters = clusters
        self.membership_subsets = membership_subsets
        self.path = path

def AgreementMatrices(clustering):
    """
    Generate agreement matrices for each membership subset in the clustering.

    Parameters:
    clustering (Clustering): The clustering object containing clusters and membership subsets.

    Returns:
    list: A list of agreement matrices for each membership subset.
    """
    agreement_matrices = [AgreementMatrix(clustering, membership_subset) for membership_subset in clustering.membership_subsets]
    
    return agreement_matrices

def AgreementMatrix(clustering, membership_subset):
    """
    Compute the agreement matrix for a given membership subset in the clustering.

    Parameters:
    clustering (Clustering): The clustering object containing clusters and membership subsets.
    membership_subset (str): The membership subset to compute the agreement matrix for.

    Returns:
    np.ndarray: A symmetric matrix representing the agreement between clusters.
    """
    def GetAgreement(cluster_indices):
        cluster1_membership = clustering.clusters[cluster_indices[0]]['membership'][membership_subset]
        cluster2_membership = clustering.clusters[cluster_indices[1]]['membership'][membership_subset]
        return np.sum(cluster1_membership & cluster2_membership)

    num_clusters = len(clustering.clusters)
    common_segment_counts = np.full((num_clusters, num_clusters), np.nan)

    # Fill lower triangle
    for i in range(num_clusters):
        for j in range(i + 1):
            common_segment_counts[i, j] = GetAgreement([i, j])

    # Make symmetric
    upper_triangle_indices = np.triu_indices(num_clusters, k=1)
    common_segment_counts[upper_triangle_indices] = common_segment_counts.T[upper_triangle_indices]
    
    return common_segment_counts

def BCubedPrecisionMatrix(clustering, size_name, agreement_matrices=None):
    """
    Compute the BCubed precision matrix for the clustering.

    Parameters:
    clustering (Clustering): The clustering object containing clusters and membership subsets.
    size_name (str): The name of the size attribute in the clusters.
    agreement_matrices (list, optional): Precomputed agreement matrices. If None, they will be computed.

    Returns:
    np.ndarray: The BCubed precision matrix.
    """
    if agreement_matrices is None:
        agreement_matrices = AgreementMatrices(clustering)

    cluster_sizes = np.array([cluster[size_name] for cluster in clustering.clusters])
    

    num_clusters = len(clustering.clusters)

    def MultiplicityPrecision(l1, l2, agreement_matrix_algorithm, agreement_matrix_truth):
        """
        Calculate the multiplicity precision for given labels and agreement matrices.

        Multiplicity precision is defined as the ratio of the minimum number of segments
        between the algorithm's and the truth's agreement matrices to the number of segments
        in the algorithm's agreement matrix.

        Parameters:
        l1 (int): The first label.
        l2 (int): The second label.
        agreement_matrix_algorithm (numpy.ndarray): The agreement matrix for the algorithm.
        agreement_matrix_truth (numpy.ndarray): The agreement matrix for the ground truth.

        Returns:
        float: The multiplicity precision value. Returns NaN if the number of segments in
               the algorithm's agreement matrix is zero.
        """
        num_segments_algorithm = agreement_matrix_algorithm[l1, l2]
        num_segments_truth = agreement_matrix_truth[l1, l2]
        if num_segments_algorithm == 0:
            return np.nan
        else:
            res = min(num_segments_algorithm, num_segments_truth) / num_segments_algorithm            
            return res

    def AverageMultiplicityPrecision(l1, agreement_matrix_algorithm, agreement_matrix_truth):
        """
        Calculate the average multiplicity precision for a given cluster.
        Parameters:
        l1 (int): The cluster label for which the average multiplicity precision is calculated.
        agreement_matrix_algorithm (np.ndarray): The agreement matrix for the algorithm's clustering.
        agreement_matrix_truth (np.ndarray): The agreement matrix for the ground truth clustering.
        Returns:
        float: The average multiplicity precision for the given cluster label. Returns NaN if the total is zero.
        """
        
        multiplicity_precision = np.array([MultiplicityPrecision(l1, l2, agreement_matrix_algorithm, agreement_matrix_truth) for l2 in range(num_clusters)])
        common = ~np.isnan(multiplicity_precision)
        total = np.sum(cluster_sizes[common])
        if total == 0:
            return np.nan
        else:
            return np.sum(multiplicity_precision[common] * cluster_sizes[common]) / total

    def BCubedPrecision(agreement_matrix_algorithm_index, agreement_matrix_truth_index):
        """
        Calculate the B-Cubed Precision for a given clustering algorithm.
        B-Cubed Precision is a metric used to evaluate the precision of a clustering algorithm
        by comparing the algorithm's clustering results with the ground truth clustering.
        Parameters:
            agreement_matrix_algorithm_index (int): Index of the agreement matrix for the clustering algorithm.
            agreement_matrix_truth_index (int): Index of the agreement matrix for the ground truth clustering.
        Returns:
            float: The B-Cubed Precision value, ranging from 0 to 1, where 1 indicates perfect precision.
        """
        
        if agreement_matrix_algorithm_index == agreement_matrix_truth_index:
            return 1.0

        average_multiplicity_precision = np.array([AverageMultiplicityPrecision(l1, agreement_matrices[agreement_matrix_algorithm_index], agreement_matrices[agreement_matrix_truth_index]) for l1 in range(num_clusters)])
        common = ~np.isnan(average_multiplicity_precision)
        total = np.sum(cluster_sizes[common])
        if total == 0:
            return 0
        bcubed_precision = np.sum(average_multiplicity_precision[common] * cluster_sizes[common]) / total
        return bcubed_precision

    num_matrices = len(agreement_matrices)
    bcubed_precision_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in range(num_matrices):
        for j in range(num_matrices):
            bcubed_precision_matrix[i, j] = BCubedPrecision(i, j)

    return bcubed_precision_matrix


def CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, combination_function):
    """
    Combine BCubed precision and recall values using a specified combination function.
    This function takes a BCubed precision matrix and applies a combination function
    to each pair of precision and recall values in the lower triangle of the matrix.
    The resulting combined values are stored in a new matrix, which is then made
    symmetric by copying the lower triangle to the upper triangle.
    
    Parameters:
    bcubed_precision_matrix (numpy.ndarray): A square matrix containing BCubed precision values.
    combination_function (callable): A function that takes two arguments (precision, recall)
                                     and returns a combined value.
    Returns:
    numpy.ndarray: A symmetric matrix containing the combined BCubed precision and recall values.
    """
    
    
    # Create a copy of the precision matrix to store the combined values
    bcubed_combined_matrix = bcubed_precision_matrix.copy()
    
    # Get the lower triangle of the matrix (excluding the diagonal)
    lower_triangle_indices = np.tril_indices_from(bcubed_combined_matrix, k=-1)
    
    # Apply the combination function to each pair (precision, recall)
    for i, j in zip(*lower_triangle_indices):
        recall = bcubed_combined_matrix[i, j]
        precision = bcubed_combined_matrix[j, i]
        bcubed_combined_matrix[i, j] = combination_function(precision, recall)
    
    # Make the matrix symmetric by copying the lower triangle to the upper triangle
    upper_triangle_indices = np.triu_indices_from(bcubed_combined_matrix, k=1)
    bcubed_combined_matrix[upper_triangle_indices] = bcubed_combined_matrix.T[upper_triangle_indices]
    
    return bcubed_combined_matrix


def BCubedF1Matrix(bcubed_precision_matrix):
    """
    Compute the BCubed F1 matrix from the given BCubed precision matrix.
    This function takes a BCubed precision matrix and combines it with recall
    to compute the F1 score using the CombineBCubedPrecisionAndRecall function.
    Parameters:
        bcubed_precision_matrix (np.ndarray): The BCubed precision matrix, which should be a subclass of numpy ndarray.
    Returns:
        np.ndarray: The resulting BCubed F1 matrix.
    Raises:
        TypeError: If the input is not an instance of np.ndarray.
    """
    
    # Check if input is a BCubedPrecisionMatrix (a subclass of matrix)
    if not isinstance(bcubed_precision_matrix, np.ndarray):
        raise TypeError(f"Method requires a BCubedPrecisionMatrix, but got a {type(bcubed_precision_matrix)}")

    # Combine precision and recall to compute F1 using the given CombineBCubedPrecisionAndRecall function
    bcubed_f1_matrix = CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, F1)
    # If you need to add a specific attribute for "BCubedF1Matrix" class, you can do it here
    return bcubed_f1_matrix

def BCubedMaxMatrix(bcubed_precision_matrix):
    """
    Compute the BCubed maximum matrix by taking the higher value of either precision of recall.

    Parameters:
        bcubed_precision_matrix (np.ndarray): A BCubed precision matrix.
    Returns:
        np.ndarray: The resulting BCubed maximum matrix.
    Raises:
        TypeError: If the input is not a numpy ndarray.
    """
    
    # Check if input is a BCubedPrecisionMatrix
    if not isinstance(bcubed_precision_matrix, np.ndarray):
        raise TypeError(f"Method requires a BCubedPrecisionMatrix, but got a {type(bcubed_precision_matrix)}")

    # Combine precision and recall to compute the maximum
    bcubed_max_matrix = CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, np.maximum)
    # Add any necessary attributes if needed
    return bcubed_max_matrix

def EvaluationMatrixMean(evaluation_matrix):
    """
    Compute the mean of the lower triangular part of an evaluation matrix, excluding the diagonal.
    Parameters:
    evaluation_matrix (np.ndarray): The evaluation matrix to compute the mean from. 
                                    It should be a subclass of numpy ndarray.
    Returns:
    float: The mean of the lower triangular part of the matrix, excluding the diagonal.
           NaN values are ignored in the computation.
    Raises:
    TypeError: If the input is not an instance of numpy ndarray.
    """
    
    # Check if input is an EvaluationMatrix (a subclass of matrix)
    if not isinstance(evaluation_matrix, np.ndarray):
        raise TypeError(f"Method requires an EvaluationMatrix, but got a {type(evaluation_matrix)}")

    # Compute the mean of the lower triangular part of the matrix (excluding diagonal)
    lower_triangle = lower_triangle_with_nan(evaluation_matrix)  # lower triangle, excluding diagonal
    return  np.nanmean(lower_triangle)# Use nanmean to ignore NaNs if they are present

def F1(precision, recall):
    if not (precision and recall):
        return 0
    return 2 * precision * recall / (precision + recall)

def lower_triangle_with_nan(matrix):
    # Create a matrix of the same shape, filled with NaNs
    result_matrix = np.full_like(matrix, np.nan, dtype=float)
    # Set the lower triangle of the result matrix to be the lower triangle of the input matrix
    lower_triangle_indices = np.tril_indices_from(matrix, k=-1)  # k=0 to include the diagonal
    result_matrix[lower_triangle_indices] = matrix[lower_triangle_indices]
    return result_matrix

def evaluate_pairwise(clusterings, memberships, size_name, path, verbose=False):
    """
    Evaluate the clustering performance using BCubed metrics.
    Parameters:
        clusterings (list): A list of clustering results.
        memberships (list): A list of membership assignments.
        size_name (str): The name of the size metric to be used.
        path (str): The path to the data.
        verbose (bool, optional): If True, prints detailed information. Defaults to False.
    Returns:
        tuple: A tuple containing the mean F1 score and the mean Max score.
    """
    
    clustering = Clustering(clusterings, memberships, path)

    agreement_matrices = AgreementMatrices(clustering)
    # pprint(agreement_matrices)
    bcubed_precision_matrix = BCubedPrecisionMatrix(clustering, size_name, agreement_matrices)

    f1_mean = EvaluationMatrixMean(BCubedF1Matrix(bcubed_precision_matrix))
    max_mean = EvaluationMatrixMean(BCubedMaxMatrix(bcubed_precision_matrix))
    if verbose:
        print(size_name, 'BCubed Precision Matrix:')
        pprint(bcubed_precision_matrix)
        print('F1', f1_mean)
        print('Max', max_mean)
        
    return {'F1': f1_mean, 'MaxPR': max_mean}

def evaluate_prediction(clusterings, memberships, size_name, path, verbose=False):
    """
    Evaluate the prediction using BCubed metrics.
    Parameters:
        clusterings (list): A list of clusterings.
        memberships (list): A list of memberships corresponding to the clusterings.
        size_name (str): The name of the size metric.
        path (str): The path to the data.
        verbose (bool, optional): If True, prints detailed evaluation results. Defaults to False.
    Returns:
        tuple: A tuple containing F1 score, precision, and recall.
    """
    
    clustering = Clustering(clusterings, memberships, path)

    agreement_matrices = AgreementMatrices(clustering)
    bcubed_precision_matrix = BCubedPrecisionMatrix(clustering, size_name, agreement_matrices)

    precision = bcubed_precision_matrix[0][1]
    recall = bcubed_precision_matrix[1][0]
    
    f1 = F1(precision, recall)

    if verbose:
        print(size_name[5:], '\tF1:', round(f1,3), ', Precision:', round(precision,3), ', Recall:', round(recall,3))

    return {'F1': f1, 'Precision': precision, 'Recall': recall, 'MaxPR': max(precision, recall)}
