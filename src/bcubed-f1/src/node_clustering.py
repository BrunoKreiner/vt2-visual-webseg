from lxml import etree
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import os

def nodes_clustering(task, node_visible_only=False, path_to_mhtml=''):
    
    """
    Perform clustering of nodes based on the given task and visibility criteria.
    Args:
        task (object): The task object containing clustering information and paths.
        node_visible_only (bool, optional): If True, only consider visible nodes. Defaults to False.
        path_to_mhtml (str, optional): Path to the MHTML file for visibility checking. Defaults to ''.
    Returns:
        dict: A dictionary containing the clusters and membership information.
            - 'clusters' (list): A list of dictionaries with cluster information.
                - 'membership' (numpy.ndarray): Membership matrix indicating cluster intersections.
                - 'size_nodes' (int): Number of nodes in the cluster.
                - 'size_chars' (int): Number of characters in the cluster.
            - 'membership' (dict): Subsets of membership information for each annotator.
    """
    
    # Parse the DOM tree
    with open(f"{task.path}dom.html", 'r') as html_file:
        html = html_file.read()
    dom = etree.HTML(html)
    
    
    # get driver if node visible only
    if node_visible_only:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        path_to_mhtml = os.path.join(path_to_mhtml, f'{task.id}.mhtml')
        driver.get(f"file://{path_to_mhtml}") 
        node_is_not_visible = set()
    
    # get all nodes per 'cluster'
    res = {}
    text_nodes = {}
    
    for user in task.hyuclusters:
        res[user] = []
        clusters = task.hyuclusters[user]
        for cluster in clusters:
            node = dom.xpath(f"//*[@hyu='{cluster['hyuIndex']}']")
            node_hyu = int(cluster['hyuIndex'])
            # get text length of this node
            if len(node) == 0:
                print('------------------- ERROR hyu not found'+ str(node_hyu) +'  '+ str(task.id))
                continue
            if node[0].text and node[0].text.strip():
                text_nodes[node_hyu] = len(node[0].text.strip())
            
            # start hyu values with this node    
            # check start value for visibilty aswell (just to make sure, should be visible)
            if node_visible_only:
                selector = f"[hyu='{node_hyu}']"
                visible_selenium = is_element_visible_selenium(driver, selector, by=By.CSS_SELECTOR)
                if not visible_selenium:
                    hyu_values = set()
                else:
                    hyu_values = set([node_hyu])
            else:
                hyu_values = set([node_hyu])
            
            # get all children nodes
            for child in node[0].iterdescendants():
                hyu_child = child.get('hyu')
                if hyu_child:
                    # Check if node is visible, if type = visible 
                    if node_visible_only:
                        if int(hyu_child) in node_is_not_visible:
                            continue
                        selector = f"[hyu='{hyu_child}']"                        
                        visible_selenium = is_element_visible_selenium(driver, selector,  by=By.CSS_SELECTOR)
                        if not visible_selenium:
                            node_is_not_visible.add(int(hyu_child))
                            continue
                        
                    hyu_values.add(int(hyu_child))
                    # get text length of the children node
                    if child.text and child.text.strip():
                        text_nodes[int(hyu_child)] = len(child.text.strip())
  
            res[user].append(hyu_values)
            
    if node_visible_only:
        driver.quit()


    # Remove duplicate sets from unique_clusters
    unique_clusters = [cluster for user in res for cluster in res[user]]
    unique_clusters = list(set([frozenset(cluster) for cluster in unique_clusters]))

    split_clusters = split_and_list_clusters(unique_clusters.copy())
    
    # Build membership matrix
    membership = {}
    membership['matrix'] = np.zeros((len(split_clusters), len(unique_clusters)), dtype=int)

    for i, distinct_poly in enumerate(split_clusters):
        for j, original_poly in enumerate(unique_clusters):

            intersects = distinct_poly.intersection(original_poly)
            if len(intersects) > 0:
                membership['matrix'][i, j] = 1


    # Define subsets for each annotator
    membership_subsets = get_subsets(res, unique_clusters)  # Placeholder for subset logic
    clusterings = {'clusters': [], 
                   'membership': membership_subsets}
    
    for i, cluster in enumerate(split_clusters):
        info = {
            'membership': membership['matrix'][i],
            'size_nodes': len(cluster),
            'size_chars': get_size(cluster, text_nodes)
        }
        clusterings['clusters'].append(info)
    return clusterings
    

def get_size(cluster, text_nodes):
    size = 0
    for node in cluster:
        size += text_nodes.get(node, 0)
    return size
    
    
def get_subsets(res, unique_clusters):
    dict_segmens = {c:i for i, c in enumerate(unique_clusters)}
    subsets = []
    for user, clusters in res.items():
        subsets_annot = []
        for cluster in clusters:
            subsets_annot.append(dict_segmens[frozenset(cluster)])
        subsets.append(subsets_annot)
    return subsets


def split_and_list_clusters(clusters):
    """
    Splits overlapping clusters into non-overlapping clusters.

    :param clusters: List of clusters (each cluster is a set of numbers)
    :return: A list of non-overlapping clusters
    """
    non_overlapping_clusters = []

    while clusters:
        # Start with the first cluster and remove it
        current = clusters.pop(0)
        new_clusters = []

        for other in clusters:
            # Compute overlap and differences
            overlap = current & other
            diff_current = current - other
            diff_other = other - current

            # Update the current cluster with non-overlapping part
            if overlap:
                new_clusters.append(overlap)
            if diff_other:
                new_clusters.append(diff_other)
            current = diff_current

        # Add the non-overlapping part of the current cluster
        if current:
            non_overlapping_clusters.append(current)

        # Update the clusters list for the next iteration
        clusters = new_clusters

    return non_overlapping_clusters

def is_element_visible_selenium(driver, selector, by=By.CSS_SELECTOR):   
    """
    Check if an element is visible on the page using Selenium.
    Args:
        driver (WebDriver): The Selenium WebDriver instance.
        selector (str): The selector string to locate the element.
        by (By, optional): The method to locate the element (default is By.CSS_SELECTOR).
    Returns:
        bool: True if the element is visible, False otherwise.
    """
    
    try:
        element = driver.find_element(by, selector)
        visible = element.is_displayed()
    except:
        visible = False

    return visible