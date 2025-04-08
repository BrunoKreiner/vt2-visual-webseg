# Extended BCubed F1-score for Web Page Segmentation

This project is a modified version of the [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset). The original code has been adapted to Python and tailored for specific use.

## Changes from [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset)
The extended BCubed F1-score calculation in [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset) takes in only bounding Boxes and matches these to the HTML nodes by their bounding Boxes. 

This project takes as input for each segment: 
- annotations as polygons (same as previous)
- annotations as hyu values (hyu values are specific for each node, and added by using this [content extraction software](https://github.com/JasminSaxer/content-extraction-framework).)

Thus the BCubed F1-score which are pixel-based (pixel, edges_fine, edges_coarse) are calculated using the polygons, 
and the node-based (node, characters) are calculated using the hyu values (speific nodes).

## Description

- main.py: The main script that processes the folders in the data directory, evaluates the segmentations, and prints the results.
- src: Contains the source code for various functionalities:
    - bcubed.py: Implements the B-Cubed evaluation metrics.
    - canny.py: Contains functions for applying Canny edge detection.
    - classes.py: Defines the main classes used in the project, such as Task, Segmentation, and Segmentations.
    - node_clustering.py: Implements node-based clustering.
    - pixel_clustering.py: Implements pixel-based clustering.

The evaluation results are saved in bcubed_res.csv files within each test data folder. The results include B-Cubed F1 and Max scores for each page.

## Installation

Install the required dependencies (using python version 3.11.0):
```bash
pip install -r requirements.txt
```

Install imagemagick for extracting canny edges (using version: 6.9.11-60).

For Ubuntu:
```bash
sudo apt install imagemagick
```


## Usage


```
usage: main.py [-h] --folder_path FOLDER_PATH [--file_postfix FILE_POSTFIX] --operation {prediction,pairwise_agreement} [--path_to_mhtml PATH_TO_MHTML]
               [--per_class] [--measure {F1,MaxPR,Precision,Recall}] [--use_pool] [--processes PROCESSES] [--verbose] [--pixel_based] [--node_based]
               [--add_visible_nodes]

        Calculates the extended B-Cubed score for Web Page Segmentation in Python, adapted from the R code available at:
        https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset
        
        Example usage for pairwise agreement:
        python main.py --folder_path test_data/test_data_pairwise --file_postfix 'FC' --operation 'pairwise_agreement' --pixel_based --node_based
    
        Example usage for prediction:
        python main.py --folder_path test_data/test_data_prediction --file_postfix 'FC' --operation 'prediction'        
    

options:
  -h, --help            show this help message and exit
  --folder_path FOLDER_PATH
                        The path to the folder containing the data
  --file_postfix FILE_POSTFIX
                        The name where files are stored : annotations_{file_postfix}.json (default: )
  --operation {prediction,pairwise_agreement}
                        The operation to perform (prediction or pairwise_agreement)
  --path_to_mhtml PATH_TO_MHTML
                        The path to the folder containing the mhtml files. Needed only for getting visual only nodes.
  --per_class           Calculate score per class
  --measure {F1,MaxPR,Precision,Recall}
                        The measure to use for per class calculation (default: F1). Can use F1, MaxPR and Recall or Precision for predicition only.
  --use_pool            Use multiprocessing pool to process folders
  --processes PROCESSES
                        Number of processes to use in the pool (default: 4)
  --verbose             Increase output verbosity
  --pixel_based         Use pixel based atomic element
  --node_based          Use node based atomic element
  --add_visible_nodes   Add measuring only visible nodes.
```

For example: 
```bash
python main.py --folder_path test_data/test_data_pairwise --file_postfix 'FC' --operation 'pairwise_agreement' --pixel_based --node_based
python main.py --folder_path test_data/test_data_prediction --file_postfix 'FC' --operation 'prediction'   
```

## Data Structure

The data directory should have the following structure:

```
data/
├── page1/
│   ├── annotations_{file_postfix}.json
│   ├── dom.html
│   └── screenshot.png
├── page2/
│   ├── annotations_{file_postfix}.json
│   ├── dom.html
│   └── screenshot.png
└── ...

path/to/mhtml/
├── page1.mhtml
├── page2.mhtml
└── ...

```

Each page should have its own directory containing:
- `annotations_{file_postfix}.json`: Contains the segmentation annotations for the page with the specific file postfix.
- `dom.html`: The HTML DOM of the web page (with hyu Indexes), extracted from the MHTML generated using [content extraction software](https://github.com/JasminSaxer/content-extraction-framework).
- `screenshot.png`: A screenshot of the web page for visual reference.
- `path/to/mhtml/`: Containing the paths to MHTML only needed for visible nodes only calculations.

### annotations.json

The `annotations.json` file contains the segmentation annotations for a specific web page. It includes the following fields:

- `id`: A unique identifier for the web page.
- `height`: The height of the web page screenshot.
- `width`: The width of the web page screenshot.
- `segmentations`: A dictionary where each key is an annotator's identifier and the value is a list of segmentation coordinates. Each segmentation is represented as a list of polygons, where each polygon is a list of points (x, y coordinates) or as shapely polygons dicts. The tagType is optional, and only needed for per_class calculations.
- `nodes`: A dictionary where each key is an annotator's identifier and the value is a dictionary with 'hyuIndex': index value, corresponding to the dom.html.

Example structure:

Polygon as nested list.
```json
{
  "id": "66ffde79306dfe2088fd9ff7",
  "height": 3716,
  "width": 2560,
  "segmentations": {
    "annotator1": [
      {
        "polygon": [
          [
            [
              [480,0],
              [2080,0],
              [2080,221],
              [480,221],
              [480,0]
            ]
          ]
        ],
        "tagType": "header" 
      },
      ...
    ] 
  },
  "nodes": {
    "annotator1": [
      {
        "tagType": "header",
        "hyuIndex": 994
      },
      ...
    ]
  }
}

```

Polygon as shapely polygons dicts (without classes (tagTypes)):
```json
{
  "id": "66ffde79306dfe2088fd9ff7",
  "height": 3716,
  "width": 2560,
  "segmentations": {
    "annotator1":[
      {
        "polygon": 
        {
          "type": "Polygon", 
          "coordinates": 
          [
            [
              [480,0],
              [2080,0],
              [2080,221],
              [480,221],
              [480,0]
            ]
          ] 
          }},
      ...
    ] 
  },
  "nodes": {
    "annotator1": [
      {
        "hyuIndex": 994
      },
      ...
    ]
  }
}

```

## License

This project is licensed under the terms of the original repository.

## Acknowledgments

- The original code are from the [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset).