# Code for 2018 Qualifying Exam

This repository contains the code used for my 2018 Qualifying exam, which used data from the 2018 Zillow prize hosted by Kaggle.  Notebooks Part 1-3 focus on data analysis and visualization and Notebook Part 4 performs on modeling using LightGBM.

To run this code, first download and extract the Zillow dataset from the url below:

`https://www.kaggle.com/c/zillow-prize-1/data`

Next, install packages in the requirements.txt file with
`pip install -r requirements.txt`

The mapping package is matplotlib's basemap, which I do not believe is avalible on pip, but is available from the following url below.

`https://matplotlib.org/basemap/`

It is used in a single map in part 2, but if basemap is not present, the import error will be caught and the map will simply not be shown:

Finally, every notebook has in its first cell a 'data_dir' variable that needs to be set to the directory with extracted Zillow Prize data.
