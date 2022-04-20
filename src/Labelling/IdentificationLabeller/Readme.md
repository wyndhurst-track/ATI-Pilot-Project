### This is a python-based GUI to help manualling categorising images when there are lots of categories.

### Installation

Just clone this repository to your local directory

#### Dependencies/Requirements

python3, OpenCV, TK

#### Instructions on using the program:
run the program with: `python3 main.py --input_folder=/path/to/input/folder --output_folder=/path/to/output/folder`
dependencies: python 3.x, tk, numpy, opencv
* The query image is the first unlabelled image in the provided input folder.
* To create a new category, press the "Create new category" button. The current query image will be automatically put in this category.
* To associate the query image with a particular category, click on the corresponding button.
* A image larger preview of that category will be displayed below the query image to aide categorisation.
* If the preview image for that category isn't clear enough, use the arrow buttons below each category to cycle through other images for it.
* The keyboard arrow keys allow you to quickly navigate between the pages of possible categories.
* You can quit the program anytime you like, progress will be restored provided the input/output folders are left untouched.
* If you want to rename a category manually (you don't just want a number), quit the program, edit the corresponding folder name in the output directory and re-launch the program

Note that the program is optimised for a 1920x1080p display, if you need to resize the window, edit lines 39, 40 (`self.__max_cols=w` and `self.__max_rows=h`) according to your display size.

#### Notes on files
Each time a new category is created, a folder will be created in the given output directory.
Each time an image is categorised, it will be moved into the corresponding category folder in the output directory.
An example might look like this after a few images have been categorised:
```
/path/to/input/folder/:
  0006.jpg
  0007.jpg
  0008.jpg
  ...
/path/to/output/folder/:
  001/
    001.jpg
    003.jpg
  002/
    002.jpg
    005.jpg
  003/
    004.jpg
  ...
```

#### Bugs/crashses
Create an issue here with the steps to replication or email screenshots or terminal output of any bugs/crashes to: will.andrew@bristol.ac.uk.
