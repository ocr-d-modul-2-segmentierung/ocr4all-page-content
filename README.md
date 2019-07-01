# Page Content detection on historical handwritten documents

## Prerequisites
Dependencies for the project are located in requirement.txt.<br />
Major Dependencies are:
* Pixel classifier (CWick: https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation)
* numpy
* tensorflow
* scipy
* pillow
* opencv

## Installing

This projects requires the pixel classifier of
https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation
to be installed locally.
 * Clone the page segmentation repository `git clone https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation`
 * (Optional but recommended) Activate your virtual environment 
   (adapt to your path): `source venv/bin/activate`
 * install page segmentation `cd page-segmentation && python setup.py install`
 * the line detection is installed
 * install the specific requirements of page content `cd page-content && pip install -r requirements.txt`
