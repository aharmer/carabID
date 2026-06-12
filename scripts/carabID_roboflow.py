import os
import sys
from roboflow import Roboflow

ROBOFLOW_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_KEY:
    sys.exit("Error: ROBOFLOW_API_KEY environment variable not set.")

rf = Roboflow(api_key=ROBOFLOW_KEY)

HOME = "D:/Dropbox/data/carabID/"

### Download annotated imageset version from roboflow project ###
os.chdir(HOME + "imgs")
# project = rf.workspace("bugider").project("carabidae_extra-qxccn")
# version = project.version(1)
project = rf.workspace("rainna").project("carabids_genus_v3")
version = project.version(6)

dataset = version.download("yolov11")

  
