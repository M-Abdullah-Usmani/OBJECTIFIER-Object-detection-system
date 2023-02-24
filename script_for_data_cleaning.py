import xml.etree.ElementTree as ET
import os
path = r"C:/Users/abdul/OneDrive/Desktop/VOC2011/Annotations"
directories = os.listdir( path )
for file in directories:
    tree = ET.parse(str(file))
    root = tree.getroot()
    for i in range(len(root)):
        for state in root[i].findall('part'):
            root[i].remove(state)
    tree.write(str(file))