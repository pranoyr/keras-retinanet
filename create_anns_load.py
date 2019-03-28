import xml.etree.cElementTree as ET

class Annotations:

    def __init__(self, filename):
        self.root = ET.parse(filename).getroot()

    def create_filename(self,file_name):
        # folder
        ET.SubElement(self.root, "filename").text = file_name

    def create_img_size(self,w,h,d):
        # size
        size=ET.SubElement(self.root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(d)

    def create_object(self,name,xmin,ymin,xmax,ymax):
        # object
        obj=ET.SubElement(self.root, "object")
        ET.SubElement(obj, "name").text = name
        bndbox=ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "truncated").text = "0"

    def create_xml(self,path):
        tree = ET.ElementTree(self.root)
        tree.write(path)