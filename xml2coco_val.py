import os.path as osp
import xml.etree.ElementTree as ET
import mmcv

from glob import glob
from tqdm import tqdm
from PIL import Image
coco_classes = ["not sperm","sperm"]
label_ids = {name: i + 1 for i, name in enumerate(coco_classes)}



def get_segmentation(points):
  a=points[0]
  b=points[1]
  c=points[2] + points[0]
  d=points[1]

  e=points[2] + points[0]
  f=points[3] + points[1]
  g=points[0]
  h=points[3] + points[1]

  if a>640:
     a=640

  if b>640:
     b=640

  if c>640:
     c=640

  if d>640:
     d=640
     
  if e>640:
     e=640

  if f>640:
     f=640

  if g>640:
     g=640

  if h>640:
     h=640 
  
  

  return [a,b,c,d,e,f,g,h]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w*h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": [xmin, ymin, w, h],
                        "iscrowd": 0,                
                        })
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 0
    anno_id = 0
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)         
        img = {"license": 0,"url":None ,"file_name": img_name, "height": int(h), "width": int(w),"date_captured":None , "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = [{"supercategory": None, "id": 0, "name": "_background_"},{"supercategory": None, "id": 1, "name": "not sperm"},{"supercategory": None, "id": 2, "name": "sperm"}]
    #for k,v in label_ids.items():
        #categories.append({"name": k, "id": v})
    final_result = {
      "info": {"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-10-16 01:51:18.310299"}, 
      "licenses": [{"url": None, "id": 0, "name": None}],
      "images": images,"type": "instances", "annotations": annotations,"categories":categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    xml_path = '/content/mask_rcnn_train/dataset/val/xml' #xml
    print('xml_path: ', xml_path)
    # xml_path = '../../../Datasets/underwater/data0/18test/label'
    img_path = '/content/mask_rcnn_train/dataset/val/images' #img
    # img_path = '../../../Datasets/underwater/data0/18test/image'
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, '/content/mask_rcnn_train/dataset/val/annotations.json')  # json
    # cvt_annotations(img_path, xml_path, './data/coco/annotations/test.json')
    print('Done!')


if __name__ == '__main__':
    main()