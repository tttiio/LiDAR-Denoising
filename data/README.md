## 数据集

- SemanticSTF(过拟合验证)

  - .lable 分类信息

    0: "unlabeled",1: "car",2: "bicycle",3: "motorcycle",4: "truck",5: "other-vehicle",6: "person",7: "bicyclist",8: "motorcyclist",9: "road",10: "parking",11: "sidewalk",12: "other-ground",13: "building",14: "fence",15: "vegetation",16: "trunk",17: "terrain",18: "pole",19: "traffic-sign",20: "invalid"

  - .bin 点云信息

    五个维度信息，[x,y,z,r,i]

  - 点云几何位置可视化：

  ![image-20251202204704383](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20251202204704383.png)![image-20251202204809194](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20251202204809194.png)

- 能不能根据强度信息把噪声点恢复到正确的位置？