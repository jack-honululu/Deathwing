import xml.etree.ElementTree as ET
import numpy as np

"""
tag，即标签，用于标识该元素表示哪种数据，即APP_KEY
attrib，即属性，用Dictionary形式保存，即{'channel' = 'CSDN'}
text，文本字符串，可以用来存储一些数据，即hello123456789
"""

import os
import glob
i = 0
for filename in glob.glob(os.path.join('data/', '*.xml')):
    print(filename)
    # 1。先加载文档到内存里，形成一个倒桩的树结构

    tree=ET.parse(filename)
    #2.获取根节点
    root=tree.getroot()
    for ele in root:
        if 'rhetoricalClass' not in ele.attrib.keys():
            pass
        else:
            if i == 0:

                X = np.array(ele.text)
                y = np.array(ele.attrib['rhetoricalClass'])
            else:
                X = np.append(X,ele.text)
                y = np.append(y,ele.attrib['rhetoricalClass'])
            i+=1
            #print((ele.attrib['rhetoricalClass']), ele.text)
print(i,X.shape,y.shape)
data = np.concatenate((X[:,None],y[:,None]),axis = 1)
print (data.shape)
np.save('1_data',data)

