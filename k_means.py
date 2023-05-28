import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
class Kmeans:
   def __init__(self, img) -> None:
      self.img=img
      pass
   def process(self,K=2):
      img = self.img
      data = img.reshape((-1,3))
      data = np.float32(data)

      #定义中心 (type,max_iter,epsilon)
      criteria = (cv2.TERM_CRITERIA_EPS +
                  cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

      #设置标签
      flags = cv2.KMEANS_RANDOM_CENTERS

      compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)

      #图像转换回uint8二维类型
      centers = np.uint8(centers)
      res = centers[labels.flatten()]
      dst = res.reshape((img.shape))
      #图像转换为RGB显示
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
      return labels, centers
#读取原始图像
if __name__ =="__main__":
   # img = cv2.imread('data/FW_2_10mA.tiff')[200:310,300:400]
   img = cv2.imread('data/device1/0mA_3.tiff')[150:450,150:430]
   # img = cv2.imread('data/FW_5_0mA.tiff')[185:285,250:355]
   #图像二维像素转换为一维
   data = img.reshape((-1,3))
   data = np.float32(data)

   #定义中心 (type,max_iter,epsilon)
   criteria = (cv2.TERM_CRITERIA_EPS +
               cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

   #设置标签
   flags = cv2.KMEANS_RANDOM_CENTERS

   #K-Means聚类 聚集成2类
   compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

   #K-Means聚类 聚集成4类
   compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

   #K-Means聚类 聚集成8类
   compactness, labels6, centers6 = cv2.kmeans(data, 6, None, criteria, 10, flags)

   #图像转换回uint8二维类型
   centers2 = np.uint8(centers2)
   res = centers2[labels2.flatten()]
   dst2 = res.reshape((img.shape))

   centers4 = np.uint8(centers4)
   res = centers4[labels4.flatten()]
   dst4 = res.reshape((img.shape))

   centers6 = np.uint8(centers6)
   res = centers6[labels6.flatten()]
   dst6 = res.reshape((img.shape))



   #图像转换为RGB显示
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
   dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
   dst6 = cv2.cvtColor(dst6, cv2.COLOR_BGR2RGB)

   #显示图像
   titles = [u'Original', u'K-means, K=2', u'K-means, K=4',
            u'K-means, K=6']  

   images = [img, dst2, dst4, dst6]  

   # for i in range(2):  
      # plt.title('K-means Clustering', fontsize=20) 
   plt.figure(figsize=(50,25))


   fig = plt.subplot(1,2,1)
   plt.imshow(images[0], 'gray')
   plt.title(titles[0],fontsize=60,pad=30)  
   plt.xticks([]),plt.yticks([]) 


   fig = plt.subplot(1,2,2)
   plt.imshow(images[1], 'gray')
   plt.title(titles[1],fontsize=60,pad=30)  
   plt.xticks([]),plt.yticks([]) 
   colors = centers2/255.0
   labels = ['Class 1', 'Class 2',]
   patches = [mpatches.Patch(color=color[::-1], label=label) for color, label in zip(colors, labels)]
   plt.legend(handles=patches, prop={'size': 20})
   # plt.show()
   plt.savefig('1.png')
   # plt.cla()

   plt.figure(figsize=(50,25))
   fig = plt.subplot(1,2,1)
   plt.imshow(images[2], 'gray')
   plt.title(titles[2],fontsize=60,pad=30)  
   plt.xticks([]),plt.yticks([]) 
   colors = centers4/255.0
   labels = ['Class 1', 'Class 2','Class 3', 'Class 4',]
   patches = [mpatches.Patch(color=color[::-1], label=label) for color, label in zip(colors, labels)]
   plt.legend(handles=patches, prop={'size': 20})

   fig = plt.subplot(1,2,2)
   plt.imshow(images[3], 'gray')
   plt.title(titles[3],fontsize=60,pad=30)  
   plt.xticks([]),plt.yticks([]) 
   colors = centers6/255.0
   labels = ['Class 1', 'Class 2','Class 3', 'Class 4','Class 5', 'Class 6',]
   patches = [mpatches.Patch(color=color[::-1], label=label) for color, label in zip(colors, labels)]
   plt.legend(handles=patches, prop={'size': 20})
   # plt.show()
   plt.savefig('2.png')