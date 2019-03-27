# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

if __name__ == '__main__':
    # 图像文件夹，对应的标注文件夹和输出文件夹
    image_dir = "image/"
    gt_dir = "gt/"
    cv2.namedWindow('display_rbndbox')

    gt_list = []
	img_list = []
    im_infos = []
    for label in os.listdir(gt_dir):
        gt_file = os.path.join(gt_dir, label)
        gt_list.append(label)
    for i in range(len(img_list)):
		#print os.path.join(DATASET_DIR, gt_list[i])
		gt_file = open(os.path.join(gt_dir, gt_list[i]))
		gt_content = gt_file.read()
		gt_li =  gt_content.split('\n')
		hard_boxes = []
		easy_boxes = []

		img = cv2.imread(os.path.join(DATASET_DIR, img_list[i]))









    # 定义变量i，用于确定输出图像的编号
    i = 1
    #########################################################
    # 解析每一个.xml标注文件，得到其中所有的旋转包围框信息：cx,cy,w,h,angle
    tree = ET.parse(annotation_dir)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.float32)
    for ix, obj in enumerate(objs):
        robndbox = obj.find('robndbox')
        # Make pixel indexes 0-based
        cx = float(robndbox.find('cx').text) - 1
        cy = float(robndbox.find('cy').text) - 1
        w = float(robndbox.find('w').text) - 1
        h = float(robndbox.find('h').text) - 1
        # 注意：利用标注工具得到的旋转包围框的angle信息有以下特征：
        # 1 单位是弧度，而下面的cv2.getRotationMatrix2D()中的旋转角是角度，故需做转换
        # 2 以水平位置为分界，当包围框框顺时针倾斜时，角度是正的，没问题；但是当包围框框
        #   逆时针倾斜时，正确的角度angle应该是负值，但是。.xml文件中的值=pi-|angle|，
        #   所以下面使用语句angle-=np.pi将其纠正。
        angle = float(robndbox.find('angle').text)
        if angle>1.57:
            angle-=np.pi
        angle = angle*180/np.pi
        boxes[ix, :] = [cx, cy, w, h, angle]
    ##############################################################
    # 根据.xml文件名，读取同名的集装箱原始图像
    img = cv2.imread(image_dir)
    # 1 要想求得透视变换矩阵，需要确定源图像和目标图像的对应的4个顶点坐标，
    #   并使用cv2.getPerspectiveTransform()求得透视变换矩阵M1
    # 2 目标图像的4个顶点坐标就使用旋转前的矩形框的4个顶点坐标p1-p4
    # 3 源图像的4个顶点坐标使用旋转后的矩形框的4个顶点坐标rp1-rp4
    # 4 再使用cv2.line()和cv2.imshow()将水平矫正后的图像进行可视化，按键盘任意键（除了esc）
    #   依次显示下一个矫正的旋转包围框图像，按esc键一次跳到下一张图像；
    # 5 截取矫正后的字符块为roi区域，将其保存。

    # 之前的透视变换的4组顶点事实上并不一一对应，导致透视变换的效果和旋转仿射变换一样，
    # 水平矫正后字符任然是倾斜的，事实上源图像的4个顶点构成的应该是一个平行四边形，这里
    # 为了演示，人为添加k值，将源图像中的矩形变成平行四边形。
    k = 25
    for j in range(num_objs):
        M = cv2.getRotationMatrix2D((boxes[j][0], boxes[j][1]), -boxes[j][-1], 1)
        ix = boxes[j][0]
        iy = boxes[j][1]
        iw = boxes[j][2]
        ih = boxes[j][3]
        left_top = np.array([ix-iw/2.0, iy-ih/2.0, 1])
        right_top = np.array([ix+iw/2.0-k, iy-ih/2.0, 1])
        left_bottom = np.array([ix-iw/2.0+k, iy+ih/2.0, 1])
        right_bottom = np.array([ix+iw/2.0, iy+ih/2.0, 1])
        rp1 = np.dot(M, left_top)
        rp2 = np.dot(M, right_top)
        rp3 = np.dot(M, left_bottom)
        rp4 = np.dot(M, right_bottom)
        cv2.line(img, (int(rp1[0]),int(rp1[1])), (int(rp2[0]),int(rp2[1])), (255,0,0), 1)
        cv2.line(img, (int(rp2[0]),int(rp2[1])), (int(rp4[0]),int(rp4[1])), (255,0,0), 1)
        cv2.line(img, (int(rp4[0]),int(rp4[1])), (int(rp3[0]),int(rp3[1])), (255,0,0), 1)
        cv2.line(img, (int(rp3[0]),int(rp3[1])), (int(rp1[0]),int(rp1[1])), (255,0,0), 1)
        cv2.imshow('image', img)
        cv2.waitKey()
        #src_corner = ((int(rp1[0]),int(rp1[1])), (int(rp2[0]),int(rp2[1])), (int(rp3[0]),int(rp3[1])),(int(rp4[0]),int(rp4[1])))
        src_corner = np.float32([[rp1[0],rp1[1]], [rp2[0],rp2[1]], [rp3[0],rp3[1]],[rp4[0],rp4[1]]])
        p1 = np.array([ix-iw/2.0, iy-ih/2.0])
        p2 = np.array([ix+iw/2.0, iy-ih/2.0])
        p3 = np.array([ix-iw/2.0, iy+ih/2.0])
        p4 = np.array([ix+iw/2.0, iy+ih/2.0])
        #dst_corner = ((int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])),(int(p4[0]),int(p4[1])))
        dst_corner = np.float32([[p1[0],p1[1]], [p2[0],p2[1]], [p3[0],p3[1]],[p4[0],p4[1]]])
        M1 = cv2.getPerspectiveTransform(src_corner, dst_corner)
        rows, cols, channels = img.shape
        dst = cv2.warpPerspective(img, M1, (cols, rows))
        cv2.line(dst, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0), 1)
        cv2.line(dst, (int(p2[0]),int(p2[1])), (int(p4[0]),int(p4[1])), (255,0,0), 1)
        cv2.line(dst, (int(p4[0]),int(p4[1])), (int(p3[0]),int(p3[1])), (255,0,0), 1)
        cv2.line(dst, (int(p3[0]),int(p3[1])), (int(p1[0]),int(p1[1])), (255,0,0), 1)
        cv2.imshow('image', dst)
        if cv2.waitKey()==27:
            break

        roiImage = dst[int(p1[1]):int(p3[1]), int(p1[0]):int(p2[0])]
        cv2.imwrite(dst_dir+'{:d}.jpg'.format(i), roiImage)
        i+=1

    print('there are %d images to be saved!' % int(i-1))
    print('done')
    cv2.destroyAllWindows()
