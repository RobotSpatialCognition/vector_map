import cv2
import numpy as np
# import matplotlib.pyplot as plt
import math
import random

tp = [(-1,0),(0,1),(1,0),(0,-1),(-1,1), (1,1), (1,-1),(-1,-1) ]
til_list = [(-1,-1),(-1,1),(1,-1),(1,1)]

def gen_sk_map(gray_img, kernel = 5):#背景白，線黒　→ 背景黒，線白
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
	_, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
	img = 255*np.ones_like(gray_img) - gray_img
	img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	skeleton_map = cv2.ximgproc.thinning(img_closing, thinningType=cv2.ximgproc.THINNING_GUOHALL)
	return skeleton_map

def wall_detected1(sk_map): #0,1のマップ
	y_len,x_len = sk_map.shape
	property_map = np.zeros_like(sk_map)
	c = 0
	for x in range(x_len):
		for y in range(y_len):
			if sk_map[y,x] == 0:
				property_map [y,x] = 2
			else:
				property_map [y,x] = 12
	return property_map

def make_wall_list(sk_map):
	return list(zip(*np.where(sk_map != 0)))

def detect_keyp(local_map):
    is_keyp = False
    lm = local_map.ravel().tolist()
    for i in range((len(lm)-1)//2):
        if lm[i]*lm[-i-1] != 0:
            break
    else:
        is_keyp =True
    return is_keyp

def get_corner_list(wall_list, sk_map):
	new_map = np.zeros_like(sk_map)
	corner_list = list()
	for x,y in wall_list:
		local_map = np.empty((3,3))
		for k in range(-1,2):
			for l in range(-1,2):
				local_map[k+1][l+1]=sk_map[x+k][y+l]
		if detect_keyp(local_map):
			corner_list.append((x,y))
	return corner_list

def wall_detected2(property_map,sk_map):
	# print(sk_map)#0,1のマップ #cornerのラベル貼り
	wall_list = make_wall_list(sk_map)
	corner_list = get_corner_list(wall_list, sk_map)
	# print(corner_list)
	y_len,x_len = property_map.shape
	c = 0
	for x in range(x_len):
		for y in range(y_len):
			if (y,x) in corner_list:
				property_map[y,x] = 10

	return property_map

def make_rotate_map(sk_map):
	m,n = sk_map.shape
	c = np.empty([len(tp),m*n])
	cnt = 0
	for s in tp:
		tmp = np.roll(sk_map,s,axis=(0,1))
		c[cnt] = tmp.reshape([1,m*n])
		cnt += 1
	return c

def get_tp_points(bmap,wlist):
	terminal_points = []
	bmap = bmap.copy()
	for i,j in wlist:
		if bmap[i-1:i+2,j-1:j+2].sum() < 3:
			terminal_points.append((i,j))
	return terminal_points

def find_next_wall(x,y,rotate_map,n,before_x, before_y):
	index = x + n * y
	l =	[i for i in range(8) if rotate_map[i][index]  != 0 ]
	for j in l:
		if (x,y) != (x - tp[j][1], y-tp[j][0]) and (x - tp[j][1], y-tp[j][0]) != (before_x,before_y):
			return x - tp[j][1],y - tp[j][0], j , index
	else:
		return None
	
def get_wall_order(sk_map):
	wall_list = make_wall_list(sk_map)
	rotatemap = make_rotate_map(sk_map)
	n = sk_map.shape[1]
	ret_gtp = get_tp_points(sk_map, wall_list)
	
	if ret_gtp == []:

		current_y,current_x = wall_list[0]
	else:
		current_y,current_x = ret_gtp[0]
	sort_wall = [(current_x,current_y)]
	prev_x,prev_y = current_x,current_y

	while True:
		ret_fmw = find_next_wall(current_x,current_y,rotatemap,n,prev_x,prev_y)
		if ret_fmw != None:
			next_x,next_y,j,index =find_next_wall(current_x,current_y,rotatemap,n,prev_x,prev_y)
			
			sort_wall.append(np.array([next_x,next_y]))

			rotatemap[j][index]  = 0
			prev_x,prev_y=current_x,current_y
			current_x,current_y=next_x,next_y
		else:
			break

	return np.array(sort_wall)

def wall_detected3(property_map,sk):#塗りつぶし後の実行，内部をラベリング
	sk_cp = sk.copy()
	pm_cp = property_map.copy()
	wall_list = make_wall_list(sk_cp)
	points = get_wall_order(sk_cp)
	cv2.fillPoly(sk_cp, pts=[points], color=1)
	y_len,x_len = property_map.shape
	for x in range(x_len):
		for y in range(y_len):
			if sk_cp[y,x] == 1 and property_map[y,x] == 2:
				property_map[y,x] = 1
	return property_map

def smooth_corner(property_map,renew_point,corner_list,wall_list):
	r = np.array(property_map[renew_point[0]-1:renew_point[0]+2,renew_point[1]-1:renew_point[1]+2])
	flag_is_first = True
	for y,x in tp:
		if property_map[renew_point[0]+y,renew_point[1]+x] == 10:##上の関数で12とした座標の近傍に10があるかを判断
			if flag_is_first:
				if (renew_point[0]+y,renew_point[1]+x) in corner_list:
					corner_list.append((renew_point[0],renew_point[1]))
			else:
				if (renew_point[0]+y,renew_point[1]+x) in corner_list:
					corner_list.remove((renew_point[0]+y,renew_point[1]+x))
			if not (renew_point[0],renew_point[1]) in wall_list:
				wall_list.append((renew_point[0],renew_point[1]))
			r[y+1,x+1]=12
	return r

def renew_corner_func(property_map,sk_map):
	wall_list = make_wall_list(sk_map)
	corner_list = get_corner_list(wall_list, sk_map)
	cp_property = np.copy(property_map)
	new_corner = []
	for ty,tx in corner_list:
		renew_flag = False
		for y,x in til_list: ##斜めの座標と中心座標との差分をもつリストでループ
			if cp_property[ty+y,tx+x] == 10: #斜めの座標が10かどうか判断
				c_list = [(y,0),(0,x)]
				for cy, cx in c_list: #10である座標と現在の中心座標と隣接する座標との差分のリストを回す
					if cp_property[ty+cy,tx+cx] == 2: #中心と10の座標と隣接する座標のプロパティが2かどうかを見る
						new_corner.append((ty+cy,tx+cx))
						renew_flag = True
						break
					else:
						renew_flag = True
						break
		if renew_flag == False:
			cp_property[ty,tx] = 11
	# print(new_corner)
	for corner in new_corner:
		cp_property[corner] = 11
	cp_property[cp_property==10] = 12
	return cp_property


def get_corner_list(wall_list, sk_map):
	new_map = np.zeros_like(sk_map)
	corner_list = list()
	for x,y in wall_list:
		local_map = np.empty((3,3))
		for k in range(-1,2):
			for l in range(-1,2):
				local_map[k+1][l+1]=sk_map[x+k][y+l]
		if detect_keyp(local_map):
			corner_list.append((x,y))
	return corner_list


def get_corner_list_from_pm(propertymap):
	return list(zip(*np.where(propertymap == 11)))

class cycle:
    def __init__(self, list):
        self.i = 0
        self.list = list
 
    def next(self):
        self.i = (self.i + 1) % len(self.list)
        return self.list[self.i]
 
    def previous(self):
        self.i = (self.i - 1 + len(self.list)) % len(self.list)
        return self.list[self.i]
 
    def present(self):
        return self.list[self.i]
    

def get_new_walllist(property_map):
	cp_pm = property_map.copy()
	cp_pm = cp_pm // 10
	new_wl = []
	y ,x = cp_pm.shape
	for py in range(y):
		for px in range(x):
			if cp_pm[(py, px)] == 1:
				new_wl.append((py,px))
	return new_wl


def wallsort(property_map):
	map = property_map.copy()//10
	m,n = map.shape
	search_order = cycle([(+1, -1), (+1, 0), (+1, +1),
		(0, +1),(-1, +1), (-1, 0),
		(-1,-1),(0, -1)])

	sortlist = []
	#近傍探索順　左下から反時計回り(y,x)
	v_old = 0
	v_new = 0
	wall_list = get_new_walllist(property_map)
	ret_gtp = get_tp_points(map, wall_list)
	if ret_gtp == []:
		current_y,current_x = wall_list[0]
	else:
		current_y,current_x = ret_gtp[0]
	y,x = current_y,current_x
	end_flag = True
	sortlist = []
	ty,tx = search_order.list[0]
	while end_flag:
		point = search_order.list[v_new]
		ty,tx = point

		if map[y+ty,x+tx] == 1:
			sortlist.append((y,x))
			v_old = search_order.list.index((ty,tx))
			v_new = (v_old + 6) % len(search_order.list)
			y = y+ty
			x = x+tx
		else:
			v_new += 1
			if v_new > 7:
				v_new = 0
		if (y,x) == (current_y,current_x) and sortlist != []:
			end_flag = False
	return sortlist


def sort_corner(corner,sortwall):  ##真のコーナーに対してのコーナーソート
	sort = []
	for y,x in sortwall:

		A = (y,x)
		if A in corner and A not in sort:
			sort.append(A)
	return sort


def la(corner, tolerance):
    cp = corner
    for i in range(len(corner)):
        ty, tx = cp[i%len(corner)]
        by, bx = cp[(i-1)%len(corner)]
        ay, ax = cp[(i+1)%len(corner)]

        numer = math.fabs((ay-by)*tx - (ax-bx)*ty + ax*by - ay*bx)
        denom = math.sqrt(math.pow(ay-by,2) + math.pow(ax-bx,2))

        d = numer / denom

        if d <= tolerance:
            cp.remove((ty,tx))
            
    return cp


def calc_degree(cornerlist, propertymap):
	degree_list = []
	corner_num = len(cornerlist)
	for i in range(corner_num):
		apex = cornerlist[i % corner_num]
		p1 = cornerlist[(i-1) % corner_num]
		p2 = cornerlist[(i+1) % corner_num]
		
		vec1 = [p1[0]-apex[0], p1[1]-apex[1]]
		vec2 = [p2[0]-apex[0], p2[1]-apex[1]]
		absvec1 = np.linalg.norm(vec1)
		absvec2 = np.linalg.norm(vec2)
		inner = np.inner(vec1, vec2)
		cos_theta = inner / (absvec1 * absvec2)
		wx = (p1[0] + p2[0] + apex[0]) // 3
		wy = (p1[1] + p2[1] + apex[1]) // 3

		if cos_theta > 1.0:
			cos_theta = 1.0
		elif cos_theta < -1.0:
			cos_theta = -1.0
		
		theta = math.degrees(math.acos(cos_theta))

		if propertymap[wx, wy] == 2:
			theta = 360 - theta
		degree_list.append(theta)
	return degree_list


def delete_180(clist, dlist):
	for degree in dlist:
		index = dlist.index(degree)
		if degree == 180:
			clist.remove(clist[index])
			dlist.remove(degree)
	return  clist, dlist

def intersection(A, B, C, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # 2つの直線の傾きを計算
    slope1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
    slope2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

    # 2つの直線が平行である場合
    if slope1 == slope2:
        return None

    # 2つの直線が平行でない場合
    else:
        # 交点のx座標を計算
        if slope1 is None:
            x = x1
        elif slope2 is None:
            x = x3
        else:
            x = (y3 - y1 + slope1 * x1 - slope2 * x3) / (slope1 - slope2)

        # 交点のy座標を計算
        if slope1 is None:
            y = slope2 * (x - x3) + y3
        else:
            y = slope1 * (x - x1) + y1

        return (x, y)



def unit_vector(point_a, point_b):
    
    # AからBへのベクトルを計算する
    vec_ab = point_b - point_a
    
    # AからBへの距離を計算する
    dist_ab = np.linalg.norm(vec_ab)
    
    # AからBへの単位ベクトルを計算する
    unit_vec = vec_ab / dist_ab
    
    return unit_vec

def search_nearpoint(sorted_corner, degree_list):
	dlist = degree_list
	clist = sorted_corner
	c_num = len(clist)
	nodes= []
	for i in range(c_num):
		d1 = dlist[i % c_num]
		d2 = dlist[(i-1) % c_num]
		if int(d1) > 180 and int(d2) > 180 :
			candidates = []
			nearest = ()
			distance = 10000
			corner_y, corner_x = clist[(i-1) % c_num]
			before_y, before_x = clist[(i-2) % c_num]
			for j in range(c_num):
				if clist[(j) % c_num] != clist[(i) % c_num] and clist[(j-1) % c_num] != clist[(i-1) % c_num]:
					ay,ax = clist[(j) % c_num]
					by,bx = clist[(j-1) % c_num]
					cross= intersection((corner_x,corner_y),(before_x,before_y),(ax,ay),(bx,by))

					if cross != None:
						intersect_x,intersect_y = cross
						if min(ay,by) <= intersect_y <= max(ay,by) and min(ax,bx) <= intersect_x <= max(ax,bx):
							candidates.append((intersect_y,intersect_x))
							
			if candidates != []:
				for y,x in candidates:
					before = np.array((before_y, before_x))
					corner = np.array((corner_y,corner_x))
					point = np.array((y,x))
					
					e1 = unit_vector(before,corner)
					e2 = unit_vector(corner,point)
					dist = np.linalg.norm(point-corner)
					if dist <= distance and np.allclose(e1, e2):
						distance = dist
						nearest = point
				if nearest != ():
					nodes.append((corner,nearest))

	return nodes


def unit_normal_vectors(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    norm = math.sqrt(dx ** 2 + dy ** 2)
    nx1 = -dy / norm
    ny1 = dx / norm
    nx2 = dy / norm
    ny2 = -dx / norm

    return ((nx1, ny1), (nx2, ny2))

def calc_half(x1, y1, x2, y2):
    x = (x1 + x2)/2
    y = (y1 + y2)/2
    
    return y, x


def mono2color(mono_img):
    out_img = np.zeros((mono_img.shape[0],mono_img.shape[1],3))
    out_img[:,:,0] = mono_img.copy()
    out_img[:,:,1] = mono_img.copy()
    out_img[:,:,2] = mono_img.copy()
    return out_img


def make_mapbb(img):
    # boundingboxの中心位置(y,x)と回転時に余裕のあるサイズを返す

    vy, vx = np.where(img==0)
    max_vx = vx.max()
    max_vy = vy.max()
    min_vx = vx.min()
    min_vy = vy.min()
    dx = max_vx-min_vx
    dy = max_vy-min_vy
    d = max(dx,dy)
    r = (np.sqrt(2)*d)//2
    center = ((min_vx+max_vx)//2, (min_vy+max_vy)//2)
    return center, int(r)


def img_crop(img):
	center ,r = make_mapbb(img)
	# print(center,r)
	if 2*r < img.shape[0] and 2*r < img.shape[1]:
		img = img[center[1]-r:center[1]+r, center[0]-r:center[0]+r]
		return img,(center[1]-r, center[0]-r) 
	else:
		return img,(0,0)

	
def rotation(img, angle = -233):
    scale = 1.0
    unicolor = np.unique(img)
    if len(unicolor)%2 ==0:
        unicolor = np.append(unicolor,np.max(unicolor))
    tc = np.median(unicolor)

    #getRotationMatrix2D関数を使用
    center = (img.shape[1]//2, img.shape[0]//2)
    # 回転行列の作成
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    #アフィン変換
    image2 = cv2.warpAffine(img, trans, (img.shape[0], img.shape[1]), borderValue=tc, flags=cv2.INTER_NEAREST)

    return image2

def addition_property(bin_raster):
	
	raster = bin_raster.astype(np.int32) ##要素の型変更
	p_map = wall_detected1(raster) ##壁とそれ以外を分類
	p2_map = wall_detected2(p_map,raster) ##暫定コーナーの抽出
	p3_map = wall_detected3(p2_map, raster) ##内部と外部を塗り分け
	p = renew_corner_func(p3_map, raster) ##暫定コーナーをコーナーにする処理
	corner_list = get_corner_list_from_pm(p)
	return p, corner_list

def approximate_corner(tmp_property,tmp_corners):
	cp_property = tmp_property.copy()
	cp_property[cp_property == 1] = 19# 内部塗りつぶし


	sort = wallsort(cp_property)

	sorted_corner = sort_corner(tmp_corners, sort)

#	reduced_corner = la(sorted_corner, 4)
	reduced_corner = la(sorted_corner, 3)
	reduced_corner_cv2 = []
	for py,px in reduced_corner:
		reduced_corner_cv2.append((px,py))
	degree_list = calc_degree(reduced_corner,tmp_property)
	
	testp = tmp_property.copy()
	for point in tmp_corners:
		testp[point] = 12

	for point in reduced_corner:
		testp[point]=15
	
	clist, dlist  = delete_180(reduced_corner, degree_list)
	for point in tmp_corners:
		testp[point] = 12

	for point in clist:
		testp[point]=15

	return testp, clist, dlist




def getMovePoint(img_org):
	
	
	img_org, _ = img_crop(img_org)

	skeleton_map = gen_sk_map(img_org, 11) ##スケルトンマップの生成　（0,255)



	##関数化済み　->addition_property()
	tmp_sk =skeleton_map/255##(0,1)のスケルトンマップに変更

	tmp_sk = np.pad(tmp_sk, 10, constant_values=0) ##padding 各方向に10画素ずつ
	tmp_sk = tmp_sk.astype(np.int32) ##要素の型変更
	p_map = wall_detected1(tmp_sk) ##壁とそれ以外を分類
	p2_map = wall_detected2(p_map,tmp_sk) ##暫定コーナーの抽出
	p3_map = wall_detected3(p2_map, tmp_sk) ##内部と外部を塗り分け
	p = renew_corner_func(p3_map, tmp_sk) ##暫定コーナーをコーナーにする処理
	corner_list = get_corner_list_from_pm(p)

	###関数化済み -> approximate_corner

	p[p == 1] = 19# 内部塗りつぶし


	sort = wallsort(p)

	sorted_corner = sort_corner(corner_list, sort)

	reduced_corner = la(sorted_corner, 3)
	reduced_corner_cv2 = []
	for py,px in reduced_corner:
		reduced_corner_cv2.append((px,py))


	degree_list = calc_degree(reduced_corner,p)
	testp = p.copy()

	for point in corner_list:
		testp[point] = 12

	for point in reduced_corner:
		testp[point]=15
	
	


	clist, dlist  = delete_180(reduced_corner, degree_list)
	for point in corner_list:
		testp[point] = 12

	for point in clist:
		testp[point]=15
	
	
	output = clist
	
	###########分割関数
	nodes = search_nearpoint(clist, dlist)

	img = mono2color(tmp_sk)
	binimg = img * 255
	for points in nodes:
		cv2.line(binimg, (int(points[0][1]),int(points[0][0])), (int(points[1][1]),int(points[1][0])), (255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

	binimg = binimg.astype("uint8")
	h,w,_ = binimg.shape
	bin = np.zeros((h,w), dtype=np.uint8)
	bin = binimg[:,:,0].copy()

	reverse = 255*np.ones_like(bin) - bin
	kernel = np.ones((2,2), np.uint8)
	reverse_bin = cv2.erode(reverse, kernel, iterations=1)
	reverse_bin[testp != 19] = 0
	label = cv2.connectedComponentsWithStats(reverse_bin)

	n = label[0] -1
	labelImage = label[1]
	data = np.delete(label[2], 0, 0)
	center = np.delete(label[3], 0, 0)

	# ラベリング結果書き出し用に二値画像をカラー変換
	color_src = cv2.cvtColor(reverse, cv2.COLOR_GRAY2BGR)
	# debug.show(color_src)
	height, width = reverse.shape[:2]
	colors = []
	for i in range(1, n+2):
		colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))
	for y in range(0, height):
		for x in range(0, width):
			if labelImage[y, x] > 0:
				color_src[y, x] = colors[labelImage[y, x]]

			else:
				color_src[y, x] = [0, 0, 0]

	# オブジェクト情報を利用してラベリング結果を表示
	# 画像の保存
	for c in center:
		cv2.putText(color_src,
				text=str(labelImage[int(c[1]),int(c[0])]),
				org=(int(c[0]),int(c[1])),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(255, 255, 255),
				thickness=2,
				lineType=cv2.LINE_4)
		cv2.drawMarker(color_src,
				position=(int(c[0]),int(c[1])),
				color=(255, 0, 0),
				markerType=cv2.MARKER_CROSS,
				markerSize=5,
				thickness=2,
				line_type=cv2.LINE_4
				)

	#########


	#隣接行列作成
	adjacent_matrix = np.zeros((n,n),dtype="uint8")
	for node in nodes:
		py, px = node[0]
		ty, tx = node[1]
		vec1, vec2 = unit_normal_vectors(px,py,tx,ty)
		y, x = calc_half(px,py,tx,ty)

		label1 = labelImage[int(y + 5 * vec1[1]),int(x + 5 * vec1[0])]
		label2 = labelImage[int(y + 5 * vec2[1]),int(x + 5 * vec2[0])]
		adjacent_matrix[label1-1, label2-1] = 1
		adjacent_matrix[label2-1, label1-1] = 1


#	center[:,0] += offset[0]
#	center[:,1] += offset[1]

	center_list = []
	for c in center:
		center_list.append([c[0],c[1]])

	output = np.empty((len(clist),2),dtype=np.int64)

	for n,c in enumerate(clist):
		output[n][0] = c[0]
		output[n][1] = c[1]
#	output[:,0] += offset[0]
#	output[:,1] += offset[1]




	return center, adjacent_matrix, center_list, output
