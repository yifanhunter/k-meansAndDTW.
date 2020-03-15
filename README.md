# k-meansAndDTW.
Clustering and fitting of time series based on DTW and k-means

一、问题分析
1、首先尝试了使用：提取时间序列的统计学特征值，例如最大值，最小值等。然后利目前常用的算法根据提取的特征进行分类，例如Naive Bayes, SVMs，KNN 等。发现效果并不是很好。
2、尝试基于K-means的无监督形式分类，这种分类方式基于两个数据的距离进行分类，这样要定义号距离的概念，后来查阅资料，考虑使用动态时间规整（Dynamic Time Warping, DTW）。

二、数据处理
给出的数据较为完整，就一个excel表格，做了以下简单的排序，原始数据可见文末github地址。

三、代码实现
3.1 动态时间规整（Dynamic Time Warping, DTW）

如果是欧拉距离：则ts3比ts2更接近ts1，但是肉眼看并非如此。故引出DTW距离。
        动态时间规整算法，故名思议，就是把两个代表同一个类型的事物的不同长度序列进行时间上的“对齐”。比如DTW最常用的地方，语音识别中，同一个字母，由不同人发音，长短肯定不一样，把声音记录下来以后，它的信号肯定是很相似的，只是在时间上不太对整齐而已。所以我们需要用一个函数拉长或者缩短其中一个信号，使得它们之间的误差达到最小。下面这篇博文给了比较好的解释：https://blog.csdn.net/lin_limin/article/details/81241058。 简单英文解释如下（简而言之：就是允许错开求差值，并且取最小的那个作为距离。）

DTW距离代码定义如下：
    1 	def DTWDistance(s1, s2):
    2 	    DTW={}
    3 	    for i in range(len(s1)):
    4 	        DTW[(i, -1)] = float('inf')
    5 	    for i in range(len(s2)):
    6 	        DTW[(-1, i)] = float('inf')
    7 	    DTW[(-1, -1)] = 0
    8 	    for i in range(len(s1)):
    9 	        for j in range(len(s2)):
   10 	            dist= (s1[i]-s2[j])**2
   11 	            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
   12 			
   13 	    return math.sqrt(DTW[len(s1)-1, len(s2)-1])
   14 
   15 
这样求解相对较为麻烦，时间复杂度比较高，故做了一个小的加速：
#DTW距离,只检测前W个窗口的值，定义错开的部分W，减少递归寻找量
    1 	def DTWDistance(s1, s2, w):
    2 	    DTW = {}
    3 	    w = max(w, abs(len(s1) - len(s2)))
    4 	    for i in range(-1, len(s1)):
    5 	        for j in range(-1, len(s2)):
    6 	            DTW[(i, j)] = float('inf')
    7 	    DTW[(-1, -1)] = 0
    8 	    for i in range(len(s1)):
    9 	        for j in range(max(0, i - w), min(len(s2), i + w)):
   10 	            dist = (s1[i] - s2[j]) ** 2
   11 	            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
   12 	    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])
3.2  LB_Keogh距离
主要思想是在搜索数据很大的时候， 逐个用DTW算法比较每一条是否匹配非常耗时。那我们能不能使用一种计算较快的近似方法计算LB， 通过LB处理掉大部分不可能是最优匹配序列的序列，对于剩下的序列在使用DTW逐个比较呢？英文解释如下：

中文解释如下，主要参考其它博文：LB_keogh的定义相对复杂，包括两部分。
第一部分为Q的{U， L} 包络曲线（具体如图）， 给Q序列的每个时间步定义上下界。 定义如下

其中 r 是一段滑行窗距离，可以自定义。
示意图如下：

U 为上包络线，就是把每个时间步为Q当前时间步前后r的范围内最大的数。L 下包络线同理。那么LB_Keogh定义如下：

用图像描述如下：

    1 	def LB_Keogh(s1, s2, r):
    2 	    LB_sum = 0
    3 	    for ind, i in enumerate(s1):
    4 	        # print(s2)
    5 	        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
    6 	        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
    7 	        if i >= upper_bound:
    8 	            LB_sum = LB_sum + (i - upper_bound) ** 2
    9 	        elif i < lower_bound:
   10 	            LB_sum = LB_sum + (i - lower_bound) ** 2
   11 	    return math.sqrt(LB_sum)
  3.3  使用k-means算法实现聚类
    1 	#3  定义K-means算法
    2 	#num_clust分类的数量，
    3 	def k_means_clust(data, num_clust, num_iter, w=3):
    4 	    ## 步骤一: 初始化均值点
    5 	    centroids = random.sample(list(data), num_clust)
    6 	    counter = 0
    7 	    for n in range(num_iter):
    8 	        counter += 1
    9 	        # print
   10 	        # counter
   11 	        assignments = {}  #存储类别0，1，2等类号和所包含的类的号码
   12 	        # 遍历每一个样本点 i ,因为本题与之前有所不同，多了ind的编码
   13 	        for ind, i in enumerate(data):
   14 	            min_dist = float('inf')   #最近距离，初始定一个较大的值
   15 	            closest_clust = None     # closest_clust：最近的均值点编号
   16 	            ## 步骤二: 寻找最近的均值点
   17 	            for c_ind, j in enumerate(centroids):  #每个点和中心点的距离，共有num_clust个值
   18 	                if LB_Keogh(i, j, 3) < min_dist:    #循环去找最小的那个
   19 	                    cur_dist = DTWDistance(i, j, w)
   20 	                    if cur_dist < min_dist:         #找到了ind点距离c_ind最近
   21 	                        min_dist = cur_dist
   22 	                        closest_clust = c_ind
   23 	            ## 步骤三: 更新 ind 所属簇
   24 	            # print(closest_clust)
   25 	            if closest_clust in assignments:
   26 	                assignments[closest_clust].append(ind)
   27 	            else:
   28 	                assignments[closest_clust] = []
   29 	                assignments[closest_clust].append(ind)
   30 	        # recalculate centroids of clusters  ## 步骤四: 更新簇的均值点
   31 	        for key in assignments:
   32 	            clust_sum = 0
   33 	            for k in assignments[key]:
   34 	                clust_sum = clust_sum + data[k]
   35 	            centroids[key] = [m / len(assignments[key]) for m in clust_sum]
   36 	    return centroids,assignments    #返回聚类中心值，和聚类的所有点的数组序号
3.4  根据聚类打印出具体分类情况：
    1 	num_clust = 2  #定义需要分类的数量
    2 	centroids,assignments = k_means_clust(WBCData, num_clust,800, 3)
    3 	for i in range(num_clust):
    4 	    s = []
    5 	    WBC01 = []
    6 	    days01 = []
    7 	    for j, indj in enumerate(assignments[i]):  #画出各分类点的坐标
    8 	        s.append(int(Numb[indj*30]))
    9 	        WBC01 = np.hstack((WBC01,WBC[30 * indj:30 * indj + 30]))
   10 	        days01 = np.hstack((days01 , days[0: 30]))
   11 	    print(s)
   12 	    plt.title('%s' % s)
   13 	    plt.plot(centroids[i],c="r",lw=4)
   14 	    plt.scatter(days01, WBC01 )
   15 	    plt.show()
四、结果
定义了分成两类的情形，可以根据num_clust 的值进行灵活的调整，等于2是的分类和图示情况如下：

WBC01：[6774, 7193, 8070, 8108, 8195, 2020006799, 2020007003, 2020007251, 2020007420, 2020007636, 2020007718, 2020007928, 2020007934, 2020008022, 2020008196, 2020008239, 2020008302, 2020008354, 2020008418, 2020008513, 2020008535, 2020008737, 2020008890, 2020008909, 2020009042, 2020009043, 2020009050, 2020009201, 2020009213, 2020009289, 2020009420, 2020009557]

WBC02：[2020007250, 2020007388, 2020007389, 2020007422, 2020007625, 2020007703, 2020007927, 2020009049, 2020009158, 2020009284, 2020009580]

说明：
代码训练过程中，一定要注意数据类型，比如matrix和ndarray,虽然打印的时候都是（45，30），但是再训练的时候，稍加不注意，就会导致乱七八糟的问题，需要打印排查好久。
本文的数据和代码，请登录：my github ,进行下载。若是对您有用，请不吝给颗星。
