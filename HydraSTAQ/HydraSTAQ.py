import os
import numpy as np
from cv2 import medianBlur, imread, imwrite, warpAffine, IMREAD_GRAYSCALE
from math import sqrt
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import scoreatpercentile
from skimage.measure import regionprops, label as lbl
from time import time, process_time
from tkinter import Tk, IntVar, DoubleVar, StringVar, Scale, Radiobutton, Button, Label, HORIZONTAL, filedialog


class Config:  
    # Configuration class to hold and manage all the configuration parameters.   
    def __init__(self):
        self.basepath = 'C:/F/astro/matlab/m1test/'
        self.darkPathRGB = 'darks/darkframe10.tif'
        self.darkPathH = 'darks/darkframe20.tif'
        self.inputFormat = '.png'
        self.filter = 'R'
        self.align = 'R'
        self.maxStars = 10
        self.discardPercentage = 10
        self.topMatchesMasterAlign = 3
        self.topMatchesMonoAlign = 3
        self.medianOver = 10
        self.ROI_y =  [1, 2822]
        self.ROI_x =  [1, 4144]


def getFilenames(config):   
    #Function to get all the file names in the given directory.   
    filenames = []
    for root, dirs, files in os.walk(os.path.join(config.basepath, 'lights')):
        if config.filter in root:
            for file in files:
                if file.endswith(config.inputFormat):
                    filenames.append(os.path.join(root, file))
  
    return filenames


def analyzeStarField(lightFrame, config):
    #Function to analyze the star field in the given light frame.
    if config.filter == "H":
        threshold = 0.5
        factor = 3
    else:
        threshold = 0.88
        factor = 1
    filteredImage = medianBlur(factor*lightFrame,3)   
    BW = filteredImage > (threshold*255) # HACK
    labels = lbl(BW)
    stats = regionprops(labels)
    starMatrix = np.array([[stat.bbox[1], stat.bbox[0], stat.bbox[2]-stat.bbox[0], stat.bbox[3]-stat.bbox[1], np.sqrt((stat.bbox[2]-stat.bbox[0])**2 + (stat.bbox[3]-stat.bbox[1])**2)] for stat in stats])
    
    return starMatrix


def triangles(x, y):
    #Function to calculate the triangles formed by the stars in the given frame. 
    triangleParameters = np.zeros((int(len(x)*(len(x)-1)*(len(x)-2)/6),5))
    count = 0
    for i in range(len(x)-2):
        for j in range (i+1, len(x)-1):
            for k in range(j+1, len(x)): 
                d = [sqrt((x[i]-x[j])**2 +(y[i]-y[j])**2), sqrt((x[j]-x[k])**2 +(y[j]-y[k])**2), sqrt((x[i]-x[k])**2 +(y[i]-y[k])**2)];                        
                a = sorted(d) 
                m = len(a)//2
                u = ((a[m]+a[-m-1])/2)/max(d) #Median calculation trick
                v = min(d)/max(d)
                triangleParameters[count] = [i, j, k, u, v]
                count = count+1     

    return triangleParameters


def findRT(A, B):
    #Find optimal rotation and translation from two sets of points. Cred to https://nghiaho.com/?page_id=671
    centroid_A = np.mean(A, axis=1).reshape(-1, 1)
    centroid_B = np.mean(B, axis=1).reshape(-1, 1)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = np.dot(Am, Bm.T)

    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = np.dot(V, U.T)

    if np.linalg.det(R) < 0:
        V[:, 1] = V[:, 1] * -1
        R = np.dot(V, U.T)

    theta = np.arcsin(R[1, 0])
    t = -np.dot(R, centroid_A) + centroid_B

    return theta, t


def alignFrames(refVectorX, refVectorY, refTriangles, topMatches, xvec, yvec):
    #Function that aligns the frames with a "vote matrix"
    e = 0.0005
    frameTriangles = triangles(xvec, yvec)
    vote = np.zeros((len(refVectorX), len(yvec)))
    cVote = np.zeros((len(refVectorX), len(yvec)))

    for a in range(refTriangles.shape[0]):
        triangleList = np.where((refTriangles[a, 3] - e < frameTriangles[:, 3]) & (frameTriangles[:, 3] < refTriangles[a, 3] + e))[0]
        for c in range(len(triangleList)):
            b = triangleList[c]
            if np.sqrt((refTriangles[a, 3] - frameTriangles[b, 3])**2 + (refTriangles[a, 4] - frameTriangles[b, 4])**2) < e:
                vote[int(refTriangles[a, 0]), int(frameTriangles[b, 0])] += 1
                vote[int(refTriangles[a, 1]), int(frameTriangles[b, 1])] += 1
                vote[int(refTriangles[a, 2]), int(frameTriangles[b, 2])] += 1

    for row in range(vote.shape[0]):
        maxRowVote = np.max(vote[row, :])
        ind = np.argmax(vote[row, :])
        cVote[row, ind] = maxRowVote - max(max(np.delete(vote[row, :],ind)),max(np.delete(vote[:, ind],row)))

    cVote = np.maximum(vote, 0)

    maxVote = cVote.max(axis=0)
    maxVoteIndex = cVote.argmax(axis=0)

    votePairs = np.array([np.arange(len(maxVoteIndex)), maxVoteIndex, maxVote]).T
    rankPairs = votePairs[np.argsort(votePairs[:, 2])][::-1]
    rankPairs = rankPairs.astype(int)

    if len(rankPairs[:, 0]) >= topMatches:
        referenceMatrix = np.array([refVectorX[rankPairs[:topMatches, 1]], refVectorY[rankPairs[:topMatches, 1]]])
        frameMatrix = np.array([xvec[rankPairs[:topMatches, 0]], yvec[rankPairs[:topMatches, 0]]])
        theta, t = findRT(frameMatrix, referenceMatrix)
        d = 0
    else:
        print('Cannot stack frame')
        theta = 0
        t = np.zeros((2, 1))
        d = 1

    return theta, t, d


def readImages(config):
    start_time = time()
    start_timeP = process_time()

    xvec = []
    yvec = []
    qual = []

    fileNameArray = getFilenames(config)
    stars = []
    background = []
    xvec = np.empty(len(fileNameArray), dtype=object)
    yvec = np.empty(len(fileNameArray), dtype=object)
    
    for n in range(len(fileNameArray)):  
        lightFrame = np.asarray(imread(fileNameArray[n], IMREAD_GRAYSCALE))
        lightFrame = lightFrame[config.ROI_y[0]:config.ROI_y[1], config.ROI_x[0]:config.ROI_x[1]]

        background.append(np.sum(lightFrame))

        starMatrix = analyzeStarField(lightFrame, config)
        stars.append(len(starMatrix))

        corrMatrix = starMatrix[np.argsort(starMatrix[:, 4])][::-1]

        corrMatrix = corrMatrix.T
        
        if corrMatrix.shape[1] > config.maxStars:
            corrMatrix = corrMatrix[:, :config.maxStars:]
   
        xvec[n] = (corrMatrix[0,:] + corrMatrix[2,:]/2)
        yvec[n] = (corrMatrix[1,:] + corrMatrix[3,:]/2)
        
        if n % 10 == 0:
            print("Reading", f'{len(fileNameArray)}', "frames: {}%".format(int(100*n/(len(fileNameArray)-1))), end=" ", flush=True)
            print("\r", end='')
    
    qual = [s/max(stars) for s in stars]
    q = qual.index(max(qual))
    refVectorX = xvec[q]
    refVectorY = yvec[q]
    maxQualFramePath = fileNameArray[q]

    end_time = time()
    end_timeP = process_time()
    print("\n")
    print("Elapsed time:", f'{end_time - start_time:.4f}') 
    print("Elapsed CPU time:", f'{end_timeP - start_timeP:.4f}') 
    
    savemat(os.path.join(config.basepath, 'parametersPY', f'xvec{config.filter}.mat'), {'xvec': xvec})
    savemat(os.path.join(config.basepath, 'parametersPY', f'yvec{config.filter}.mat'), {'yvec': yvec})
    savemat(os.path.join(config.basepath, 'parametersPY', f'background{config.filter}.mat'), {'background': background})
    savemat(os.path.join(config.basepath, 'parametersPY', f'qual{config.filter}.mat'), {'qual': qual})
    savemat(os.path.join(config.basepath, 'parametersPY', f'maxQualFramePath{config.filter}.mat'), {'maxQualFramePath': maxQualFramePath})
    savemat(os.path.join(config.basepath, 'parametersPY', f'refVectorX{config.filter}.mat'), {'refVectorX': refVectorX})
    savemat(os.path.join(config.basepath, 'parametersPY', f'refVectorY{config.filter}.mat'), {'refVectorY': refVectorY})


def computeOffsets(config):
    start_time = time()
    start_timeP = process_time()

    xvec = loadmat(os.path.join(config.basepath, 'parametersPY', f'xvec{config.filter}.mat'))['xvec'].ravel()
    yvec = loadmat(os.path.join(config.basepath, 'parametersPY', f'yvec{config.filter}.mat'))['yvec'].ravel()
    qual = loadmat(os.path.join(config.basepath, 'parametersPY', f'qual{config.filter}.mat'))['qual'].ravel()
    maxQualFramePath = loadmat(os.path.join(config.basepath, 'parametersPY', f'maxQualFramePath{config.filter}.mat'))['maxQualFramePath'].ravel()
    refVectorX = loadmat(os.path.join(config.basepath, 'parametersPY', f'refVectorX{config.filter}.mat'))['refVectorX'].ravel()
    refVectorY = loadmat(os.path.join(config.basepath, 'parametersPY', f'refVectorY{config.filter}.mat'))['refVectorY'].ravel()
    refVectorXAlign = loadmat(os.path.join(config.basepath, 'parametersPY', f'refVectorX{config.align}.mat'))['refVectorX'].ravel()
    refVectorYAlign = loadmat(os.path.join(config.basepath, 'parametersPY', f'refVectorY{config.align}.mat'))['refVectorY'].ravel()

    qt = scoreatpercentile(qual, config.discardPercentage)

    selectedFrames = np.where(qt <= qual)[0]

    dx = np.zeros(len(selectedFrames))
    dy = np.zeros(len(selectedFrames))
    th = np.zeros(len(selectedFrames))
    discardFrames = np.zeros(len(selectedFrames), dtype=np.uint32)

    refTriangles = triangles(refVectorX, refVectorY)
    refTrianglesAlign = triangles(refVectorXAlign, refVectorYAlign)

    refTriangles = refTriangles[np.argsort(refTriangles[:, 3])]
    refTrianglesAlign = refTrianglesAlign[np.argsort(refTrianglesAlign[:, 3])]

    mth, mt, bla = alignFrames(refVectorXAlign, refVectorYAlign, refTrianglesAlign, config.topMatchesMasterAlign, refVectorX, refVectorY)
    for i in range(len(selectedFrames)):
        theta, t, d = alignFrames(refVectorX, refVectorY, refTriangles, config.topMatchesMonoAlign, xvec[selectedFrames[i]].ravel(), yvec[selectedFrames[i]].ravel())
        tmp = np.array([[np.cos(mth), -np.sin(mth)], [np.sin(mth), np.cos(mth)]]).dot(np.array([t[0], t[1]])) + np.array([mt[0], mt[1]])
        dx[i] = tmp[0]
        dy[i] = tmp[1]
        th[i] = theta + mth
        discardFrames[i] = d

    dx = dx[discardFrames == 0]
    dy = dy[discardFrames == 0]
    th = th[discardFrames == 0]
    selectedFrames = selectedFrames[discardFrames == 0]

    maxQualFrame = np.asarray(imread(maxQualFramePath[0], IMREAD_GRAYSCALE))
    maxQualFrame = maxQualFrame[config.ROI_y[0]:config.ROI_y[1], config.ROI_x[0]:config.ROI_x[1]]

    end_time = time()
    end_timeP = process_time()
    
    print("Computed offsets for", f'{len(selectedFrames)}', "frames")
    print("Elapsed time:", f'{end_time - start_time:.4f}') 
    print("Elapsed CPU time:", f'{end_timeP - start_timeP:.4f}') 

    plt.figure(1)
    plt.imshow(maxQualFrame, cmap='gray', vmin = 0, vmax = 255)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.axis('off')
    plt.scatter(refVectorXAlign, refVectorYAlign, s=100, facecolors='none', edgecolors='r')

    for i in range(1,len(selectedFrames)):
        R = np.array([[np.cos(th[i]), -np.sin(th[i])], [np.sin(th[i]), np.cos(th[i])]])
        t = np.array([dx[i], dy[i]])
        debugMatrix = R.dot(np.array([xvec[selectedFrames[i]].ravel(), yvec[selectedFrames[i]].ravel()])) + np.repeat(t[:, np.newaxis], len(xvec[selectedFrames[i]].ravel()), axis=1)
        plt.scatter(debugMatrix[0, :], debugMatrix[1, :], s=60, facecolors='none', edgecolors='g')

    plt.show()

    #plt.figure(5)
    #plt.plot(th)
    #plt.show()

    #plt.figure(2)
    #plt.plot(qual)
    #plt.plot(background/np.max(background))
    #plt.legend(['Quality', 'Background'])
    #plt.show()
    
    #plt.figure(3)
    #plt.plot(qual[selectedFrames])
    #plt.plot(background[selectedFrames]/np.max(background[selectedFrames]))
    #plt.legend(['Quality', 'Background'])
    #plt.show() 

    savemat(os.path.join(config.basepath, 'parametersPY', f'dx{config.filter}.mat'), {'dx': dx})
    savemat(os.path.join(config.basepath, 'parametersPY', f'dy{config.filter}.mat'), {'dy': dy})
    savemat(os.path.join(config.basepath, 'parametersPY', f'th{config.filter}.mat'), {'th': th})
    savemat(os.path.join(config.basepath, 'parametersPY', f'selectedFrames{config.filter}.mat'), {'selectedFrames': selectedFrames})


def stackImages(config):  
    start_time = time()
    start_timeP = process_time()

    dx = loadmat(os.path.join(config.basepath, 'parametersPY', f'dx{config.filter}.mat'))['dx'].ravel()
    dy = loadmat(os.path.join(config.basepath, 'parametersPY', f'dy{config.filter}.mat'))['dy'].ravel()
    th = loadmat(os.path.join(config.basepath, 'parametersPY', f'th{config.filter}.mat'))['th'].ravel()
    selectedFrames = loadmat(os.path.join(config.basepath, 'parametersPY', f'selectedFrames{config.filter}.mat'))['selectedFrames'].ravel()
    background = loadmat(os.path.join(config.basepath, 'parametersPY', f'background{config.filter}.mat'))['background'].ravel()

    darkPath = config.darkPathH if config.filter == "H" else config.darkPathRGB

    darkFrame = imread(os.path.join(config.basepath, darkPath), IMREAD_GRAYSCALE)  
    darkFrame = darkFrame[config.ROI_y[0]:config.ROI_y[1], config.ROI_x[0]:config.ROI_x[1]]
    darkFrame = darkFrame.astype(np.float32)/255.0 

    stackFrame = np.zeros(((config.ROI_y[1] - config.ROI_y[0]), (config.ROI_x[1] - config.ROI_x[0])), dtype=np.float32)
    temparray = np.zeros(((config.ROI_y[1] - config.ROI_y[0]), (config.ROI_x[1] - config.ROI_x[0]), config.medianOver), dtype=np.float32)

    tempcount = 1
    fileNameArray = getFilenames(config)    

    for i in range(len(selectedFrames)):
        lightFrame = np.asarray(imread(fileNameArray[selectedFrames[i]],IMREAD_GRAYSCALE))
        lightFrame = lightFrame[config.ROI_y[0]:config.ROI_y[1], config.ROI_x[0]:config.ROI_x[1]]
        lightFrame = lightFrame.astype(np.float32)/255.0
        lightFrame *= max(background[selectedFrames])/background[selectedFrames[i]]
        lightFrame -= darkFrame
         
        M = np.float32([[np.cos(th[i]), -np.sin(th[i]), dx[i]], [np.sin(th[i]), np.cos(th[i]), dy[i]]])
        lightFrame = warpAffine(lightFrame,M,(lightFrame.shape[1], lightFrame.shape[0]))
       
        temparray[:, :, tempcount-1] = lightFrame[:(config.ROI_y[1] - config.ROI_y[0]), :(config.ROI_x[1] - config.ROI_x[0])]

        tempcount += 1
        if (((i+1) % config.medianOver) == 0):
            print("Stacking", f'{len(selectedFrames)}', "frames: {}%".format(int(100*i/(len(selectedFrames)-1))), end=" ", flush=True)
            print("\r", end='')
            stackFrame = stackFrame + np.median(temparray,axis=2)/(len(selectedFrames)//config.medianOver);
            temparray = np.zeros(((config.ROI_y[1] - config.ROI_y[0]), (config.ROI_x[1] - config.ROI_x[0]), config.medianOver), dtype=np.float32)
            tempcount = 1;

    end_time = time()
    end_timeP = process_time()
    print("\n")
    print("Elapsed time:", f'{end_time - start_time:.4f}') 
    print("Elapsed CPU time:", f'{end_timeP - start_timeP:.4f}') 

    plt.imshow(stackFrame, cmap='gray')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.axis('off')
    plt.show()

    imwrite(os.path.join(config.basepath, 'outPY', f'{len(selectedFrames)}_{config.filter}.tif'), stackFrame)


def selection():
   filterTuple = ("L","R","G","B","H")
   alignTuple = ("L","R","G","B","H")
   if todoSelector.get()==0:
        config.maxStars=s0.get()
        config.filter = filterTuple[filterSelector.get()]
        readImages(config)  
   elif todoSelector.get()==1:
        config.discardPercentage=s1.get()
        config.filter = filterTuple[filterSelector.get()]
        config.align = alignTuple[alignSelector.get()]
        config.topMatchesMasterAlign=(s2.get()-1)
        config.topMatchesMonoAlign=(s3.get()-1)
        computeOffsets(config)
   elif todoSelector.get()==2:
        config.filter = filterTuple[filterSelector.get()]
        config.medianOver=s4.get()
        stackImages(config)

def selectPathButton():
    basePathName = filedialog.askdirectory()
    pathString.set(basePathName)
    config.basepath = basePathName


###################################################


win = Tk()
win.title("HydraSTAQ")
config = Config()

todoSelector = IntVar()
filterSelector = IntVar()
alignSelector = IntVar()
filterSelector.set(1)
alignSelector.set(1)

v0 = DoubleVar()
v1 = DoubleVar()
v2 = DoubleVar()
v3 = DoubleVar()
v4 = DoubleVar()

pathString = StringVar()

t0 = Radiobutton(win, text="Read images", variable=todoSelector, value=0).grid(row=0, sticky='w')
t1 = Radiobutton(win, text="Compute offsets", variable=todoSelector, value=1).grid(row=1, sticky='w')
t2 = Radiobutton(win, text="Stack images", variable=todoSelector, value=2).grid(row=2,  sticky='w')
B1 = Button(win, text ="Execute", command = selection).grid(row=3)
label = Label(text="Make a choice").grid(row=4)

Label(win, text="Max stars").grid(row=0, column=1)
Label(win, text="Discard percentage").grid(row=1, column=1)
Label(win, text="Ref. align stars").grid(row=2, column=1)
Label(win, text="Align stars").grid(row=3, column=1)
Label(win, text="Median over").grid(row=4, column=1)

s0 = Scale(win, variable = v0, from_ = 5, to = 10, orient = HORIZONTAL); s0.grid(row=0, column=2); s0.set(10)
s1 = Scale(win, variable = v1, from_ = 0, to = 99, orient = HORIZONTAL); s1.grid(row=1, column=2); s1.set(10)
s2 = Scale(win, variable = v2, from_ = 3, to = 8, orient = HORIZONTAL); s2.grid(row=2, column=2); s2.set(4)
s3 = Scale(win, variable = v3, from_ = 3, to = 8, orient = HORIZONTAL); s3.grid(row=3, column=2); s3.set(4)
s4 = Scale(win, variable = v4, from_ = 10, to = 30, orient = HORIZONTAL); s4.grid(row=4, column=2); s4.set(30)

f0 = Radiobutton(win, text="Process L", variable=filterSelector, value=0).grid(row=0, column=3, sticky='w')
f1 = Radiobutton(win, text="Process R", variable=filterSelector, value=1).grid(row=1, column=3, sticky='w')
f2 = Radiobutton(win, text="Process G", variable=filterSelector, value=2).grid(row=2, column=3, sticky='w')
f3 = Radiobutton(win, text="Process B", variable=filterSelector, value=3).grid(row=3, column=3, sticky='w')
f3 = Radiobutton(win, text="Process Ha", variable=filterSelector, value=4).grid(row=4, column=3, sticky='w')

a0 = Radiobutton(win, text="Align by L", variable=alignSelector, value=0).grid(row=0, column=4, sticky='w')
a1 = Radiobutton(win, text="Align by R", variable=alignSelector, value=1).grid(row=1, column=4, sticky='w')
a2 = Radiobutton(win, text="Align by G", variable=alignSelector, value=2).grid(row=2, column=4, sticky='w')
a3 = Radiobutton(win, text="Align by B", variable=alignSelector, value=3).grid(row=3, column=4, sticky='w')
a3 = Radiobutton(win, text="Align by Ha", variable=alignSelector, value=4).grid(row=4, column=4, sticky='w')

B2 = Button(text="Select base path", command=selectPathButton).grid(row=0, column=5)
Label(master=win,textvariable=pathString).grid(row=1, column=5)

win.mainloop()



