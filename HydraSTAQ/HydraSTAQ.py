import os
import numpy as np
from cv2 import medianBlur, imread, imwrite, warpAffine, IMREAD_GRAYSCALE, IMREAD_ANYDEPTH
from math import sqrt
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label as lbl
from time import time, process_time
from tkinter import Tk, IntVar, DoubleVar, StringVar, Scale, Radiobutton, Button, Label, HORIZONTAL, filedialog


class Config:  
    # Configuration class to hold and manage all the configuration parameters.   
    def __init__(self):
        self.basePath = ''
        self.parameterPath = 'parametersPy'
        self.outputPath = 'outPy'
        self.darkPathRGB = 'darks/RGB'
        self.darkPathNarrowband = 'darks/Narrowband'
        self.lightInputFormat = str
        self.filter = str
        self.align = str
        self.maxStars = int
        self.discardPercentage = int
        self.topMatchesMasterAlign = int
        self.topMatchesMonoAlign = int
        self.medianOver = int
        self.ROI_y = [1, 2822]
        self.ROI_x = [1, 4144]


def getLights(config, frameType, fileFormat):   
    #Function to get all the file names in the given directory.   
    filenames = []
    for root, dirs, files in os.walk(os.path.join(config.basePath, frameType)):
        if config.filter in root:
            for file in files:
                if file.endswith(fileFormat):
                    filenames.append(os.path.join(root, file))
  
    return filenames


def getCalibrationFrames(config, frameType, fileFormat):   
    #Function to get all the file names in the given directory.   
    filenames = []
    for root, dirs, files in os.walk(os.path.join(config.basePath, frameType)):
        for file in files:
            if file.endswith(fileFormat):
                filenames.append(os.path.join(root, file))
  
    return filenames


def analyzeStarField(lightFrame, config):
    #Function to analyze the star field in the given light frame.
    if config.filter == 'H':
        threshold = 0.44
        factor = 3
    else:
        threshold = 0.88
        factor = 1

    filteredImage = medianBlur(factor*lightFrame,3) 
    
    BW = filteredImage > threshold*(255**lightFrame.dtype.itemsize) 
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
                d = [sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2), sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2), sqrt((x[i]-x[k])**2 + (y[i]-y[k])**2)];                        
                a = sorted(d) 
                m = len(a)//2
                u = ((a[m]+a[-m-1])/2)/max(d) #Median calculation trick
                v = min(d)/max(d)
                triangleParameters[count] = [i, j, k, u, v]
                count = count+1     

    return triangleParameters


def findRT(A, B):
    #Function to find optimal rotation and translation from two sets of points. Cred to Nghia Ho.
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

    theta = np.arctan2(R[1,0], R[0,0])
    t = -np.dot(R, centroid_A) + centroid_B

    return theta, t


def createMasterFrame(path, type):
    #Function to create a master calibration frame
    if(os.path.isfile(os.path.join(config.basePath, path, 'Master.tif'))):
        outString.set('Loading master' + ' ' + type + ' ' + 'frame')
        win.update()
        frame = imread(os.path.join(config.basePath, path, 'Master.tif'), flags=(IMREAD_GRAYSCALE | IMREAD_ANYDEPTH))  
        frame = frame.astype(np.float32)      
    else:
        frameArray = getCalibrationFrames(config, path, '.png')  
        frame = np.zeros(((1+config.ROI_y[1] - config.ROI_y[0]), (1+config.ROI_x[1] - config.ROI_x[0])), dtype=np.float32)
        for k in range(len(frameArray)):
            outString.set('Stacking' + ' ' + f'{len(frameArray)}' + ' ' + type + 's: {}%'.format(int(100*k/(len(frameArray)-1))))
            win.update()
            tmpFrame = np.asarray(imread(frameArray[k], IMREAD_GRAYSCALE))
            tmpFrame = tmpFrame[config.ROI_y[0]-1:config.ROI_y[1], config.ROI_x[0]-1:config.ROI_x[1]]
            tmpFrame = tmpFrame.astype(np.float32)/(255**tmpFrame.dtype.itemsize)
            frame = frame+tmpFrame/len(frameArray)
         
        imwrite(os.path.join(config.basePath, path, 'Master.tif'), frame)

    return frame


def alignFrames(refVectorX, refVectorY, refTriangles, topMatches, xvec, yvec):
    #Function to align the frames with a "vote matrix"
    e = 0.01
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
        cVote[row, ind] = maxRowVote - max(max(np.delete(vote[row, :], ind)),max(np.delete(vote[:, ind], row)))

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
        theta = 0
        t = np.zeros((2, 1))
        d = 1

    return theta, t, d


def readImages(config):
    #Function to read images
    start_time = time()
    start_timeP = process_time()

    xvec = []
    yvec = []

    lightFrameArray = getLights(config, 'lights', config.lightInputFormat)

    if(len(lightFrameArray)>0):
        stars = []
        background = []
        xvec = np.empty(len(lightFrameArray), dtype=object)
        yvec = np.empty(len(lightFrameArray), dtype=object)
    
        for n in range(len(lightFrameArray)):  
            lightFrame = np.asarray(imread(lightFrameArray[n], IMREAD_GRAYSCALE))
            lightFrame = lightFrame[config.ROI_y[0]-1:config.ROI_y[1], config.ROI_x[0]-1:config.ROI_x[1]]

            background.append(np.sum(lightFrame))

            starMatrix = analyzeStarField(lightFrame, config)
            stars.append(len(starMatrix))

            if len(starMatrix) > 5:
                corrMatrix = starMatrix[np.argsort(starMatrix[:, 4])][::-1]
                corrMatrix = corrMatrix.T
        
                if corrMatrix.shape[1] > config.maxStars:
                    corrMatrix = corrMatrix[:, :config.maxStars:]
   
                xvec[n] = (corrMatrix[0,:] + corrMatrix[2,:]/2)
                yvec[n] = (corrMatrix[1,:] + corrMatrix[3,:]/2)
            else:
                xvec[n] = []
                yvec[n] = []

            if n % 10 == 0:
                outString.set('Reading' + ' ' + f'{len(lightFrameArray)}' + ' ' + 'lights: {}%'.format(int(100*n/(len(lightFrameArray)-1))))
                win.update()
    
        qual = [s/max(stars) for s in stars]
        q = qual.index(max(qual))

        qualVector = np.array([qual, background]).T
        refVector = np.array([xvec[q], yvec[q]])

        maxQualFrameFile = lightFrameArray[q]

        end_time = time()
        end_timeP = process_time()
        outString.set('Elapsed time:' + ' ' + f'{end_time - start_time:.2f}' + ' ' + 'CPU time:' + ' ' + f'{end_timeP - start_timeP:.2f}') 

        if not os.path.isdir(os.path.join(config.basePath, config.parameterPath)): os.makedirs(os.path.join(config.basePath, config.parameterPath))         

        np.save(os.path.join(config.basePath, config.parameterPath, f'xvec{config.filter}'), xvec)
        np.save(os.path.join(config.basePath, config.parameterPath, f'yvec{config.filter}'), yvec)
        np.save(os.path.join(config.basePath, config.parameterPath, f'qualVector{config.filter}'), qualVector)
        np.save(os.path.join(config.basePath, config.parameterPath, f'maxQualFrameFile{config.filter}'), maxQualFrameFile)
        np.save(os.path.join(config.basePath, config.parameterPath, f'refVector{config.filter}'), refVector)

    else:
        outString.set('No image files found.')


def computeOffsets(config):
    #Function to compute translational and rotational offsets
    start_time = time()
    start_timeP = process_time()

    xvecPath = os.path.join(config.basePath, config.parameterPath, f'xvec{config.filter}.npy')
    yvecPath = os.path.join(config.basePath, config.parameterPath, f'yvec{config.filter}.npy')
    qualVectorPath = os.path.join(config.basePath, config.parameterPath, f'qualVector{config.filter}.npy')
    maxQualFrameFilePath = os.path.join(config.basePath, config.parameterPath, f'maxQualFrameFile{config.filter}.npy')
    refVectorPath = os.path.join(config.basePath, config.parameterPath, f'refVector{config.filter}.npy')
    refVectorAlignPath = os.path.join(config.basePath, config.parameterPath, f'refVector{config.align}.npy')

    if(all([os.path.isfile(f) for f in [xvecPath, yvecPath, qualVectorPath, maxQualFrameFilePath, refVectorPath, refVectorAlignPath]])):  
        xvec = np.load(xvecPath, allow_pickle = True).ravel() 
        yvec = np.load(yvecPath, allow_pickle = True).ravel() 
        qualVector = np.load(qualVectorPath) 
        maxQualFrameFile = np.load(maxQualFrameFilePath).ravel() 
        refVector = np.load(refVectorPath) 
        refVectorAlign = np.load(refVectorAlignPath) 

        qual = qualVector[:,0].T
        background = qualVector[:,1].T
        refVectorX = refVector[0,:]
        refVectorY = refVector[1,:]
        refVectorXAlign = refVectorAlign[0,:]
        refVectorYAlign = refVectorAlign[1,:] 

        frames = np.array([np.arange(len(qual)),qual]).T
        orderedFrames = frames[np.argsort(frames[:, 1])][::-1]
        selectedFrames = np.sort(orderedFrames[:int(len(qualVector)*(1-config.discardPercentage/100)),0].astype(int))

        dx = np.zeros(len(selectedFrames))
        dy = np.zeros(len(selectedFrames))
        th = np.zeros(len(selectedFrames))
        discardFrames = np.zeros(len(selectedFrames), dtype=np.uint32)

        refTriangles = triangles(refVectorX, refVectorY)
        refTrianglesAlign = triangles(refVectorXAlign, refVectorYAlign)

        refTriangles = refTriangles[np.argsort(refTriangles[:, 3])]
        refTrianglesAlign = refTrianglesAlign[np.argsort(refTrianglesAlign[:, 3])]

        outString.set('Computing offsets for' + ' ' + f'{len(selectedFrames)}' + ' ' + 'frames')
        win.update()

        mth, mt, d = alignFrames(refVectorXAlign, refVectorYAlign, refTrianglesAlign, config.topMatchesMasterAlign, refVectorX, refVectorY)
        
        if(d==0):
            for i in range(len(selectedFrames)):
                if(len(xvec[selectedFrames[i]].ravel()) != 0):
                    theta, t, d = alignFrames(refVectorX, refVectorY, refTriangles, config.topMatchesMonoAlign, xvec[selectedFrames[i]].ravel(), yvec[selectedFrames[i]].ravel())
                    tmp = np.array([[np.cos(mth), -np.sin(mth)], [np.sin(mth), np.cos(mth)]]).dot(np.array([t[0], t[1]])) + np.array([mt[0], mt[1]])
                    dx[i] = tmp[0]
                    dy[i] = tmp[1]
                    th[i] = theta + mth
                    discardFrames[i] = d
                else:
                    discardFrames[i] = 1

            dx = dx[discardFrames == 0]
            dy = dy[discardFrames == 0]
            th = th[discardFrames == 0]
            selectedFrames = selectedFrames[discardFrames == 0]

            maxQualFrame = np.asarray(imread(maxQualFrameFile[0], IMREAD_GRAYSCALE))
            maxQualFrame = maxQualFrame[config.ROI_y[0]-1:config.ROI_y[1], config.ROI_x[0]-1:config.ROI_x[1]]

            end_time = time()
            end_timeP = process_time()
    
            outString.set('Elapsed time:' + ' ' + f'{end_time - start_time:.2f}' + ' ' + 'CPU time:' + ' ' + f'{end_timeP - start_timeP:.2f}')
        
            offsets = np.array([dx, dy, th, selectedFrames]).T
            np.save(os.path.join(config.basePath, config.parameterPath, f'offsets{config.filter}'), offsets)
  
            plt.figure(1)
            plt.imshow(maxQualFrame, cmap='gray', vmin = 0, vmax = (255**maxQualFrame.dtype.itemsize))
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.axis('off')
            plt.scatter(refVectorXAlign, refVectorYAlign, s=100, facecolors='none', edgecolors='r')

            for i in range(len(selectedFrames)):
                R = np.array([[np.cos(th[i]), -np.sin(th[i])], [np.sin(th[i]), np.cos(th[i])]])
                t = np.array([dx[i], dy[i]])
                debugMatrix = R.dot(np.array([xvec[selectedFrames[i]].ravel(), yvec[selectedFrames[i]].ravel()])) + np.repeat(t[:, np.newaxis], len(xvec[selectedFrames[i]].ravel()), axis=1)
                debugMatrix = debugMatrix[:,debugMatrix.min(axis=0)>=0]
                plt.scatter(debugMatrix[0, :], debugMatrix[1, :], s=60, facecolors='none', edgecolors='g')

            plt.show()

            plt.figure(5)
            plt.plot(th)
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')  
            ax1.plot(qual)
            ax1.plot(background/np.max(background))
            ax1.legend(['Quality', 'Background'])
  
            ax2.plot(qual[selectedFrames])
            ax2.plot(background[selectedFrames]/np.max(background[selectedFrames]))
            ax2.legend(['Quality', 'Background'])
            plt.show()
        else:
            outString.set('Filter align failure.')
    else:
        outString.set('Missing input files.')


def stackImages(config):  
    #Function to stack and calibrate the images
    start_time = time()
    start_timeP = process_time()

    lightFrameArray = getLights(config, 'lights', config.lightInputFormat)    
    offsetsPath = os.path.join(config.basePath, config.parameterPath, f'offsets{config.filter}.npy')
    qualVectorPath = os.path.join(config.basePath, config.parameterPath, f'qualVector{config.filter}.npy')

    if(len(lightFrameArray)>0 and all([os.path.isfile(f) for f in [offsetsPath, qualVectorPath]])):
        offsets = np.load(offsetsPath)
        qualVector = np.load(qualVectorPath)

        dx = offsets[:,0].T
        dy = offsets[:,1].T
        th = offsets[:,2].T
        selectedFrames = offsets[:,3].T.astype(int)
        background = qualVector[:,1].T

        darkPath = config.darkPathNarrowband if config.filter == 'H' else config.darkPathRGB

        darkFrame = createMasterFrame(darkPath, 'dark')

        stackFrame = np.zeros(((1+config.ROI_y[1] - config.ROI_y[0]), (1+config.ROI_x[1] - config.ROI_x[0])), dtype=np.float32)
        temparray = np.zeros(((1+config.ROI_y[1] - config.ROI_y[0]), (1+config.ROI_x[1] - config.ROI_x[0]), config.medianOver), dtype=np.float32)

        tempcount = 1

        iterator = np.arange(0,len(selectedFrames),1,dtype = int)
        np.random.shuffle(iterator)

        for k in range(len(selectedFrames)):
            i = iterator[k];
            lightFrame = np.asarray(imread(lightFrameArray[selectedFrames[i]],IMREAD_GRAYSCALE))
            lightFrame = lightFrame[config.ROI_y[0]-1:config.ROI_y[1], config.ROI_x[0]-1:config.ROI_x[1]]
            lightFrame = lightFrame.astype(np.float32)/(255**lightFrame.dtype.itemsize)
            lightFrame *= np.mean(background[selectedFrames])/background[selectedFrames[i]]
            lightFrame -= darkFrame
         
            M = np.float32([[np.cos(th[i]), -np.sin(th[i]), dx[i]], [np.sin(th[i]), np.cos(th[i]), dy[i]]])
            temparray[:, :, tempcount-1] = warpAffine(lightFrame, M, (lightFrame.shape[1], lightFrame.shape[0]))

            tempcount += 1
            if (((k+1) % config.medianOver) == 0):
                outString.set('Stacking' + ' ' + f'{len(selectedFrames)}' + ' ' + 'lights: {}%'.format(int(100*k/(len(selectedFrames)-1))))
                win.update()
                stackFrame = stackFrame + np.median(temparray,axis=2)/(len(selectedFrames)//config.medianOver);
                temparray = np.zeros(((1+config.ROI_y[1] - config.ROI_y[0]), (1+config.ROI_x[1] - config.ROI_x[0]), config.medianOver), dtype=np.float32)
                tempcount = 1;

        end_time = time()
        end_timeP = process_time()

        outString.set('Elapsed time:' + ' ' + f'{end_time - start_time:.2f}' + ' ' + 'CPU time:' + ' ' + f'{end_timeP - start_timeP:.2f}') 

        if not os.path.isdir(os.path.join(config.basePath, config.outputPath)): os.makedirs(os.path.join(config.basePath, config.outputPath))  
        imwrite(os.path.join(config.basePath, config.outputPath, f'{len(selectedFrames)}_{config.filter}.tif'), stackFrame)

        plt.imshow(stackFrame, cmap='gray')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.axis('off')
        plt.show()
    else:
        outString.set('Missing input files.')


def selectMethod():
   if config.basePath == '':
        outString.set('Please enter a valid directory.')
   else:
        filterTuple = ('L','R','G','B','H')
        alignTuple = ('L','R','G','B','H')
        lightInputTuple = ('.png', '.tif')
        if todoSelector.get()==0:
            config.maxStars=s0.get()
            config.filter = filterTuple[filterSelector.get()]
            config.lightInputFormat = lightInputTuple[lightInputFormatSelector.get()]
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
            config.lightInputFormat = lightInputTuple[lightInputFormatSelector.get()]
            config.medianOver=s4.get()
            stackImages(config)


def selectPath():
    basePathName = filedialog.askdirectory()
    pathString.set(basePathName)
    config.basePath = basePathName


###################################################


#This is the GUI
win = Tk()

win.geometry('750x220')
win.title('HydraSTAQ')
config = Config()

todoSelector = IntVar()
filterSelector = IntVar()
alignSelector = IntVar()
lightInputFormatSelector = IntVar()
filterSelector.set(1)
alignSelector.set(1)
lightInputFormatSelector.set(0)

v0 = DoubleVar()
v1 = DoubleVar()
v2 = DoubleVar()
v3 = DoubleVar()
v4 = DoubleVar()

pathString = StringVar()
outString = StringVar()

Radiobutton(win, text='Read images', variable=todoSelector, value=0).grid(row=0, sticky='w')
Radiobutton(win, text='Compute offsets', variable=todoSelector, value=1).grid(row=1, sticky='w')
Radiobutton(win, text='Stack images', variable=todoSelector, value=2).grid(row=2, sticky='w')
Button(win, text ='Execute', command = selectMethod).grid(row=3)
Label(text='Make a choice').grid(row=4)

Label(win, text='Max stars').grid(row=0, column=1)
Label(win, text='Discard percentage').grid(row=1, column=1)
Label(win, text='Ref. align stars').grid(row=2, column=1)
Label(win, text='Align stars').grid(row=3, column=1)
Label(win, text='Median over').grid(row=4, column=1)

s0 = Scale(win, variable = v0, from_ = 5, to = 15, orient = HORIZONTAL); s0.grid(row=0, column=2); s0.set(15)
s1 = Scale(win, variable = v1, from_ = 0, to = 99, orient = HORIZONTAL); s1.grid(row=1, column=2); s1.set(10)
s2 = Scale(win, variable = v2, from_ = 4, to = 8, orient = HORIZONTAL); s2.grid(row=2, column=2); s2.set(6)
s3 = Scale(win, variable = v3, from_ = 4, to = 8, orient = HORIZONTAL); s3.grid(row=3, column=2); s3.set(6)
s4 = Scale(win, variable = v4, from_ = 10, to = 30, orient = HORIZONTAL); s4.grid(row=4, column=2); s4.set(30)

Radiobutton(win, text='Process L', variable=filterSelector, value=0).grid(row=0, column=3, sticky='w')
Radiobutton(win, text='Process R', variable=filterSelector, value=1).grid(row=1, column=3, sticky='w')
Radiobutton(win, text='Process G', variable=filterSelector, value=2).grid(row=2, column=3, sticky='w')
Radiobutton(win, text='Process B', variable=filterSelector, value=3).grid(row=3, column=3, sticky='w')
Radiobutton(win, text='Process Ha', variable=filterSelector, value=4).grid(row=4, column=3, sticky='w')

Radiobutton(win, text='Align by L', variable=alignSelector, value=0).grid(row=0, column=4, sticky='w')
Radiobutton(win, text='Align by R', variable=alignSelector, value=1).grid(row=1, column=4, sticky='w')
Radiobutton(win, text='Align by G', variable=alignSelector, value=2).grid(row=2, column=4, sticky='w')
Radiobutton(win, text='Align by B', variable=alignSelector, value=3).grid(row=3, column=4, sticky='w')
Radiobutton(win, text='Align by Ha', variable=alignSelector, value=4).grid(row=4, column=4, sticky='w')

Button(text='Select base path', command=selectPath).grid(row=0, column=5, sticky='w')
Label(master=win, textvariable=pathString).grid(row=1, column=5, sticky='w')
Radiobutton(win, text='Read PNG', variable=lightInputFormatSelector, value=0).grid(row=2, column=5, sticky='w')
Radiobutton(win, text='Read TIF', variable=lightInputFormatSelector, value=1).grid(row=3, column=5, sticky='w')
Label(master=win, textvariable=outString, fg='#f00').grid(row=4, column=5, sticky='w')

win.mainloop()