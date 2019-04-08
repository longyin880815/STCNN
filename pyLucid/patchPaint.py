import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PatchMatch.PatchMatchCuda import PatchMatch

def paint(Iorg, Mask, verbose=True, sigma=0.1):
    Iorg=cv2.cvtColor(Iorg,cv2.COLOR_BGR2Lab)

    width=7
    match_iter=10
    
    diffthresh=1

    if width%2==0:
        raise Exception('The width should be an odd integer.')
    padwidth=width/2

    if Mask.ndim!=2:
        if Mask.ndim==3 and Mask.shape[2]==1:
            Mask=Mask[:,:,0]
        else:
            raise Exception('The dimension of Mask is incorrect.')
    
    [m,n,chn]=Iorg.shape
    startscale=int(-np.ceil(np.log2(min(m,n)))+5)
    scale=2**startscale
    
    I=cv2.resize(Iorg,(0,0),fx=scale,fy=scale)
    M=cv2.resize(Mask,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    M=M>0
    
    [m,n,chn]=I.shape
    Rnd=np.random.randint(256,size=[m,n,chn],dtype='uint8')
    I[M]=Rnd[M]

    for logscale in xrange(startscale,1):
        scale=2**logscale
        iterations=10

        if verbose:
            print 'Scale = 2^%d'%logscale
        
        for iter in xrange(iterations):
            if verbose:
                plt.imshow(cv2.cvtColor(I,cv2.COLOR_Lab2RGB))
                plt.pause(0.001)

            Iprev=I.astype('float32')
            I=Iprev/255
        
            B=I.copy()
            B[M]=0

            maxoff=max(I.shape[:2])
            pm=PatchMatch(I,I,B,B,width)
            pm.propagate(iters=match_iter,rand_search_radius=maxoff)
            ann=pm.nnf.copy()
            Ipad=np.pad(I,((padwidth,padwidth),(padwidth,padwidth),(0,0)),'reflect')

            patchj,patchi=np.meshgrid(np.arange(width),np.arange(width))
            indj,indi=np.meshgrid(np.arange(n),np.arange(m))
            patchi=indi[:,:,np.newaxis,np.newaxis]+patchi[np.newaxis,np.newaxis]
            patchj=indj[:,:,np.newaxis,np.newaxis]+patchj[np.newaxis,np.newaxis]

            matchj=ann[:,:,0]
            matchi=ann[:,:,1]
            orgind=np.vstack((indi.ravel(),indj.ravel()))
            matchind=np.vstack((matchi[orgind[0],orgind[1]],matchj[orgind[0],orgind[1]]))
            indmap=np.vstack((orgind,matchind))
            indmap=indmap[:,M[orgind[0],orgind[1]]]

            curi=patchi[indmap[0],indmap[1]]
            curj=patchj[indmap[0],indmap[1]]
            orgim=Ipad[curi,curj]
            groupind=np.ravel_multi_index((curi,curj),(m+width-1,n+width-1))

            curi=patchi[indmap[2],indmap[3]]
            curj=patchj[indmap[2],indmap[3]]
            patchim=Ipad[curi,curj]

            #I 3 channels
            d=np.sum((orgim-patchim)**2,axis=(1,2,3))
            sim=np.exp(-d/(2*sigma**2),dtype='float64')

            R=sim[:,np.newaxis,np.newaxis,np.newaxis]*patchim
            sumpatch=[np.bincount(groupind.ravel(),weights=R[...,i].ravel()) for i in xrange(chn)]
            Rlst=[np.zeros([m+width-1,n+width-1],dtype='float64') for _ in xrange(chn)]
            for i in xrange(chn):
                Rlst[i].ravel()[:sumpatch[i].size]=sumpatch[i]
            R=np.dstack(Rlst)

            sim=np.tile(sim[:,np.newaxis,np.newaxis],[1,width,width])
            sumsim=np.bincount(groupind.ravel(),weights=sim.ravel())
            Rcount=np.zeros([m+width-1,n+width-1],dtype='float64')
            Rcount.ravel()[:sumsim.size]=sumsim

            Rcountmsk=Rcount>0
            R[Rcountmsk]=R[Rcountmsk]/Rcount[Rcountmsk,np.newaxis]
            R=R[padwidth:m+padwidth,padwidth:n+padwidth]
            R[~M]=I[~M]
            I=(255*R+0.5).astype('uint8')

            if iter>0:
                diff=np.sum((I.astype('float32')-Iprev)**2)/np.sum(M)
                if verbose:
                    print 'diff = %f'%diff
                if diff<diffthresh:
                    break
            elif verbose:
                print
        
        if logscale<0:
            Idata=cv2.resize(Iorg,(0,0),fx=scale*2,fy=scale*2)
            m,n,chn=Idata.shape
            I=cv2.resize(I,(n,m))

            M=cv2.resize(Mask,(n,m),interpolation=cv2.INTER_NEAREST)
            M=M>0
            
            I[~M]=Idata[~M]
    
    return cv2.cvtColor(I,cv2.COLOR_Lab2BGR)