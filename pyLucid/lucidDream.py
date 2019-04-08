import cv2
import numpy as np 

def dreamData(img, gt, bgimg, consequent_frames):
    if isinstance(bgimg,basestring):
        back_img=cv2.imread(bgimg)
    else:
        back_img=bgimg.copy()
    if img.ndim==2 or img.shape[2]==1:
        img=np.dstack((img,)*3)

    object_ids=np.unique(gt)
    if object_ids[0]==0:
        object_ids=object_ids[1:]

    number_of_objects=np.random.randint(object_ids[-1])+1
    mask_object_ids=np.random.choice(object_ids,number_of_objects,replace=False)
    mask_object_ids=np.sort(mask_object_ids)
    mask=gt.copy()
    mask[np.isin(mask,mask_object_ids,invert=True)]=0

    org_back_img=back_img.copy()
    seg=gt.copy()

    if np.random.randint(2):
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilmsk=cv2.dilate(mask,kernel)
        back_img[dilmsk==0]=img[dilmsk==0]
    else:
        seg[mask==0]=0
    
    seg_backgr=np.zeros(seg.shape,dtype='uint8')
    seg_backgr[mask==0]=seg[mask==0]

    if np.random.randint(2):
        back_img=np.fliplr(back_img)
        org_back_img=np.fliplr(org_back_img)
        seg_backgr=np.fliplr(seg_backgr)
    
    if np.random.randint(2):
        back_img,org_back_img=change_illumination(back_img,org_back_img)

    if np.random.randint(2):
        new_img,new_msk=spline_transform_multi(img,mask)
        while np.all(seg_backgr==0) and np.all(new_msk==0):
            print 'All objects are missed, so we have to retry random transform.'
            new_img,new_msk=spline_transform_multi(img,mask)
        new_img,new_seg=blend_mask_multi(seg_backgr,new_img,back_img,new_msk)
    else:
        new_img,new_seg=blend_mask_multi(seg_backgr,img,back_img,mask)

    im_1,gt_1,bb_1,bg_1=augment_image_mask_illumination_deform_random_img_multi(new_img,new_seg,org_back_img)

    if consequent_frames:
        object_ids=np.unique(gt_1)
        if object_ids[0]==0:
            object_ids=object_ids[1:]
        number_of_objects=np.random.randint(object_ids.size)+1
        mask_object_ids_fg=np.random.choice(object_ids,number_of_objects,replace=False)
        mask_object_ids_fg=np.sort(mask_object_ids_fg)
        mask_fg=gt_1.copy()
        mask_fg[np.isin(gt_1,mask_object_ids_fg,invert=True)]=0
        mask_bg=gt_1.copy()
        mask_bg[mask_fg>0]=0

        back_img_obj_1=bg_1.copy()
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilmsk=cv2.dilate(mask_fg,kernel)
        back_img_obj_1[dilmsk==0]=im_1[dilmsk==0]

        transf_id=np.random.randint(3)
        if transf_id<2:
            fogr_id=np.random.randint(3)
            fogr_var=np.array([[1,1],[1,0],[0,1]])
            fogr=fogr_var[fogr_id]
        if transf_id==0:
            back_img_2,back_gt_2,F_Flowb,B_Flowb=augment_background(back_img_obj_1,mask_bg,bg_1)
            new_img2,new_seg2,new_bb2,F_Flowf,B_Flowf=augment_foreground(im_1,mask_fg,back_img_2,back_gt_2,fogr[0],fogr[1])

            mask_fg2=new_seg2.copy()
            mask_fg2[np.isin(new_seg2,mask_object_ids_fg,invert=True)]=0
            B_Flowf[mask_fg2<=0]=0
            B_Flowb[mask_fg2>0]=0
            Flow_B=B_Flowf+B_Flowb
            F_Flowf[mask_fg<=0]=0
            F_Flowb[mask_fg>0]=0
            Flow_F=F_Flowf+F_Flowb
        elif transf_id==1:
            new_img2,new_seg2,new_bb2,F_Flowf,B_Flowf=augment_foreground(im_1,mask_fg,back_img_obj_1,mask_bg,fogr[0],fogr[1])

            mask_fg2=new_seg2.copy()
            mask_fg2[np.isin(new_seg2,mask_object_ids_fg,invert=True)]=0
            B_Flowf[mask_fg2<=0]=0
            Flow_B=B_Flowf
            F_Flowf[mask_fg<=0]=0
            Flow_F=F_Flowf
        else:
            back_img_2,back_gt_2,F_Flowb,B_Flowb=augment_background(back_img_obj_1,mask_bg,bg_1)
            new_img2,new_seg2, new_bb2, F_Flowf, B_Flowf=augment_foreground(im_1,mask_fg,back_img_2,back_gt_2,0,0)

            mask_fg2=new_seg2.copy()
            mask_fg2[np.isin(new_seg2,mask_object_ids_fg,invert=True)]=0
            B_Flowb[mask_fg2>0]=0
            Flow_B=B_Flowb
            F_Flowb[mask_fg>0]=0
            Flow_F=F_Flowb
        return im_1,gt_1,bb_1, new_img2, new_seg2, new_bb2, Flow_B, Flow_F
    else:
        return im_1,gt_1,bb_1
    

def change_illumination(img,bgimg=None):

    img=img.astype('float32')/255
    HSVimg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if bgimg is not None:
        bgimg=bgimg.astype('float32')/255
        HSVbg=cv2.cvtColor(bgimg,cv2.COLOR_BGR2HSV)
    else:
        HSVbg=None
    
    mult_l,multi_r=0.97,1.03
    power_l,power_r=0.8,1.2
    add_l,add_r=-0.05,0.05

    multi_val=np.random.rand()*(multi_r-mult_l)+mult_l
    power_val=np.random.rand()*(power_r-power_l)+power_l
    add_val=np.random.rand()*(add_r-add_l)+add_l
    S=multi_val*HSVimg[:,:,1]**power_val+add_val
    S[S<0]=0
    S[S>1]=1
    if HSVbg is not None:
        Sbg=multi_val*HSVbg[:,:,1]**power_val+add_val
        Sbg[Sbg<0]=0
        Sbg[Sbg>1]=1
    
    multi_val=np.random.rand()*(multi_r-mult_l)+mult_l
    power_val=np.random.rand()*(power_r-power_l)+power_l
    add_val=np.random.rand()*(add_r-add_l)+add_l
    V=multi_val*HSVimg[:,:,2]**power_val+add_val
    V[V<0]=0
    V[V>1]=1
    if HSVbg is not None:
        Vbg=multi_val*HSVbg[:,:,2]**power_val+add_val
        Vbg[Vbg<0]=0
        Vbg[Vbg>1]=1

    newHSV=np.dstack((HSVimg[:,:,0],S,V))
    newBGR=cv2.cvtColor(newHSV,cv2.COLOR_HSV2BGR)
    newBGR=(newBGR*255+0.5).astype('uint8')
    if HSVbg is not None:
        newHSVbg=np.dstack((HSVbg[:,:,0],Sbg,Vbg))
        newbg=cv2.cvtColor(newHSVbg,cv2.COLOR_HSV2BGR)
        newbg=(newbg*255+0.5).astype('uint8')
        return newBGR,newbg
    else:
        return newBGR

def spline_transform_multi(img, mask):
    bimask=mask>0
    M,N=np.where(bimask)
    w=np.ptp(N)+1
    h=np.ptp(M)+1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bound=cv2.dilate(bimask.astype('uint8'),kernel)-bimask
    y,x=np.where(bound>0)

    if x.size>4:
        newxy=thin_plate_transform(x,y,w,h,mask.shape[:2],num_points=5)

        new_img=cv2.remap(img,newxy,None,cv2.INTER_LINEAR)
        new_msk=cv2.remap(mask,newxy,None,cv2.INTER_NEAREST)
    elif x.size>0:
        new_img=img
        new_msk=mask
    return new_img,new_msk

def blend_mask_multi(seg_or, img, back_img, mask):
    if img.ndim==2 or img.shape[2]==1:
        img=np.dstack((img,)*3)
    
    M,N=np.where(mask>0)
    if M.size==0:
        return back_img,seg_or
    topM,bottomM=M.min(),M.max()
    leftN,rightN=N.min(),N.max()
    msk2=mask[topM:bottomM+1,leftN:rightN+1]
    img2=img[topM:bottomM+1,leftN:rightN+1,:]

    scale_l=0.85
    scale_r=1.15
    scaleval=np.random.rand()*(scale_r-scale_l)+scale_l

    ih=min(max(int(round(msk2.shape[0]*scaleval)),100),back_img.shape[0]-50)
    iw=min(max(int(round(msk2.shape[1]*scaleval)),100),back_img.shape[1]-50)
    imMask=cv2.resize(msk2,(iw,ih),interpolation=cv2.INTER_NEAREST)
    img2=cv2.resize(img2,(iw,ih))

    max_offY=back_img.shape[0]-imMask.shape[0]
    max_offX=back_img.shape[1]-imMask.shape[1]
    offY=np.random.randint(max_offY+1)
    offX=np.random.randint(max_offX+1)

    clonedIm2,new_msk=SeamlessClone_trimap(img2,back_img,imMask,offX,offY)

    new_msk2=seg_or.copy()
    new_msk2[new_msk>0]=new_msk[new_msk>0]

    return clonedIm2,new_msk2

def SeamlessClone_trimap(srcIm,dstIm,imMask,offX,offY):
    dstIm=dstIm.copy()
    bimsk=imMask>0

    new_msk=np.zeros(dstIm.shape[:2],dtype='uint8')
    new_msk[offY:offY+imMask.shape[0],offX:offX+imMask.shape[1]]=imMask

    dstIm[new_msk>0]=srcIm[imMask>0]

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bimsk=bimsk.astype('uint8')
    bdmsk=cv2.dilate(bimsk,kernel)-cv2.erode(bimsk,kernel)
    mask255=bdmsk>0
    mask255=(mask255*255).astype('uint8')

    offCenter=(offX+imMask.shape[1]/2,offY+imMask.shape[0]/2)

    if np.any(bdmsk>0):
        outputIm=cv2.seamlessClone(srcIm,dstIm,mask255,offCenter,cv2.MIXED_CLONE)
    else:
        outputIm=dstIm
        #when one object have very few pixels, bdmsk will be totally zero, which will cause segmentation fault.

    return outputIm,new_msk

def augment_image_mask_illumination_deform_random_img_multi(im0,gt0,bg0=None):
    illumination,flip,rotate=np.random.randint(2,size=3)

    resize=True

    im_dim1,im_dim2=im0.shape[:2]
    angle=np.random.randint(-15,16)*2

    bimask=gt0>0
    M,N=np.where(bimask)
    w=np.ptp(N)+1
    h=np.ptp(M)+1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bound=cv2.dilate(bimask.astype('uint8'),kernel)-bimask
    y,x=np.where(bound>0)

    shift_l=-0.05
    shift_r=0.05
    #bb1=gt0.copy()
    bb1=gt0

    if x.size>4:
        newxy=thin_plate_transform(x,y,w,h,gt0.shape[:2],num_points=5)

        bb1=cv2.remap(bb1,newxy,None,cv2.INTER_NEAREST)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        bb1=cv2.dilate(bb1,kernel)

    im1_rot_crop=im0
    gt_rot_crop=gt0
    bb1_rot_crop=bb1
    bg1_rot_crop=bg0

    if rotate:
        gt_rot_crop=rotate_image(gt0,angle,cv2.INTER_NEAREST)
        while np.all(gt_rot_crop==0):
            print 'After rotating, the objects are missed. So we have to try a different angle.'
            angle=np.random.randint(-15,16)*2
            gt_rot_crop=rotate_image(gt0,angle,cv2.INTER_NEAREST)
        im1_rot_crop=rotate_image(im0,angle,cv2.INTER_CUBIC)
        bb1_rot_crop=rotate_image(bb1,angle,cv2.INTER_NEAREST)
        if bg0 is not None:
            bg1_rot_crop=rotate_image(bg0,angle,cv2.INTER_CUBIC)
        if resize:
            im1_rot_crop=cv2.resize(im1_rot_crop,(im_dim2,im_dim1),interpolation=cv2.INTER_CUBIC)
            gt_rot_crop=cv2.resize(gt_rot_crop,(im_dim2,im_dim1),interpolation=cv2.INTER_NEAREST)
            bb1_rot_crop=cv2.resize(bb1_rot_crop,(im_dim2,im_dim1),interpolation=cv2.INTER_NEAREST)
            if bg1_rot_crop is not None:
                bg1_rot_crop=cv2.resize(bg1_rot_crop,(im_dim2,im_dim1),interpolation=cv2.INTER_CUBIC)
    
    if flip:
        im1_rot_crop=np.fliplr(im1_rot_crop)
        gt_rot_crop=np.fliplr(gt_rot_crop)
        bb1_rot_crop=np.fliplr(bb1_rot_crop)
        if bg1_rot_crop is not None:
            bg1_rot_crop=np.fliplr(bg1_rot_crop)
    
    if illumination:
        im1_rot_crop=change_illumination(im1_rot_crop,bg1_rot_crop)
        if bg1_rot_crop is not None:
            im1_rot_crop,bg1_rot_crop=im1_rot_crop
    
    if bg1_rot_crop is not None:
        return im1_rot_crop,gt_rot_crop,bb1_rot_crop,bg1_rot_crop
    else:
        return im1_rot_crop,gt_rot_crop,bb1_rot_crop

def rotate_image(image, angle,interp=cv2.INTER_LINEAR):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]

    y_coords = [pt[1] for pt in rotated_coords]

    left_bound=min(x_coords)
    right_bound=max(x_coords)
    top_bound=min(y_coords)
    bot_bound=max(y_coords)

    new_w = int(right_bound - left_bound)
    new_h = int(bot_bound - top_bound)

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=interp
    )

    wr,hr=rotatedRectWithMaxArea(image_size[0],image_size[1],np.pi*angle/180)

    result=crop_around_center(result,wr,hr)

    return result

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    width=int(width)
    height=int(height)
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1=image_center[0]-width/2
    x2=x1+width
    y1=image_center[1]-height/2
    y2=y1+height

    return image[y1:y2, x1:x2]

def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

def augment_background(img, gt, back_img):
    padsz=[int(img.shape[0]/2.+0.5),int(img.shape[1]/2.+0.5)]
    padwidth=[(sz,sz) for sz in padsz]
    padwidth.append((0,0))
    padded_image=np.pad(img,padwidth,'symmetric')
    padded_bimage=np.pad(back_img,padwidth,'symmetric')

    angle=np.random.randint(-3,4)
    angle_param=np.pi*angle/180

    scale_l=0.95
    scale_r=1.05
    scale_x=np.random.rand()*(scale_r-scale_l)+scale_l
    scale_y=np.random.rand()*(scale_r-scale_l)+scale_l

    h,w=img.shape[:2]
    shift_x=int(0.05*w+0.5)
    shift_y=int(0.05*h+0.5)
    tx=np.random.randint(-shift_x,shift_x+1)
    ty=np.random.randint(-shift_y,shift_y+1)
    
    sc=scale_x*np.cos(angle_param)
    ss=scale_y*np.sin(angle_param)

    #opencv use transpose transformation matrix.
    T=np.array([[sc,ss,tx],[-ss,sc,ty],[0,0,1]])
    Tshift=np.array([[1,0,-padded_image.shape[1]/2],[0,1,-padded_image.shape[0]/2],[0,0,1]])
    Tshift0=np.array([[1,0,-img.shape[1]/2],[0,1,-img.shape[0]/2],[0,0,1]])
    Tshift1=np.array([[1,0,img.shape[1]/2],[0,1,img.shape[0]/2],[0,0,1]])
    
    bgT=Tshift1.dot(T).dot(Tshift)
    bg_flowT=Tshift1.dot(T).dot(Tshift0)

    transformed_image=cv2.warpAffine(padded_image,bgT[:2],(img.shape[1],img.shape[0]))
    transformed_bimage=cv2.warpAffine(padded_bimage,bgT[:2],(back_img.shape[1],back_img.shape[0]))

    transformed_mask=cv2.warpAffine(gt+1,bg_flowT[:2],(gt.shape[1],gt.shape[0]),flags=cv2.INTER_NEAREST)
    transformed_image[transformed_mask==0]=transformed_bimage[transformed_mask==0]
    transformed_mask[transformed_mask>0]-=1

    F_Flow=transformPointsForward(bg_flowT[:2],gt.shape[1],gt.shape[0])
    F_Flow=np.dstack(F_Flow)

    B_Flow=transformPointsInverse(bg_flowT[:2],gt.shape[1],gt.shape[0])
    B_Flow=np.dstack(B_Flow)

    return transformed_image,transformed_mask,F_Flow,B_Flow

def transformPointsForward(T,width,height):
    x,y=np.meshgrid(np.arange(width),np.arange(height))
    xymat=np.vstack((x.ravel(),y.ravel(),np.ones(x.size)))
    newxy=T.dot(xymat)
    newxy[0]-=x.ravel()
    newxy[1]-=y.ravel()
    return newxy[0].reshape([height,width]),newxy[1].reshape([height,width])

def transformPointsInverse(T,width,height):
    T=cv2.invertAffineTransform(T)
    return transformPointsForward(T,width,height)

def augment_foreground(img, seg, back_img, mask_bgr, ifshift, iftransform ):

    imh,imw=seg.shape[:2]
    transform_matrix=np.zeros([imh,imw,2])
    new_seg=seg.copy()
    new_img=img.copy()
    bimask=seg>0
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bound=cv2.dilate(bimask.astype('uint8'),kernel)-bimask
    if iftransform:
        
        M,N=np.where(bimask)
        w=np.ptp(N)+1
        h=np.ptp(M)+1
        y,x=np.where(bound>0)

        shift_l=-0.05
        shift_r=0.05

        if x.size>4:
            newxy,transform_matrix=thin_plate_transform(x,y,w,h,seg.shape[:2],num_points=5,offsetMatrix=True)

            new_img=cv2.remap(img,newxy,None,cv2.INTER_LINEAR)
            new_seg=cv2.remap(seg,newxy,None,cv2.INTER_NEAREST)

    X_transb=transform_matrix[:,:,0]
    Y_transb=transform_matrix[:,:,1]

    new_img2,new_seg2=blend_mask_transf(new_seg, new_img, back_img, ifshift)

    y,x=np.where(new_seg>0)
    y2,x2=np.where(new_seg2>0)

    X_trans2=np.zeros(seg.shape,dtype='float32')
    Y_trans2=np.zeros(seg.shape,dtype='float32')
    X_trans2[seg>0]=-X_transb[seg>0]
    Y_trans2[seg>0]=-Y_transb[seg>0]

    if x.size==0 or x2.size==0:
        Tx=X_trans2
        Ty=Y_trans2
    else:
        Tx=X_trans2+(seg>0)*(x2.min()-x.min())
        Ty=Y_trans2+(seg>0)*(y2.min()-y.min())
    
    X_transb2=np.zeros(seg.shape,dtype='float32')
    Y_transb2=np.zeros(seg.shape,dtype='float32')
    X_transb2[new_seg2>0]=X_transb[new_seg>0]
    Y_transb2[new_seg2>0]=Y_transb[new_seg>0]
    
    if x.size==0 or x2.size==0:
        Txb=X_transb2
        Tyb=Y_transb2
    else:
        Txb=X_transb2-(new_seg2>0)*(x2.min()-x.min())
        Tyb=Y_transb2-(new_seg2>0)*(y2.min()-y.min())
    
    Flow=np.dstack((Tx,Ty))
    Flowb=np.dstack((Txb,Tyb))

    new_seg=mask_bgr.copy()
    new_seg[new_seg2>0]=new_seg2[new_seg2>0]
    new_seg2=new_seg

    seg=new_seg2
    M,N=np.where(seg>0)
    if M.size==0:
        return new_img2,new_seg2,seg,Flow,Flowb
    w=np.ptp(N)+1
    h=np.ptp(M)+1
    bb1=seg
    y,x=np.where(bound>0)

    if x.size>4:
        newxy=thin_plate_transform(x,y,w,h,seg.shape[:2],num_points=5)
        
        bb1=cv2.remap(seg,newxy,None,cv2.INTER_NEAREST)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        bb1=cv2.dilate(bb1,kernel)

    return new_img2,new_seg2, bb1, Flow, Flowb

def blend_mask_transf(seg, img, back_img, ifshift):
    M,N=np.where(seg>0)
    if M.size==0:
        return back_img,seg
    topM=M.min()
    bottomM=M.max()
    leftN=N.min()
    rightN=N.max()
    seg2=seg[topM:bottomM+1,leftN:rightN+1]
    img2=img[topM:bottomM+1,leftN:rightN+1]
    imMask=seg2

    min_offsetY=max(-int(0.05*imMask.shape[0]),-topM)
    max_offsetY=min(int(0.05*imMask.shape[0]),back_img.shape[0]-bottomM-1)
    min_offsetX=max(-int(0.05*imMask.shape[1]),-leftN)
    max_offsetX=min(int(0.05*imMask.shape[1]),back_img.shape[1]-rightN-1)

    if max_offsetY-min_offsetY<0 or not ifshift:
        offsetY=topM
    else:
        offsetY=topM+np.random.randint(min_offsetY,max_offsetY+1)
    
    if max_offsetX-min_offsetX<0 or not ifshift:
        offsetX=leftN
    else:
        offsetX=leftN+np.random.randint(min_offsetX,max_offsetX+1)
    
    clonedIm,clonedSeg=SeamlessClone_trimap(img2,back_img,imMask,offsetX,offsetY)

    return clonedIm,clonedSeg

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
    rand_p=np.random.choice(x.size,num_points,replace=False)
    movingPoints=np.zeros((1,num_points,2),dtype='float32')
    fixedPoints=np.zeros((1,num_points,2),dtype='float32')

    movingPoints[:,:,0]=x[rand_p]
    movingPoints[:,:,1]=y[rand_p]
    fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
    fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

    tps=cv2.createThinPlateSplineShapeTransformer()
    good_matches=[cv2.DMatch(i,i,0) for i in xrange(num_points)]
    tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

    imh,imw=imshape
    x,y=np.meshgrid(np.arange(imw),np.arange(imh))
    x,y=x.astype('float32'),y.astype('float32')
    newxy=tps.applyTransformation(np.dstack((x.ravel(),y.ravel())))[1]
    newxy=newxy.reshape([imh,imw,2])

    if offsetMatrix:
        return newxy,newxy-np.dstack((x,y))
    else:
        return newxy