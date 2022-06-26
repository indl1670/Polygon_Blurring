def direct_cv_blur_detector(input, im):
    x = input['instances']
    blur_img = cv2.GaussianBlur(im, (0, 0), 9)

    #원하는 값 얻는 방법.
    mask = x.get('pred_masks').to('cpu').numpy()
    num_instances = mask.shape[0]
    #labels = x.get('pred_classes').to('cpu').numpy()

    mask_arr = np.moveaxis(mask, 0, -1)
    mask_array_instance = []

    #마스크만 있는 이미지 생성
    #only_mask = im

    for i in range(num_instances):
        #img = np.zeros_like(im)
        mask_array_instance.append(mask_arr[:, :, i:(i+1)])
        img = np.where(mask_array_instance[i] == True, 255, im)
        array_img = np.asarray(img)
        index = np.where((array_img==[255,255,255]).all(axis=2))
        im[index]= 0 #blur_img[index]
      
    im = np.where(im == 0, blur_img, im)

    return im
