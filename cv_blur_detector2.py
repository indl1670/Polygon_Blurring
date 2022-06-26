def direct_cv_blur_detector2(input, im):
    x = input['instances']
    blur_img = cv2.GaussianBlur(im, (0, 0), 9)

    #원하는 값 얻는 방법.
    mask = x.get('pred_masks').permute(1, 2, 0).to("cpu").numpy()
    num_instances = mask.shape[2]
    mask_array_instance = []

    for i in range(num_instances):
        mask_array_instance.append(mask[:, :, i:(i+1)])
        img = np.where(mask_array_instance[i] == True, 0, 1)
        array_img = np.asarray(img)

        im[np.where((array_img==0).all(axis=2))]= 0 
      
    return np.where(im == 0, blur_img, im)
