from PIL import Image
from PIL import ImageFilter

def PIL_blur_detector(input, im):
    x = input['instances']
    PIL_img = Image.fromarray(im, mode = "RGB")
    blur_img = PIL_img.filter(ImageFilter.GaussianBlur(9))
    blur_img = np.asarray(blur_img)

    #원하는 값 얻는 방법.
    mask = x.get('pred_masks').permute(1, 2, 0).to("cpu").numpy()
    num_instances = mask.shape[2]
    mask_array_instance = []

    for i in range(num_instances):
        mask_array_instance.append(mask[:, :, i:(i+1)])

        im[np.where((mask_array_instance[i] == True).all(axis=2))] = 0

    return np.where(im == 0, blur_img, im)

dataset_dicts = get_balloon_dicts("balloon/val")
sample_dataset = [dataset_dicts[0], dataset_dicts[1], dataset_dicts[2]]
