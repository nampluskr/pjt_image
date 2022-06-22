# Projet

## [Step-1] `save_bboxes.py`

```python
def save_bboxes_csv(filename, img_paths, img_angles, img_pattern):
    bboxes = []
    pbar = tqdm(zip(img_paths, img_angles), leave=True)

    for img_path, img_angles in pbar:
        pbar.set_description(os.path.basename(img_path))
        img = get_image(img_path, img_angle)
        pixels, _ = get_pixels(img, img_pattern)

        for i_pixel in range(1, len(pixels) + 1):
            lanes = get_lanes(img, pixels[i_pixel])

            for i_lane in range(1, len(lanes) + 1):
                rods = get_rods(img, lanes[i_lane])

                for i_rod, (x, y, w, h) in enumerate(rods):
                    bboxes.append([img_path, img_angle, i_img, 
                                i_pixel, i_lane, i_rod, x, y, w, h])

        df = pd.DataFrame(data=bboxes, columns=['img_path', 'img_angle', 'i_img',
                    'i_pixel', 'i_lane', 'i_rod', 'x', 'y', 'w', 'h'])

    df.to_csv(filename, index=False)
    print(">> ", os.path.basename(filename))
```

```python
result_dir = "d:\\results"
img_dir = "d:\\images"
folder_names = ['folder_1', 'folder_2']
folder_angles = {'folder_1': 10, 'folder_2':20}
PATTERN = np.genfromtxt("pattern.txt")

for folder_names in folder_names:
    img_paths, img_angles = [], []
    for folder_name in folder_names:
        img_list = glob(img_dir, folder_name, '*.JPG')
        img_paths += img_list
        img_angles += [folder_angles[folder_name]*len(img_list)]
    
    filename = os.path.join(result_dir, "bbpxes_%s.csv" % folder_name)
    save_bboxes_csv(filename, img_paths, img_angles, PATTERN)
```

## [Step-2] `save_data.py`

```python

```
