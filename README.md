Python implementation of Xu et al., Structure Extraction from Texture via Relative Total Variation (https://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/texturesep12.pdf)

Structure extraction by penalizing relative total variation.

## Example Usage
`--lambda_S` controls strength of RTV penalty.
```
python RTV.py --image demo_images/graffiti.jpg --save_path demo_outputs/graffiti.png --lambda_S .015
python RTV.py --image demo_images/Bishapur_zan.jpg --save_path demo_outputs/Bishapur_zan.png
python RTV.py --image demo_images/mosaicfloor.jpg --save_path demo_outputs/mosaicfloor.png --lambda_S 0.01
python RTV.py --image demo_images/risk.jpg --save_path demo_outputs/risk.png --lambda_S 0.01
python RTV.py --image demo_images/crossstitch.jpg --save_path demo_outputs/crossstitch.png --lambda_S 0.01
```

## Results
Input image, Original Paper Result, Reimplementation Result
<img width="1894" height="749" alt="demo" src="https://github.com/user-attachments/assets/e248d877-4295-4550-b70a-cf8a2731bbde" />

More results:

<img width="400" height="433" alt="mosaicfloor" src="https://github.com/user-attachments/assets/9191b149-f3c9-421d-a689-ff0a770fe981" /> <img width="400" height="433" alt="mosaicfloor" src="https://github.com/user-attachments/assets/57b9494c-7f81-4837-8aa9-ccb366cdd6c9" />

<img width="400" height="300" alt="graffiti" src="https://github.com/user-attachments/assets/f88cf420-c3e5-4a46-ad95-139671c2458e" /> <img width="400" height="300" alt="graffiti" src="https://github.com/user-attachments/assets/eef163d6-d1e7-4cb1-aea7-0e526c4a47ba" />

