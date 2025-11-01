# Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline
This Python pipeline automatically generates seamless, cloud-free mosaics from multisensor satellite imagery. It integrates cloud masking, seamline optimization, and radiometric blending into a fast, reproducible workflow for large-scale or time-series processing.

Official code for: "**A Computational Pipeline for Automated Cloud Removal and Seamless Multisensor Satellite Image Mosaicking**" (Hsiao-Jou Hsu, et al.).

# Key Features
- **Optimal Seamline Generation**: Implements a **sparse-node Dijkstra's algorithm** to find the "cheapest" seamline along natural features, minimizing visible edges.
    - This algorithm is key to achieving seamless results.
- **Geometric-Aware**: Automatically calculates the true geometric intersection of overlapping, irregularly shaped image swaths to find the optimal start and end points for the seamline.
    - This ensures the seamline is always calculated only within the common data area.
- **Automated Blending**: Applies **Poisson blending** along the generated seamline for a smooth, gradual transition between adjacent tiles.

- **Iterative & Scalable**: Sorts images by spatial proximity and iteratively mosaics each image onto a growing base map (MosTai.tif), allowing it to scale to hundreds of images.

# Example: The Mosaicking Process
The core of this pipeline is its ability to find an optimal seamline and blend two images. As demonstrated in the manuscript (e.g., Figures 7, 11, 12), the algorithm finds complex paths along roads, rivers, and valleys to hide the seam.

**Installation**

1. **Clone the repository**:
``` bash
git clone https://github.com/Amy-Hsu/Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline.git
cd Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline
```

2. **Install dependencies**:We recommend using a Conda environment to manage the complex geospatial libraries.
``` bash
# Create a new conda environment
conda create -n mosaic python=3.9
conda activate mosaic

# Install dependencies (GDAL is best installed via conda-forge)
conda install -c conda-forge gdal rasterio geopandas shapely

# Install remaining packages with pip
pip install -r requirements.txt
``` 

# Configuration (para.txt)
Before running, you must edit the ``` bash para.txt ``` file to match your data and environment.
``` bash
{
    "path_method": "a6c",
    "res": "6",
    "pan_res": "1.5",
    "resampleTo": "5",
    "trueBits": "12",
    "imgPath": "E:\\paper",
    "kernel": "61"
}
``` 
```bash path_method``` : Weighting algorithm for seamline cost. `a6c` and ```a7c``` are supported (see ```seamline.py:buildGraph```).

```bash res``` : The target output resolution for the mosaic (e.g., "6" for 6 meters).

```bash pan_res``` : The panchromatic resolution of the input imagery. Used for scaling pan-sharpened data.

```bash resampleTo``` : The downsampling factor for the seamline search. ```5``` means the search is run on an image 1/5th the original resolution, which is much faster.

```bash trueBits``` : The bit-depth of your input images (e.g., "12" for 12-bit).

```bash imgPath``` : (**Critical**) The absolute path to the folder containing your input ```.img``` files. **You must change this value!**

```bash kernel``` : The size of the kernel used to erode the edges of images, preventing no-data artifacts.

# Workflow Steps (Usage)

1. **Prepare Data**:
    - Place all your georeferenced, orthorectified ```.img``` files (e.g., SPOT, Formosat) into the folder specified in ```imgPath```.
    - (For Cloud Masking): Place your corresponding cloud mask ```.tif``` files in a folder (e.g., ```SPOT67Cloud_Mask```). You may need to update this hardcoded path in ```main.py```.

2. **Configure**:
    - Edit ```para.txt``` to point to your ```imgPath``` and set the correct resolutions and parameters for your data.

3. **Run the Pipeline**:
    - Execute the main script from your terminal:
    ```python main.py```

4. **Review Outputs**:
    - The script will iteratively build the mosaic. You can monitor its progress in the console.
    - Final Mosaic: ```MosTai.tif```
    - Final Cloud Mask: ```TaiMas.tif``` (a mosaic of all cloud masks)
    - Footprint: ```res_union_new.shp``` (a shapefile of the final mosaic's footprint)
    - Intermediate Seamlines: ```*_sl.img``` files are saved for each step.

# How It Works (Internal Logic)
The ```main.py``` script follows this logical flow:

1. **Load & Sort**: Loads ```para.txt```. Finds all ```.img``` files in imgPath and sorts them by their Euclidean distance from the first image, ensuring an orderly, tile-by-tile mosaicking process.

2. **Initialize Mosaic**:
    - It processes the first two images (```ReadOrder[0]``` and ```ReadOrder[1]```) to create the initial ```MosTai.tif``` and ```TaiMas.tif``` base files.

3. **Find Overlap**:
    - Calculates the precise geometric intersection of the two images.
    - Reads the overlapping pixel data from both images.

4. **Find Seamline**:
    - Downsamples the overlap region by the ```resampleTo``` factor.
    - Builds a weighted graph (```networkx.Graph```) where each pixel is a node and edge weights are based on the pixel value difference (the "cost" of a seam).
    - Finds the cheapest path (the seamline) between the two geometric intersection points using the Dijkstra algorithm.

5. **Generate Mask & Blend**:
    - The 1D seamline path is converted into a 2D mask.
    - ```cv2.floodFill``` is used to create a binary mask separating "Image A" from "Image B".
    - This mask is blurred (```cv2.GaussianBlur****) to create a "feathered" blend zone.
    - The two images are blended using this feathered mask (e.g., ```ImgA * (1.0 - mask) + ImgB * mask```).

6. **Write & Iterate**:
    - The blended result is written back into the ```MosTai.tif``` file in the correct location.
    - The script loops to the next image (```ReadOrder[2]```), treating the entire ```MosTai.tif``` as "Image A" and the new image as "Image B".This process repeats until all images are mosaicked.

# Citation
If you use this code or methodology in your research, please cite our paper:
```(Placeholder - Add your full paper citation here once it is published)
Hsu, H.J., Tseng, K.H., Tsai, F., et al. (2025). "A Computational Pipeline for Automated Cloud Removal and Seamless Multisensor Satellite Image Mosaicking." [Journal Name], [Volume], [Pages].```
