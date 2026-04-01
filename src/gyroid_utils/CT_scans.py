import numpy as np
import pydicom
from pathlib import Path
from .logger import logger
import sys
import SimpleITK as sitk
import tifffile as tiff
from .CT_visualization_window import open_window
from gyroid_utils import CT_visualization_window


"""
#=====================================================================================================================
0 - (reserved)
1 - convert_dicomm_to_mhd
2 - _loading_bar
3 - convert_tiff_to_mhd
4 - read_mhd_file
5 - crop_images
6 - segment_from_threshold
7 - apply_threshold
8 - dilate_filter
9 - erode_filter
10 - connected_filter
11 - find_small_holes
12 - find_islands
#=====================================================================================================================
"""

"""
GENERAL NOTE ON FUNCTION DESIGN:
1) All functions should be designed to be as modular and reusable as possible.
2) Function should try to only output numpy arrays or simple data structures
3) functions should to take into input either numpy arrays or sitk images
"""


# =====================================================================
# 1) keep_largest_connected_component
# =====================================================================

def convert_dicomm_to_mhd(input_path, output_path, memory_saver=True):
    """ 
    ===========================================================================
    1)  convert_dicomm_to_mhd(input_path, output_path, memory_saver=True) -> None
    ============================================================================

    PARAMETERS
    ----------
    input_path (str): 
        path to the folder containing all the dicomm files (or a glob pattern to match the dicomm files, for example, "data/*.dcm")
    output_path (str): 
        path where the output mhd file will be saved (the function will add the .mhd extension automatically)
    memory_saver (bool, optional): 
        if True, the function will convert the pixel values to uint8 format to save memory. Default is True.

    RETURNS
    -------
    NONE (writes output files to disk)


    """

    # Collect and sort DICOM files (Path-based)
    p = Path(input_path)
    #if the input path is a directory, we will read all the files in the directory, 
    if p.is_dir():
        DICOM_directory = sorted(str(f) for f in p.iterdir() if f.is_file()) #sorted is ordering the files in the directory by their names
    #otherwise, we will read the files that match the pattern in the input path, for example, if the input path is "data/*.dcm", we will read all the files that end with .dcm in the data directory
    else:
        DICOM_directory = sorted(str(f) for f in p.parent.glob(p.name))

    if len(DICOM_directory) == 0:
        raise FileNotFoundError(f"No DICOM files found for pattern: {input_path}")
    
    # fetch metadata
    Image = pydicom.dcmread(DICOM_directory[0])
    Dimension = (int(Image.Rows), int(Image.Columns), len(DICOM_directory))
    logger.info(f"CT scan of dimension {Dimension} detected")

    try:
        Spacing = (float(Image.PixelSpacing[0]), float(Image.PixelSpacing[1]), float(Image.SliceThickness))
        logger.info(f"Pixel spacing: {Spacing}")
    except AttributeError:
        Spacing = (1.0, 1.0, 1.0)
        logger.warning("Pixel spacing or slice thickness not found in DICOM metadata. Defaulting to (1.0, 1.0, 1.0).")
    try:
        Origin = Image.ImagePositionPatient
        logger.info(f"Image origin: {Origin}")
    except AttributeError:
        Origin = (0.0, 0.0, 0.0)
        logger.warning("Image position not found in DICOM metadata. Defaulting to (0.0, 0.0, 0.0).")
    

    # Preallocate array
    dtype = np.uint8 if memory_saver else Image.pixel_array.dtype
    NpArrDc = np.zeros(Dimension, dtype=dtype)

    # loop through all images
    for i,filename in enumerate(DICOM_directory):
        _loading_bar(i,len(DICOM_directory),bar_length=30)
        df = pydicom.dcmread(filename)
        img = df.pixel_array
        if memory_saver:
            # Normalize to 0-255 before converting to uint8
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
        NpArrDc[:, :, i] = img

    print("now saving as mhd")
    NpArrDc = np.transpose(NpArrDc, (2, 0, 1))  # axis transpose
    sitk_img = sitk.GetImageFromArray(NpArrDc, isVector=False)
    sitk_img.SetSpacing(Spacing)
    sitk_img.SetOrigin(Origin)

    output_file = Path(output_path).with_suffix(".mhd")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(output_file))
    return


# =====================================================================
# 2) _loading_bar
# =====================================================================
def _loading_bar(current, total, bar_length=30):
    """
    Displays or updates a loading bar animation on the console based on the current progress.

    Parameters:
    - current: The current progress value.
    - total: The total value corresponding to 100% progress.
    - bar_length: The length of the loading bar in characters.
    """
    # Calculate the percentage of progress
    progress = current / total
    
    # Ensure the progress does not exceed 100%
    progress = min(1.0, max(0.0, progress))
    
    # Calculate the number of filled positions in the bar
    filled_length = int(bar_length * progress)
    
    # Create the bar string
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    
    # Print the loading bar with the current percentage
    sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%")
    #sys.stdout.flush()

    # Print a newline when progress reaches 100%
    if progress == 1.0:
        print()  # Move to a new line



# =====================================================================
# 3) convert_tiff_to_mhd
# =====================================================================

def convert_tiff_to_mhd(input_path, output_path, memory_saver=True):
    """ 
    ===========================================================================
    3) convert_tiff_to_mhd(input_path, output_path, memory_saver=True) -> None
    ============================================================================

    PARAMETERS
    ----------
    input_path (str): 
        path to the folder containing all the tiff files (or a glob pattern to match the tiff files, for example, "data/*.tif")
    output_path (str): 
        path where the output mhd file will be saved (the function will add the .mhd extension automatically)
    spacing (tuple, optional):
        voxel spacing in mm (x, y, z). Default is (0.2, 0.2, 0.2).
    memory_saver (bool, optional): 
        if True, the function will convert the pixel values to uint8 format to save memory. Default is True.

    RETURNS
    -------
    NONE (writes output files to disk)
    """
    
    # Collect and sort TIFF files (Path-based)
    p = Path(input_path)
    if p.is_dir():
        tiff_directory = sorted(f for f in p.iterdir() if f.is_file())
    else:
        tiff_directory = sorted(p.parent.glob(p.name))

    if len(tiff_directory) == 0:
        raise FileNotFoundError(f"No TIFF files found for pattern: {input_path}")

    logger.info(f"{len(tiff_directory)} TIFF images found.")

    # Read first image to get dimensions
    first_image = tiff.imread(str(tiff_directory[0]))
    Dimension = (int(first_image.shape[0]), int(first_image.shape[1]), len(tiff_directory))
    logger.info(f"CT scan of dimension {Dimension} detected")

    Origin = (0.0, 0.0, 0.0)
    Spacing = (0.2, 0.2, 0.2)
    logger.info(f"Using default spacing: {Spacing} and origin: {Origin}")

    # Preallocate array
    dtype = np.uint8 if memory_saver else first_image.dtype
    NpArrDc = np.zeros(Dimension, dtype=dtype)

    # Loop through all images
    for i, filepath in enumerate(tiff_directory):
        _loading_bar(i, len(tiff_directory), bar_length=30)
        img = tiff.imread(str(filepath))
        if memory_saver:
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
        NpArrDc[:, :, i] = img
    
    logger.info("Saving as mhd...")
    NpArrDc = np.transpose(NpArrDc, (2, 0, 1))  # axis transpose
    sitk_img = sitk.GetImageFromArray(NpArrDc, isVector=False)
    sitk_img.SetSpacing(Spacing)
    sitk_img.SetOrigin(Origin)

    output_file = Path(output_path).with_suffix(".mhd")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(output_file))

    return


# =====================================================================
# 4) read_mhd_file
# =====================================================================

def read_mhd_file(input_file_path):
    """
    ===========================================================================
    4) read_mhd_file(input_file_path) -> (image_array, spacing, origin)
    ============================================================================

    PARAMETERS
    ----------
    file_path: 
        str, path to the .mhd file to be read.

    RETURNS
    -------
    image_array: 
        np.ndarray, the image data as a NumPy array.
    spacing: 
        tuple, the voxel spacing in mm (x, y, z).
    origin: 
        tuple, the image origin in mm (x, y, z).
    """
    image = sitk.ReadImage(input_file_path)
    CT_visualization_window.open_window(image)

    return sitk.GetArrayFromImage(image), image.GetSpacing(), image.GetOrigin()


# =====================================================================
# 5) crop_images
# =====================================================================
def crop_images(point, direction, images):
    """
    ===========================================================================
    5) crop_images(point, direction, images)    -> cropped_image
    ============================================================================

    Crop an MHD image stack along a straight line (vertical or horizontal).

    PARAMETERS
    ----------
    point (int):
        The coordinate value at which the crop will occur.
    direction (str):
        Direction indicating which part to KEEP: 'up', 'down', 'left', 'right', 'front', or 'back'.
    images Numpy array or SimpleITK Image:
        The MHD image stack (SimpleITK Image object).

    RETURNS
    -------
    cropped_image (SimpleITK Image):
        The cropped image as a SimpleITK Image object.
    """
    if isinstance(images, sitk.Image):
        images = sitk.GetArrayFromImage(images)

    print('Cropping image')
    n = images.shape[0]
    y = images.shape[1]
    x = images.shape[2]
    print(f"Current size is {n}, {y}, {x}")

    if direction == 'down':
        start_index = [0, point, 0]
        size = [x, y - point, n]

    elif direction == 'up':
        start_index = [0, 0, 0]
        size = [x, point, n]

    elif direction == 'right':
        start_index = [point, 0, 0]
        size = [x - point, y, n]

    elif direction == 'left':
        start_index = [0, 0, 0]
        size = [point, y, n]

    elif direction =='back':
        start_index = [0, 0, point]
        size = [x , y, n - point]

    elif direction =='front':
        start_index = [0, 0, 0]
        size = [x , y, point]
    
    else:
        raise ValueError("Invalid direction. Must be one of 'up', 'down', 'right', 'left'.")

    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(size)
    extract_filter.SetIndex(start_index)
    
    # Apply the filter to crop the image
    cropped_image = extract_filter.Execute(sitk.GetImageFromArray(images))
    print("Image cropped")
    
    cropped_image = sitk.GetArrayFromImage(cropped_image)
    n = cropped_image.shape[0]
    y = cropped_image.shape[1]
    x = cropped_image.shape[2]
    print(f"New size is {n}, {y}, {x}")

    return cropped_image 


# =====================================================================
# 6) segment_from_threshold
# =====================================================================
def segment_from_threshold(image,lower_threshold,upper_threshold):
    """
    ===========================================================================
    6) segment_from_threshold(image, lower_threshold, upper_threshold)  -> binary_image
    ============================================================================

    Apply dual threshold to an image and return the binary result.

    PARAMETERS
    ----------
    image (SimpleITK Image or np.ndarray):
        The MHD image stack (SimpleITK Image object or NumPy array).
    lower_threshold (float):
        Pixels with grey value below this will be set to 0.
    upper_threshold (float):
        Pixels with grey value above this will be set to 0.

    RETURNS
    -------
    binary_image (np.ndarray):
        Binary version of the input image (255 or 0).
    """
    if isinstance(images, sitk.Image):
        images = sitk.GetArrayFromImage(images)

    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(lower_threshold)  # Adjust these values based on your image
    threshold_filter.SetUpperThreshold(upper_threshold)
    threshold_filter.SetInsideValue(255)
    threshold_filter.SetOutsideValue(0)
    out_image = threshold_filter.Execute(sitk.GetImageFromArray(images))

    return sitk.GetArrayFromImage(out_image) 



# =====================================================================
# 7) apply_threshold
# =====================================================================
def apply_threshold(image, lower_threshold, upper_threshold):
    """
    ===========================================================================
    7) apply_threshold(image, lower_threshold, upper_threshold) -> thresholded_image
    ============================================================================
    Apply dual threshold to an image and set pixels outside the range to zero.
    Note: This is NOT segmentation, only thresholding.

    PARAMETERS
    ----------
    image (SimpleITK Image or np.ndarray):
        The MHD image stack (SimpleITK Image object or NumPy array).
    lower_threshold (float):
        Pixels with grey value below this will be set to zero.
    upper_threshold (float):
        Pixels with grey value above this will be set to zero.

    RETURNS
    -------
    thresholded_image (np.ndarray):
        Thresholded version of the input image with values outside range set to zero.
    """
    if isinstance(images, sitk.Image):
        images = sitk.GetArrayFromImage(images)

    images[images > upper_threshold] = lower_threshold
    images[images < lower_threshold] = lower_threshold
    images = images - lower_threshold

    return images



# =====================================================================
# 8) dilate_filter
# =====================================================================
def dilate_filter(image, kernel):
    """
    ===========================================================================
    8) dilate_filter(image, kernel) -> dilated_image
    ============================================================================

    Apply grayscale dilation to an image.

    PARAMETERS
    ----------
    image (SimpleITK Image or np.ndarray):
        The image to dilate (SimpleITK Image object or NumPy array).
    kernel (int):
        The kernel radius for the dilation operation.

    RETURNS
    -------
    dilated_image (np.ndarray):
        Dilated version of the input image.
    """
    if isinstance(images, np.ndarray):
        images = sitk.GetImageFromArray(images)

    dilate_filter = sitk.GrayscaleDilateImageFilter()
    dilate_filter.SetKernelRadius(kernel)
    out_image = dilate_filter.Execute(image)

    return sitk.GetArrayFromImage(out_image)


# =====================================================================
# 9) erode_filter
# =====================================================================
def erode_filter(image, kernel):
    """
    ===========================================================================
    9) erode_filter(image, kernel)-> eroded_image
    ============================================================================

    Apply grayscale erosion to an image.

    PARAMETERS
    ----------
    image (SimpleITK Image or np.ndarray):
        The image to erode (SimpleITK Image object or NumPy array).
    kernel (int):
        The kernel radius for the erosion operation.

    RETURNS
    -------
    eroded_image (np.ndarray):
        Eroded version of the input image.
    """
    if isinstance(images, np.ndarray):
        images = sitk.GetImageFromArray(images)

    erode_filter = sitk.GrayscaleErodeImageFilter()
    erode_filter.SetKernelRadius(kernel)
    out_image = erode_filter.Execute(image)

    return sitk.GetArrayFromImage(out_image)



# =====================================================================
# 10) connected_filter
# =====================================================================
def connected_filter(x: int, y: int, z: int, images):
    """
    ===========================================================================
    10) connected_filter(x, y, z, images)   -> filtered_image
    ============================================================================
    Apply connected component filtering to an image starting from a seed point.

    PARAMETERS
    ----------
    x (int):
        X coordinate of the seed point.
    y (int):
        Y coordinate of the seed point.
    z (int):
        Z coordinate of the seed point.
    images (SimpleITK Image or np.ndarray):
        The image to filter (SimpleITK Image object or NumPy array).

    RETURNS
    -------
    filtered_image (np.ndarray):
        Filtered image containing only the connected component from the seed point.
    """
    # Use the connected component filter with the seed 
    if isinstance(images, np.ndarray):
        images = sitk.GetImageFromArray(images)

    connected_filter = sitk.ConnectedThresholdImageFilter()
    connected_filter.SetLower(1)
    connected_filter.SetUpper(255)
    connected_filter.SetReplaceValue(255)
    connected_filter.SetSeedList([(x, y, z)])
    out_image = connected_filter.Execute(images)

    return sitk.GetArrayFromImage(out_image)



# =====================================================================
# 11) find_small_holes
# =====================================================================
def find_small_holes(binary_image, max_hole_size):
    """
    ===========================================================================
    11) find_small_holes(binary_image, max_hole_size)   -> small_holes_image
    ============================================================================

    Find small holes from a binary image with foreground=255 (uint8 format).
    Holes are identified by inverting the image and running connected component
    analysis, then filtering by rank (size order).

    PARAMETERS
    ----------
    binary_image (SimpleITK Image or np.ndarray):
        A binary image (foreground=255, background=0).
    max_hole_size (int):
        The rank of the largest hole to include. 0 is the foreground itself,
        1 is the biggest hole, 2 the second biggest, etc. All holes at or
        above this rank are returned.

    RETURNS
    -------
    small_holes_image (np.ndarray):
        Binary image containing only the small holes (foreground=255, background=0).
    """
    # Convert numpy array to sitk image if needed
    if isinstance(binary_image, np.ndarray):
        binary_image = sitk.GetImageFromArray(binary_image)
    
    # Rescale image to binary (foreground=1, background=0)
    binary_image_rescaled = sitk.Cast(binary_image > 0, sitk.sitkUInt8)
    
    # Invert the binary image
    inverted_image = sitk.InvertIntensity(binary_image_rescaled, maximum=1)
    
    # Connected component analysis on the inverted image
    connected_components = sitk.ConnectedComponent(inverted_image)  #create a list of all islands
    print(f"{sitk.GetArrayFromImage(connected_components).max()} holes found")
          
    if sitk.GetArrayFromImage(connected_components).max() < max_hole_size:
        print(f"error: hole size too big. maximum should be {sitk.GetArrayFromImage(connected_components).max()} and you entered {max_hole_size}...")
        return
    
    # Relabel components by size and filter based on size
    relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)   #re-order the holes by size: the small the holes, the higher its rank

    #extract only the small holes
    small_holes = sitk.BinaryThreshold(                                                  #find all the holes above the thresshold. They have a value of 0 and outside a value of 1  
        relabeled_components,
        lowerThreshold=max_hole_size,           
        upperThreshold=int(sitk.GetArrayFromImage(connected_components).max()),   # get all the other holes
        insideValue=1,
        outsideValue=0
    )

    # Rescale back to uint8 format (foreground=255)
    final_image = sitk.Cast(small_holes, sitk.sitkUInt8) * 255
    
    return sitk.GetArrayFromImage(final_image)


# =====================================================================
# 12) find_islands
# =====================================================================
def find_islands(binary_image, max_island_size):
    """
    ===========================================================================
    12) find_islands(binary_image, max_island_size)     -> small_islands_image
    ============================================================================

    Find small isolated islands (foreground blobs) in a binary image with
    foreground=255 (uint8 format). This is the inverse of find_small_holes:
    it operates on the foreground directly instead of the background.

    PARAMETERS
    ----------
    binary_image (SimpleITK Image or np.ndarray):
        A binary image (foreground=255, background=0).
    max_island_size (int):
        The rank of the largest island to include. 0 is the main foreground,
        1 is the biggest island, 2 the second biggest, etc. All islands at or
        above this rank are returned.

    RETURNS
    -------
    small_islands_image (np.ndarray):
        Binary image containing only the small islands (foreground=255, background=0).
    """
    # Convert numpy array to sitk image if needed
    if isinstance(binary_image, np.ndarray):
        binary_image = sitk.GetImageFromArray(binary_image)

    # Rescale image to binary (foreground=1, background=0)
    binary_image_rescaled = sitk.Cast(binary_image > 0, sitk.sitkUInt8)
    
    # Invert the binary image so that foreground becomes holes for find_small_holes
    inverted_image = sitk.InvertIntensity(binary_image_rescaled, maximum=1)

    # Find islands by treating inverted foreground as holes
    islands = find_small_holes(inverted_image, max_island_size)
    
    return islands


# =====================================================================
# 13) watershed_algorithm
# =====================================================================
def watershed_algorithm(single_image, sure_fg, sure_bg):
    """
    ===========================================================================
    13) watershed_algorithm(single_image, sure_fg, sure_bg) -> (single_image, connection_markers_display)
    ============================================================================

    Use a watershed algorithm to color edges of zones (touching or not) black.

    PARAMETERS
    ----------
    single_image (np.ndarray or SimpleITK Image):
        Used for the "topography" of your zones.
    sure_fg (np.ndarray or SimpleITK Image):
        The sure foreground mask; foreground pixels appear as 255.
    sure_bg (np.ndarray or SimpleITK Image):
        The sure background mask; pixels appear as 0. The unknown area is
        obtained by subtracting sure_fg from this mask.

    RETURNS
    -------
    single_image (np.ndarray):
        The input image with watershed boundary pixels set to 255 (white).
    connection_markers_display (np.ndarray):
        The labeled marker image normalized to uint16 for display purposes.
    """
    #make sure all input images are np.arrays
    if isinstance(single_image, sitk.Image):
        single_image = sitk.GetArrayFromImage(single_image)
    if isinstance(sure_fg, sitk.Image):
        sure_fg = sitk.GetArrayFromImage(sure_fg)
    if isinstance(sure_bg, sitk.Image):
        sure_bg = sitk.GetArrayFromImage(sure_bg)

    #small function to write input image to uint8
    def _norm_uint8(arr):
        """Min-max normalize a numpy array to uint8 [0, 255]."""
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr.astype(np.uint8)

    # ----- Normalize inputs to uint8 using numpy -----
    original_image = np.copy(single_image)  # Keep a copy of the original image for output
    original_image_dtype = original_image.dtype
    orginal_image_max_theoretical_value = 
    single_image = _norm_uint8(single_image)
    sure_fg = _norm_uint8(sure_fg)
    sure_bg = _norm_uint8(sure_bg)

    # -----Saturating subtraction: unknown regions appear as 255 -----
    # you write to int16 since you want to avoid underflow when you subtract the sure foreground from the sure background. 
    # You want to keep the negative values as 0, which is what the np.clip does. Then you convert back to uint8 for the watershed algorithm.
    # remember that the sure foreground is 255 and the sure background is 0, so when you subtract the sure foreground from the sure background, 
    # you get -255 for the sure foreground and 0 for the sure background. The unknown regions will be 255, which is what we want for the watershed algorithm.
    unknown = np.clip(sure_bg.astype(np.int16) - sure_fg.astype(np.int16), 0, 255).astype(np.uint8) 

    # ----- Marker labeling using SimpleITK connected components -----
    sitk_sure_fg = sitk.GetImageFromArray((sure_fg > 0).astype(np.uint8))
    labeled = sitk.ConnectedComponent(sitk_sure_fg)             #this will label each connected component in the sure foreground with a unique integer value (starting from 1). The background will be labeled as 0. The output is a sitk image where each pixel has the label of the connected component it belongs to.
    connection_markers = sitk.GetArrayFromImage(labeled).astype(np.int32)

    # Add one to ensure background is 1, not 0
    connection_markers = connection_markers + 1
    # Mark the unknown regions as 0
    connection_markers[unknown == 255] = 0

    # ----- Normalise markers to uint16 for display (range [65535/5, 65535]) -----
    connection_markers_display = connection_markers.astype(np.float32)
    mn, mx = connection_markers_display.min(), connection_markers_display.max()
    if mx > mn:
        lo = 65535.0 / 5.0
        connection_markers_display = (
            (connection_markers_display - mn) / (mx - mn) * (65535.0 - lo) + lo
        )
    connection_markers_display = connection_markers_display.astype(np.uint16)
    connection_markers_display[connection_markers_display == connection_markers_display.min()] = 0

    # ----- Apply watershed using SimpleITK -----
    # markWatershedLine=True marks watershed boundary pixels as 0 in the output
    sitk_image = sitk.Cast(sitk.GetImageFromArray(single_image), sitk.sitkFloat32)                  #the watershed algorithm in SimpleITK requires the input image to be in a floating point format,
    connection_markers = sitk.Cast(sitk.GetImageFromArray(connection_markers), sitk.sitkUInt32)     #the watershed algorithm in SimpleITK requires the markers to be in an unsigned integer format (uint32) 
    watershed_result = sitk.MorphologicalWatershedFromMarkers(sitk_image, connection_markers, markWatershedLine=True)
    watershed_result = sitk.GetArrayFromImage(watershed_result).astype(np.int32)    #watershed_result is an image where each pixel has the label of the watershed region it belongs to. The watershed lines (boundaries) are marked as 0.

    # Mark boundaries in the original imagee as the max of whatever it is able to show (white)
    original_image[watershed_result == 0] = original_image.max()

    return original_image, connection_markers_display