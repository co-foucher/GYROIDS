import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import gridspec
import time
import numpy.ma as ma
import logging


"""
This script is intended to be used as you would a library. Import it as such:

    import sys
    # Add the directory to the sys.path
    sys.path.append('D:/COFO - HP/03 - Github/dvpt-at-EMPA/HappyBisons/image analysis')
    # Now you can import modules from that directory
    import tools.CT_visualization_window as CT_visualization_window
    %matplotlib qt

!!!! do not forget %matplotlib qt !!!!!

You should be able to easily add more "action buttons" by adding more functions in this script.


it is structured as follow:

- setups:
    - window: defined the figure, sliders..
    - buttons: defines the action buttons, and which function they activate
- (small) actions: list of small functions; 
    - rotate_view, next_slice, prev_slice, update_plot, reset_window, change_button_color, onscroll, greyvalue_at_coord, calculate_average, make_greyscale_inRGB
- (big) actions:
    - rainbow colors
    - histogram
    - on click

- logger: defines the logger for storing errors and displaying messages
- main: main functions that should be the only ones called a tools from the library


"""


# =======================================================================================================
# ================================================== setups =============================================
# =======================================================================================================

def setup_window():
    """ 
    opens the window an creates main global variables
    """
    global selected_pixels, images_rgb, fig, ax_image, slice, n, img_display, fig_histo, ax_histo, images
    logger.info("starting window setup function")

    selected_pixels = []

    n = images.shape[0]  # Number of slices
    y = images.shape[1]  # y axis size
    x = images.shape[2]  # x axis size

    if "images_rgb" not in globals():
        make_greyscale_inRGB(images)

    # Create a figure
    logger.debug("opening figure")
    fig, ax_image = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.90)

    # Initial display
    slice = 0
    img_display = ax_image.imshow(images_rgb[slice], cmap='gray')
    ax_image.axis('off')

    # Sliders
    setup_slider()

    # Buttons
    setup_buttons()

    # setup the histogram plot
    fig_histo, ax_histo = plt.subplots(figsize=(5, 5))
    plt.close(fig_histo)

    # Switch to the main figure for buttons and sliders
    #plt.figure(fig.number)
    plt.figure(fig)

    # Register the update function as a callback for slider value changes
    slice_slider.on_changed(update_plot)
    fig.canvas.mpl_connect('scroll_event', onscroll)

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)

    logger.info("window setup function succesfull")

# Buttons
def setup_buttons():
    logger.debug("adding buttons")
    start_time_setup_buttons = time.time() 
    global button_prev, button_next, button_3D_brush, button_histogram, button_reset, button_average, button_brush_ONOFF, button_rotate_x, button_rotate_y, button_rotate_z, button_rainbow
    ax_button_prev = plt.axes([0.02, 0.05, 0.05, 0.04])  # left, bottom, width, height
    button_prev = Button(ax_button_prev, 'Previous')
    button_prev.on_clicked(prev_slice)

    ax_button_next = plt.axes([0.9, 0.05, 0.05, 0.04])   # left, bottom, width, height
    button_next = Button(ax_button_next, 'Next')
    button_next.on_clicked(next_slice)

    ax_button_brush_ONOFF =  plt.axes([0.02, 0.86, 0.05, 0.05])     # left, bottom, width, height
    button_brush_ONOFF = Button(ax_button_brush_ONOFF, 'ON/OFF')
    button_brush_ONOFF.color = "red"
    button_brush_ONOFF.on_clicked(lambda event: change_button_color(event, button_brush_ONOFF))

    ax_button_3D_brush = plt.axes([0.02, 0.93, 0.05, 0.05])     # left, bottom, width, height
    button_3D_brush = Button(ax_button_3D_brush, '3D brush')
    button_3D_brush.color = "red"
    button_3D_brush.on_clicked(lambda event: change_button_color(event, button_3D_brush))

    ax_button_histogram = plt.axes([0.08, 0.93, 0.06, 0.05])     # left, bottom, width, height
    button_histogram = Button(ax_button_histogram, 'histogram')
    button_histogram.color = "red"
    button_histogram.on_clicked(lambda event: change_button_color(event, button_histogram))
    button_histogram.on_clicked(lambda event: update_histogram(event))

    ax_button_average = plt.axes([0.15, 0.93, 0.05, 0.05])     # left, bottom, width, height
    button_average = Button(ax_button_average, 'average')
    button_average.color = "grey"
    button_average.on_clicked(lambda event: calculate_average(event))

    ax_button_reset = plt.axes([0.21, 0.93, 0.05, 0.05])     # left, bottom, width, height
    button_reset = Button(ax_button_reset, 'reset')
    button_reset.color = "grey"
    button_reset.on_clicked(lambda event: reset_window(event))

    ax_button_rotate_x = plt.axes([0.27, 0.93, 0.03, 0.05])     # left, bottom, width, height
    button_rotate_x = Button(ax_button_rotate_x, 'Rot X')
    button_rotate_x.color = "green"
    button_rotate_x.on_clicked(lambda event: rotate_view(event,'x'))

    ax_button_rotate_y = plt.axes([0.30, 0.93, 0.03, 0.05])     # left, bottom, width, height
    button_rotate_y = Button(ax_button_rotate_y, 'Rot Y')
    button_rotate_y.color = "green"
    button_rotate_y.on_clicked(lambda event: rotate_view(event,'y'))

    ax_button_rotate_z = plt.axes([0.33, 0.93, 0.03, 0.05])     # left, bottom, width, height
    button_rotate_z = Button(ax_button_rotate_z, 'Rot Z')
    button_rotate_z.color = "green"
    button_rotate_z.on_clicked(lambda event: rotate_view(event,'z'))

    ax_button_rainbow = plt.axes([0.37, 0.93, 0.05, 0.05])     # left, bottom, width, height
    button_rainbow = Button(ax_button_rainbow, 'rainbow')
    button_rainbow.color = "grey"
    button_rainbow.on_clicked(lambda event: rainbow_colors(event))
    logger.debug(f"buttons added in {time.time() - start_time_setup_buttons} seconds.")

# Sliders
def setup_slider():
    logger.debug("adding sliders")
    start_time_setup_slider = time.time() 
    global slice, slice_slider, brush_radius_slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgrey')   # left, bottom, width, height
    slice_slider = Slider(ax_slider, 'Slice', 0, n-1, valinit=0, valstep=1)
    slice = 0

    ax_brush_radius = plt.axes([0.03, 0.2, 0.03, 0.6], facecolor='lightgrey')  # left, bottom, width, height
    brush_radius_slider = Slider(ax_brush_radius, 'brush thickness', 0, 10, valinit=1, valstep=1, orientation='vertical')
    logger.debug(f"sliders added in {time.time() - start_time_setup_slider} seconds.")


# =======================================================================================================
# ========================================= (small) actions =============================================
# =======================================================================================================
def rotate_view(event, axis):
    global images_rgb, images, fig
    if axis == "x":
        images = np.transpose(images, (1, 2, 0))
        images_rgb = np.transpose(images_rgb, (1, 2, 0, 3))
    elif axis == "y":
        images = np.transpose(images, (2, 1, 0))
        images_rgb = np.transpose(images_rgb, (2, 1, 0, 3))
    elif axis == "z":
        images = np.transpose(images, (0, 2, 1))
        images_rgb = np.transpose(images_rgb, (0, 2, 1, 3))

    logger.info(f" images have been rotated along axis {axis}, new dimensions: {images.shape}")
    plt.close(fig)
    setup_window()

def next_slice(event):
    global slice_slider, n
    current_slice = slice_slider.val
    if current_slice < n - 1:
        slice_slider.set_val(current_slice + 1)

def prev_slice(event):
    global slice_slider, n
    current_slice = slice_slider.val
    if current_slice > 0:
        slice_slider.set_val(current_slice - 1)

def update_plot(event):
    global slice, button_histogram, slice_slider, images_rgb, img_display
    start_time_update_plot = time.time() 
    slice = slice_slider.val  # Get the current value of the slider

    if button_histogram.color == "blue":
        update_histogram(None)
    
    img_display.set_data(images_rgb[slice])
    fig.canvas.draw_idle()

    logger.debug(f"plot update successful in {time.time() - start_time_update_plot} seconds.")

def reset_window(event):
    global images_rgb, selected_pixels, images
    logger.info("resetting window, please wait...")
    start_time_reset_window = time.time() 

    temp_images = images / np.max(images) #for rbg, values go from 0 to 1
    images_rgb = np.stack([temp_images, temp_images, temp_images], axis=-1)
    selected_pixels = []
    fig.canvas.draw_idle()

    logger.debug(f"reset successful in {time.time() - start_time_reset_window} seconds.")

def change_button_color(event, button):
        button.color = 'blue' if button.color == 'red' else 'red'

def onscroll(event):
    global slice_slider, slice
    current_slice = slice_slider.val
    if event.button == 'up' and current_slice < n - 1:
        slice_slider.set_val(current_slice + 1)
        slice = slice_slider.val
    elif event.button == 'down' and current_slice > 0:
        slice_slider.set_val(current_slice - 1)
        slice = slice_slider.val
    update_plot(slice_slider.val)

def greyvalue_at_coord(x, y):
    global images, slice
    numrows, numcols = images.shape[1], images.shape[2]
    col,row = int(x + 0.5), int(y + 0.5)
    if 0 <= col < numcols and 0 <= row < numrows:
        grey = images[slice, row, col]
        return grey
    else:
        return None
    
def calculate_average(event):
    global images, selected_pixels
    pixels_coordinates = np.array(selected_pixels)
    if pixels_coordinates.size == 0:
        logger.warning("No pixels selected.")
        return
    n_indices, y_indices, x_indices = pixels_coordinates[:,2], pixels_coordinates[:,1], pixels_coordinates[:,0]
    temp_average = [images[n_indices, y_indices, x_indices]]
    dat_average = np.mean(temp_average)
    logger.info(f"average value of selected pixel = {dat_average}")


def make_greyscale_inRGB(images):
    global images_rgb
    logger.debug("converting grey scale image to rgb")
    start_time = time.time() 

    if images.dtype != np.uint8:
        logger.warning(f"Your image uses {images.dtype} values, values have been rescaled in the plot to fit in UINT8, but still extracted true on click.")

    divider = 255/(images.max()-images.min())
    temp_images = np.multiply(np.subtract(images,images.min()), divider).astype(np.uint8)
    logger.debug(f"Plotted image has been reduced to UINT8.")
    
    # Convert grayscale image to RGB for color modification
    images_rgb = np.stack([temp_images, temp_images, temp_images], axis=-1).astype(np.uint8)  #in rgb, equal values will give a shade of grey
    #                                                                                               rgb is added as the fourth axis !
    del temp_images
    logger.debug(f"rgb image takes {images_rgb.nbytes/1000000000} gigabytes")
    logger.debug(f"conversion succesfull in {time.time() - start_time} seconds.")

# =======================================================================================================
# ========================================= rainbow colors =============================================
# =======================================================================================================

def rainbow_colors(event):
    global images_rgb, images, slice
    global fig_rainbow,rgb_scale
    start_time_rainbow_colors = time.time() 
    # Ensure the condition checks the values properly
    if (images_rgb[:,:,:,0] == images_rgb[:,:,:,1]).all():
        logger.info("opening rainbow color set window")
        rgb_scale = np.copy(images_rgb[:,:,:,0]).astype(np.uint8)

        """
        r = images_rgb[:,:,:,0]  # Red channel
        g = images_rgb[:,:,:,1]  # Green channel
        b = images_rgb[:,:,:,2]  # Blue channel
        """

        if not plt.fignum_exists(fig_histo.number):     # creating the rainbow scale interaction window
            global rainbow_slider_min, rainbow_slider_max, button_apply_rainbow_all
            print('opening sliders window')
            fig_rainbow, ax_rainbow = plt.subplots(figsize=(10, 1))
            ax_rainbow.axis('off')
            ax_rainbow_min = plt.axes([0.3, 0.6, 0.6, 0.1], facecolor='lightgrey')  # left, bottom, width, height
            rainbow_slider_min = Slider(ax_rainbow_min, 'min rainbow value', 0, 255, valinit=0, valstep=1)

            ax_rainbow_max = plt.axes([0.3, 0.3, 0.6, 0.1], facecolor='lightgrey')  # left, bottom, width, height
            rainbow_slider_max = Slider(ax_rainbow_max, 'max rainbow value', 0, 255, valinit=255, valstep=1)

            ax_button_apply_rainbow_all = plt.axes([0.025, 0.35, 0.05, 0.3])     # left, bottom, width, height
            button_apply_rainbow_all = Button(ax_button_apply_rainbow_all, 'Apply all')
            button_apply_rainbow_all.color = "green"
            button_apply_rainbow_all.on_clicked(lambda event: apply_rainbow_all(event))
            
            plt.show()
            rainbow_slider_min.on_changed(lambda event: update_rainbow(event,'min_slider'))
            rainbow_slider_max.on_changed(lambda event: update_rainbow(event,'max_slider'))
        
    else : 
        del rgb_scale, rainbow_slider_min, rainbow_slider_max, button_apply_rainbow_all
        logger.info("changing colorset to grey-scale")
        plt.close(fig_rainbow)
        make_greyscale_inRGB(images)
        del fig_rainbow
    
    #logger.debug(f"new color scale applied in {time.time() - start_time_rainbow_colors} seconds.")

def calculate_rainbow_masks(min_rainbow,max_rainbow,rgb_scale):
    increment_rainbow = (max_rainbow-min_rainbow)/4
    mask_0_25 = np.array((rgb_scale >= min_rainbow) & (rgb_scale < min_rainbow + increment_rainbow), dtype=np.bool)
    mask_25_50 = np.array((rgb_scale >= min_rainbow + increment_rainbow) & (rgb_scale < min_rainbow + increment_rainbow*2), dtype=np.bool)
    mask_50_75 = np.array((rgb_scale >= min_rainbow + increment_rainbow*2) & (rgb_scale < min_rainbow + increment_rainbow*3), dtype=np.bool)
    mask_75_100 = np.array((rgb_scale >= min_rainbow + increment_rainbow*3) & (rgb_scale <= min_rainbow + max_rainbow), dtype=np.bool)
    return mask_0_25, mask_25_50, mask_50_75, mask_75_100

def update_rainbow(event,event_name):
    global rgb_scale, images_rgb,r,g,b, slice
    min_val = rainbow_slider_min.val
    max_val = rainbow_slider_max.val

    r_temp = np.zeros_like(images_rgb[slice,:,:,0])
    g_temp = np.copy(r_temp)
    b_temp = np.copy(r_temp)
    scale_temp = rgb_scale[slice,:,:]

    # Reapply the rainbow mapping but restrict the values according to the sliders
    mask_0_25, mask_25_50, mask_50_75, mask_75_100 = calculate_rainbow_masks(min_val,max_val,scale_temp)
    quarter_rainbow = (max_val-min_val)/4

    multiplier = 255 / quarter_rainbow
    new_rgbscale = scale_temp - min_val
    r_temp[mask_0_25] = (new_rgbscale[mask_0_25]) * multiplier
    g_temp[mask_0_25] = 255

    multiplier = (quarter_rainbow + min_val) * 255
    r_temp[mask_25_50] = 255
    g_temp[mask_25_50] = 255 - (new_rgbscale[mask_25_50] - quarter_rainbow) * multiplier

    r_temp[mask_50_75] = 255
    b_temp[mask_50_75] = (new_rgbscale[mask_50_75] - 2*quarter_rainbow) * multiplier

    r_temp[mask_75_100] = 255 - (new_rgbscale[mask_75_100] - 3*quarter_rainbow) * multiplier
    b_temp[mask_75_100] = 255
    del new_rgbscale

    outside_mask = (~mask_0_25) & (~mask_25_50) & (~mask_50_75) & (~mask_75_100)
    r_temp[outside_mask] = scale_temp[outside_mask]
    g_temp[outside_mask] = scale_temp[outside_mask]
    b_temp[outside_mask] = scale_temp[outside_mask]

    # Combine channels into images_rgb
    images_rgb[slice,:,:,0] = r_temp  # Red channel
    images_rgb[slice,:,:,1] = g_temp  # Green channel
    images_rgb[slice,:,:,2] = b_temp  # Blue channel

    update_plot(slice)

def apply_rainbow_all(event):
    global rgb_scale, images_rgb, slice
    logger.info("Changing color set to rainbow for every images")
    start_time_apply_rainbow_all = time.time()

    min_val = rainbow_slider_min.val
    max_val = rainbow_slider_max.val

    r = np.zeros_like(images_rgb[:,:,:,0])
    g = np.copy(r)
    b = np.copy(r)

    # Reapply the rainbow mapping but restrict the values according to the sliders
    mask_0_25, mask_25_50, mask_50_75, mask_75_100 = calculate_rainbow_masks(min_val,max_val,rgb_scale)
    quarter_rainbow = (max_val-min_val)/4

    multiplier = 255 / quarter_rainbow
    new_rgbscale = rgb_scale - min_val
    r[mask_0_25] = (new_rgbscale[mask_0_25]) * multiplier
    g[mask_0_25] = 255

    multiplier = (quarter_rainbow + min_val) * 255
    r[mask_25_50] = 255
    g[mask_25_50] = 255 - (new_rgbscale[mask_25_50] - quarter_rainbow) * multiplier

    r[mask_50_75] = 255
    b[mask_50_75] = (new_rgbscale[mask_50_75] - 2*quarter_rainbow) * multiplier

    r[mask_75_100] = 255 - (new_rgbscale[mask_75_100] - 3*quarter_rainbow) * multiplier
    b[mask_75_100] = 255
    del new_rgbscale

    outside_mask = (~mask_0_25) & (~mask_25_50) & (~mask_50_75) & (~mask_75_100)
    r[outside_mask] = rgb_scale[outside_mask]
    g[outside_mask] = rgb_scale[outside_mask]
    b[outside_mask] = rgb_scale[outside_mask]

    # Combine channels into images_rgb
    images_rgb[:,:,:,0] = r  # Red channel
    images_rgb[:,:,:,1] = g  # Green channel
    images_rgb[:,:,:,2] = b  # Blue channel

    update_plot(slice)
    logger.debug(f"new color scale applied to every images in {time.time() - start_time_apply_rainbow_all} seconds.")

# =======================================================================================================
# ========================================= histogram =============================================
# =======================================================================================================

def update_histogram(event):
    '''
    Inputs: - mhd stack (SimpleITK image)

    Outputs: - matplotlib plot
    '''
    global fig_histo, ax_histo, slice, images

    start_time_update_histogram = time.time() 

    if button_histogram.color == "blue" :
        array = images[slice,:,:]

        # Get unique values and their counts
        unique_values, counts = np.unique(array, return_counts=True)

        if not plt.fignum_exists(fig_histo.number):
            logger.info('opening new histogram window')
            fig_histo, ax_histo = plt.subplots(figsize=(5, 5))
            ax_histo.plot(unique_values, counts, color='blue', alpha=0.7)
            ax_histo.set_xlabel('Pixel Value')
            ax_histo.set_ylabel('Frequency')
            ax_histo.set_title('Histogram of Pixel Values (Excluding Zeros)')
            ax_histo.grid(True)
            plt.show()

        else:
            # Clear the existing plot and redraw
            ax_histo.clear()
            ax_histo.plot(unique_values, counts, color='blue', alpha=0.7)
            ax_histo.set_xlabel('Pixel Value')
            ax_histo.set_ylabel('Frequency')
            #ax_histo.set_title(f'Histogram of Pixel Values in Slice {slice_slider.val} (Excluding Zeros)')
            ax_histo.grid(True)
            fig_histo.canvas.draw_idle()

    elif button_histogram.color == "red":
        logger.info('closing histogram window')
        plt.close(fig_histo)

    logger.debug(f"histogram updated in {time.time() - start_time_update_histogram} seconds.")


# =======================================================================================================
# ========================================= on click =============================================
# =======================================================================================================

def onclick(event):
    """
    function controlling what happens when the user clicks on a pixel
    it's in here that:
        - we check if the user click in an allowed area
        - we check wether the brush is in 3D mode or not and elect the pixels accordingly
        - after each click the color of selected pixels are turned to blue and the plot updated

    notes:
        - selected pixels are appended in the variable selected_pixels


    """
    global slice, images, ax_image, selected_pixels, images_rgb, button_3D_brush
    if event.inaxes == ax_image and event.button == 1:          # Check if the click is within the image axes and not on the slider
        toolbar = plt.get_current_fig_manager().toolbar         # save the current zoom and position on the image
        if toolbar.mode == '':                                  # do not register if using any toolbar tool 
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)       # save click position
            slice = int(slice_slider.val)
            grey = greyvalue_at_coord(x, y)              # save  grey value at click position
            if grey is not None:                                        # insures that you clicked in a not non-defined pixel
                print(f"Clicked at coordinates: x={x}, y={y}, in slice {slice}, Pixel value: {grey}")
                #now looping through all the pixels according to the brush's thickness
                if button_brush_ONOFF.color == "blue":
                    for t in range(-int(brush_radius_slider.val), int(brush_radius_slider.val) + 1):        #thickness of the brush (x-axis)
                        for g in range(-int(brush_radius_slider.val), int(brush_radius_slider.val) + 1):    #height of the brush (y-axis)
                            
                            #============= 2D brush =============
                            if button_3D_brush.color == "red":      
                                if np.sqrt(t ** 2 + g ** 2) <= int(brush_radius_slider.val):            # make the brush round
                                    grey = greyvalue_at_coord(x + t, y + g)
                                    if grey is not None:
                                        selected_pixels.append((int(x + t), int(y + g), int(slice)))    # append list of slected pixels
                                        images_rgb[slice, int(y + g), int(x + t)] = [0, 0, 255]         # Set the clicked pixel to blue (0, 0, 255)
                            
                            #============ 3D brush =============
                            elif button_3D_brush.color == "blue":  #on est en 3D
                                for k in range(-int(brush_radius_slider.val), int(brush_radius_slider.val) + 1):    # deepness of the brush (z-axis)
                                    if np.sqrt(t ** 2 + g ** 2 + k**2) <= int(brush_radius_slider.val):             # make the brush round
                                        grey = greyvalue_at_coord(x + t, y + g)
                                        if grey is not None:                                                    # insures that you clicked in a not non-defined pixel
                                            selected_pixels.append((int(x + t), int(y + g), int(slice + k)))        # append list of slected pixels
                                            images_rgb[slice + k, int(y + g), int(x + t)] = [0, 0, 255]             # Set the clicked pixel to blue (0, 0, 255)

            update_plot(slice)

# =======================================================================================================
# ========================================= logger =============================================
# =======================================================================================================

def configure_logger(level):
    """
    Configures a logger with the given logging level.
    
    Args:
        level (int): Logging level. Defaults to logging.INFO.
    """
    # Create or get the logger
    logger = logging.getLogger("CT_window_logger")
    logger.setLevel(level)  # Set the logging level

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Ensure the handler matches the logger's level
    
    # Create and set a simple formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger


# =======================================================================================================
# ============================================= main =============================================
# =======================================================================================================

def open_window(input_images,level=logging.INFO):
    global images, logger

    logger = configure_logger(level)  # Default to INFO level

    if "images_rgb" in globals():
        del globals()['images_rgb']

    images = sitk.GetArrayFromImage(input_images)
    logger.debug(f"your image takes {images.nbytes/1000000000} gigabytes")

    setup_window()

    return np.unique(np.array(selected_pixels),axis=0)      #delete all redondant pixel due to multi-click from the user


def lightweigth_open(images):
    ''' 
    Lets you browse through an mhd stack using an interactive slider within the figure. 
    When clicking somewhere in the picture, it will print the value and position of the pixel you selected.
    The image is opened in another window in order to be interactive.
    
    Inputs: -numpy array !!!

    Outputs: -none

    Other: -creates an interactive plot
           - !!!!! you need %matplotlib qt !!!! in your code
    '''
    global ax_image, slice, slice_slider


    #images = cv2.normalize(sitk.GetArrayFromImage(images), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    images = sitk.GetArrayFromImage(images)
    print(f"your image takes {images.nbytes/1000000000} gigabytes")
    divider = 255/images.max()
    images = np.multiply(images, divider)
    images = images.astype(np.uint8)
    print(f"your image has been reduced to {images.nbytes/1000000000} gigabytes")

    #images = sitk.GetArrayFromImage(sitk.Cast(sitk.RescaleIntensity(images, outputMinimum=0, outputMaximum=255), sitk.sitkUInt8))
    #

    n = images.shape[0]  # Number of slices
    v_min = np.min(images)
    v_max = np.max(images)

    def greyvalue_at_coord_light(images, slice, x, y):
        numrows, numcols = images.shape[1], images.shape[2]
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            grey = images[slice, row, col]
            return grey
        else:
            return None

    def onclick_light(event):
        global ax_image, images, slice_slider
        if event.inaxes == ax_image:
            x, y = event.xdata, event.ydata
            slice = int(slice_slider.val)
            grey = greyvalue_at_coord_light(images, slice, x, y)
            if grey is not None:
                print(f"Clicked at coordinates: x={x}, y={y}, in slice {slice}, Pixel value: {grey}")
    
    # Create a figure with gridspec
    fig, ax_image = plt.subplots(figsize=(14, 7))

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    # Initial display
    img_display = ax_image.imshow(images[0], cmap='gray', vmin=v_min, vmax=v_max)
    ax_image.axis('off')


    # Slider axis
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgrey')  # left, bottom, width, height
    slice_slider = Slider(ax_slider, 'Slice', 0, n-1, valinit=0, valstep=1)

    # Update function
    def update_plot_light(val):
        global slice, slice_slider
        slice = int(slice_slider.val)  # Get the current value of the slider
        temp = images[slice]
        img_display.set_data(temp)
        img_display.set_clim(vmin=v_min, vmax=v_max)  # Update the color limits
        fig.canvas.draw_idle()

    # Register the update function as a callback for slider value changes
    slice_slider.on_changed(update_plot_light)

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('button_press_event', onclick_light)

    plt.show()
    

def check_existence():
    print("you loaded the CT visualizer window v20.11.24")