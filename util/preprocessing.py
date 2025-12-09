import SimpleITK as sitk
import numpy as np
import pydicom
import os
import tensorflow as tf
from lxml import etree
from collections import Counter
from util.helpers import calculate_series_z_offset

MIN_HU = -1000
MAX_HU = 400
TARGET_LABEL_SHAPE = (3,)
TARGET_SHAPE = (64,64,64)

def normalize_data(ds, img3d):

  # print('CT volume shape (z,y,x):', img3d.shape)

  rescaleIntercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
  rescaleSlope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)

  hu_val = (img3d * rescaleSlope) + rescaleIntercept

  # clip extreme HU values to common window range
  data_clipped = np.clip(hu_val, MIN_HU, MAX_HU)
  
  # normalized to [0, 1]
  norm_data  = (data_clipped - MIN_HU) / (MAX_HU - MIN_HU)
  # convert to float32 for model input
  norm_data = norm_data.astype(np.float32)

  print('Normalization complete.')

  return norm_data

def resample_data(norm_data, original_spacing, target_spacing):
  # calculate scaling factor
  sitk_img = sitk.GetImageFromArray(norm_data)
  sitk_img.SetSpacing(original_spacing)

  scale_factor = original_spacing / target_spacing
  new_size = (np.array(sitk_img.GetSize()) * (original_spacing / target_spacing))
  new_size = np.ceil(new_size).astype(np.int64).tolist()

  print("Resampling...")
  # execute resampling
  resample = sitk.ResampleImageFilter()
  resample.SetOutputSpacing(target_spacing)
  resample.SetSize(new_size)
  resample.SetInterpolator(sitk.sitkBSpline)

  # ensure origin/direction are preserved (critical for coordinate tracking)
  resample.SetOutputOrigin(sitk_img.GetOrigin())
  resample.SetOutputDirection(sitk_img.GetDirection())
  resample.SetDefaultPixelValue(0.0) # Set background to 0 (since data is normalized)

  resampled_img = resample.Execute(sitk_img)

  # final Array (Ready for Cropping/Padding)
  resampled_array = sitk.GetArrayFromImage(resampled_img).astype(np.float32)

  return resampled_array

def crop_and_pad(arr, TARGET_SHAPE, pad_val=0.0):
  current_shape = np.array(arr.shape)
  target_shape = np.array(TARGET_SHAPE)
    
    # Check for non-3D input immediately (as we discussed, this is critical)
  if arr.ndim != 3:
        raise ValueError(
            f"Input array has {arr.ndim} dimensions (Shape: {arr.shape}), "
            f"but expected 3D array for target {TARGET_SHAPE}."
        )

  # Prepare lists for slicing and padding amounts across all 3 dimensions
  slicing_list = []
  padding_list = []
  
  for current_dim, target_dim in zip(current_shape, target_shape):
      
      # --- 1. Calculate the difference ---
      diff = current_dim - target_dim

      if diff > 0:
          # Case: Current is LARGER than target (Needs Cropping)
          # Center the crop by determining the excess to remove from start/end
          crop_start = diff // 2
          crop_end = current_dim - (diff - crop_start) # Ensures remainder is removed from the end
          
          slicing_list.append(slice(crop_start, crop_end))
          padding_list.append((0, 0)) # No padding needed
      
      elif diff < 0:
          # Case: Current is SMALLER than target (Needs Padding)
          # Center the padding by determining the shortfall to add to start/end
          pad_needed = -diff
          pad_start = pad_needed // 2
          pad_end = pad_needed - pad_start # Ensures remainder is added to the end
          
          slicing_list.append(slice(0, current_dim)) # Take the whole array
          padding_list.append((pad_start, pad_end))
          
      else:
          # Case: Current == Target (No change needed)
          slicing_list.append(slice(0, current_dim))
          padding_list.append((0, 0))


  print(f"Cropping/Padding: Slices: {slicing_list}, Padding: {padding_list}")
  
  # 2. Perform Cropping (using the pre-calculated slices)
  # The '*' unpacks the list of slice objects to apply to the array indices
  cropped_array = arr[tuple(slicing_list)]
  
  # 3. Perform Padding (using the pre-calculated padding list)
  padded_array = np.pad(
      cropped_array,
      padding_list,
      mode='constant',
      constant_values=pad_val
  )

  # 4. Final Validation Check (This is the most important debugging step)
  if padded_array.shape != TARGET_SHAPE:
      raise ValueError(
          f"Padded array output shape {padded_array.shape} "
          f"did not match target {TARGET_SHAPE}. Debug failed sample!"
      )

  return padded_array.astype(arr.dtype)

# get bounding boxes
def parse_bboxes(xml_path):
  tree = etree.parse(xml_path)
  root = tree.getroot()
  sop_uid = os.path.basename(xml_path)
  label = 0

  for obj in root.findall('.//object'):
    # get classifiers
    diagnosis = obj.findtext('./name')
    match diagnosis:
      case 'A'|'a':
          diagnosis = "Adenocarcinoma"
          label = 1
      case 'B'|'b':
          diagnosis = "Small Cell Carcinoma"
          label = 2
      case 'G'|'g':
          diagnosis = "Squamous Cell Carcinoma"
          label = 3
      case _:
          diagnosis = None
          label = 0

    # get bounding box coordinates
    bbox_node  = obj.find('./bndbox')
    x1 = int(bbox_node.findtext('xmin')); y1 = int(bbox_node.findtext('ymin'))
    x2 = int(bbox_node.findtext('xmax')); y2 = int(bbox_node.findtext('ymax'))
    res = {'sop': sop_uid, 'label': label, 'diagnosis': diagnosis, 'bbox': [x1,y1,x2,y2]}
  return res


def make_label_from_bboxes(transformed_bboxes, num_classes=3):
    """
    Create a one-hot 3-class label from bounding boxes.
    - Each bbox is [..., label]
    - label == 0  → background (ignored)
    - label in {1,2,3} → tumor types 0–2 after shifting
    """
    # Extract raw labels
    raw_labels = [int(b[-1]) for b in transformed_bboxes]

    # Keep only non-background labels
    tumor_labels = [l for l in raw_labels if l > 0]

    if not tumor_labels:
        raise ValueError("No non-background tumor labels found in bboxes.")

    # Convert 1–4 → 0–3
    zero_based = [l - 1 for l in tumor_labels]

    for l in zero_based:
        if l < 0 or l >= num_classes:
            raise ValueError(f"Label {l} out of range for num_classes={num_classes}")

    # Pick dominant tumor type
    counts = Counter(zero_based)
    dominant_label = counts.most_common(1)[0][0]  # integer 0–3

    onehot_vector = tf.keras.utils.to_categorical(dominant_label, num_classes=num_classes)
    return onehot_vector

    

# apply same preprocessing to coordinates as image volume
def transform_coords(coords_list, original_shape, target_shape, original_spacing, target_spacing, global_z_offset):
  print("Transforming coordinates...")
  res_coords = []
  target_shape = np.array(target_shape) #(Z, Y X)

  scale_factor = original_spacing / target_spacing
  shape_diff = np.array(original_shape) - target_shape

  # global offset for non-tumors
  global_crop_offset = np.maximum(0, shape_diff // 2)
  TARGET_CENTER_Y = target_shape[1] / 2.0
  TARGET_CENTER_X = target_shape[2] / 2.0

  # bbox = [Zmin, Xmin, Ymin, Zmax, Xmax, Ymax, Label]
  for bbox in coords_list:

    # 1. Rescale (Apply the sampling factor)
    z_coords = np.array([bbox[0], bbox[3]]) * scale_factor[2]
    y_coords = np.array([bbox[2], bbox[5]]) * scale_factor[1]
    x_coords = np.array([bbox[1], bbox[4]]) * scale_factor[0]

    # if label is present, calculate offset based on tumor
    if bbox[6] != 0:
      # print("TUMOR INDICATED: ", bbox)

      # Calculate the center of the rescaled tumor bounding box
      tumor_center_Y = (y_coords[0] + y_coords[1]) / 2
      tumor_center_X = (x_coords[0] + x_coords[1]) / 2

      # Calculate the local offset needed to center the tumor
      local_offset_Y = tumor_center_Y - TARGET_CENTER_Y
      local_offset_X = tumor_center_X - TARGET_CENTER_X

      # 2. Shift (apply crop offset)
      new_z_coords = z_coords - global_z_offset
      new_y_coords = y_coords - local_offset_Y
      new_x_coords = x_coords - local_offset_X

    # else use global center offset
    else:
      # 2. Shift (Apply the crop offset)
      new_z_coords = z_coords - global_crop_offset[0] # Offset in Z (depth)
      new_y_coords = y_coords - global_crop_offset[1] # Offset in Y (height)
      new_x_coords = x_coords - global_crop_offset[2] # Offset in X (width)

    # 3. Clip to bounds (128) and round to integer indices
    new_z_coords = np.clip(new_z_coords, 0, target_shape[0]).astype(int)
    new_y_coords = np.clip(new_y_coords, 0, target_shape[1]).astype(int)
    new_x_coords = np.clip(new_x_coords, 0, target_shape[2]).astype(int)

    # Reconstruct the new 3D bounding box
    transformed_bbox = [
        new_z_coords[0], new_x_coords[0], new_y_coords[0], # Min Corner (Z, X, Y)
        new_z_coords[1], new_x_coords[1], new_y_coords[1], # Max Corner (Z, X, Y)
        bbox[6] # Label
    ]

    res_coords.append(transformed_bbox)

  return res_coords

#-------- Final Function for raw data ---------#

 #takes in dataset map for ONE series, will later need to refactor for multiple series
def preprocess_data(dataset_map):
  try:
    # get parent directory and sampleID of series
    firstKey = next(iter(dataset_map))
    fistSampleId = dataset_map[firstKey]['sampleID']
    parentDir = os.path.dirname(firstKey)
    ds = pydicom.dcmread(firstKey)

    # get all .dcm files in the series in a list
    fileNames = []
    annotations_extracted = []
    for filepath, metadata in dataset_map.items():

      #-------------- map annotations ----------------#
      tumorCoords = []

      if dataset_map[filepath]['is_annotated'] is True:

        zCoord = dataset_map[filepath]['z_coord']
        res = parse_bboxes(dataset_map[filepath]['annotation_path'])
        bbox = res.get('bbox', 'Bounding box not found.')

        # create bounding box coordinates
        tumorCoords = [zCoord, bbox[0], bbox[1],
                       zCoord, bbox[2], bbox[3],
                       res.get('label', 'Label Error')]
      else:
        tumorCoords = [0, 0, 0, 0, 0, 0, 0]
      

      annotations_extracted.append(tumorCoords)
      #-----------------------------------------------#
      fileNames.append(filepath)

    # reconstruct 3d volume
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(fileNames)
    sitk_image_volume = reader.Execute()

    og_spacing = sitk_image_volume.GetSpacing()
    spacing_np = np.array(og_spacing)
    FALLBACK_SPACING = 5.0

    if spacing_np[2] < 0.001:
        print("WARNING: zero z-spacing detected. setting Z-spacing to 5.0mm")
        spacing_np[2] = FALLBACK_SPACING
        sitk_image_volume.SetSpacing(spacing_np.tolist())
    
    og_spacing = sitk_image_volume.GetSpacing() # update var for next call
    target_spacing = np.array([1.0, 1.0, 1.0])

    print(f"Successfully read image volume.")

    # convert sitk img to np arr for preprocessing
    img3d = sitk.GetArrayFromImage(sitk_image_volume)

    # get normalized data
    norm_data = normalize_data(ds, img3d)
    resampled_array = resample_data(norm_data,
                                    og_spacing,
                                    target_spacing)

    # define target (D x H x W) shape for 3D CNN input
    ORIGINAL_SHAPE = np.array(resampled_array.shape)
    

    global_z_offset = calculate_series_z_offset(annotations_extracted,
                                                og_spacing,
                                                target_spacing)
    # resample coordinates as well
    transformed_coords = transform_coords(annotations_extracted,
                                          ORIGINAL_SHAPE,
                                          TARGET_SHAPE,
                                          og_spacing,
                                          target_spacing,
                                          global_z_offset)

    # for i, arr in enumerate(transformed_coords):
    #   print(i, ": ", arr)

    # mask_volume = generate_mask(transformed_coords, TARGET_SHAPE)
    print("Transformed bboxes: ", transformed_coords)

    onehot_vector = make_label_from_bboxes(transformed_coords, num_classes=3)
    print("One-hot vector: ", onehot_vector)

    # CNN requires fixed input size (D x H x W)
    final_tensor = crop_and_pad(resampled_array, TARGET_SHAPE, 0.0)

    # CHECK A: Image Tensor Shape
    if final_tensor.shape != TARGET_SHAPE:
        print("\n!!! FATAL IMAGE SHAPE MISMATCH !!!")
        print(f"Expected Shape: {TARGET_SHAPE}. Actual Shape: {final_tensor.shape}")
        # Raise a clear error that includes the failing shape
        raise ValueError(f"Image shape error on sample: {final_tensor.shape}")
        
    # CHECK B: One-Hot Vector Shape
    if onehot_vector.shape != TARGET_LABEL_SHAPE:
        print("\n!!! FATAL LABEL SHAPE MISMATCH !!!")
        print(f"Expected Shape: {TARGET_LABEL_SHAPE}. Actual Shape: {onehot_vector.shape}")
        # Raise a clear error that includes the failing shape
        raise ValueError(f"One-hot shape error on sample: {onehot_vector.shape}")


    print(f"Tensor for series {fistSampleId} complete.")

    return final_tensor, onehot_vector

  except Exception as e:
    print(f"!!! PROCESSING FAILED for sample {fistSampleId}: {e}")
    return None, None


#------- functions for preparing tensorflow dataset --------#

# reshape to CNN
def cast_and_reshape(image, label):
    # our dataset is alr float32, but just in case
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    # Add a channel dimension (C=1) if it's missing (D, H, W) -> (D, H, W, 1)
    image = tf.expand_dims(image, axis=-1)

    return image, label

# batch and prefetch to each split separately
def prepare_dataset(ds, batch_size, shuffle=False, repeat=False):
    ds = ds.map(cast_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(10_000)
    
    ds = ds.batch(batch_size)

    if repeat:
        ds = ds.repeat()

    return ds.prefetch(tf.data.AUTOTUNE)
