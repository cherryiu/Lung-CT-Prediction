from lxml import etree
from pathlib import Path
import math

# get sorting key to sort series of DICOM slices by
def sort_key(slice_element):
    # ignore path, get z coordinate or instance number as key to sort by
    _, _, instanceNum, zCoord, _ = slice_element
    if zCoord is not None:
        return zCoord
    return int(instanceNum) if instanceNum is not None else 0

# find XML annotation according to SOP Instance ID
def find_xml_for_slice(slice_element, annotations_root):
  sampleName = slice_element[0]
  uid = slice_element[4]
  
  annotations = Path(annotations_root)
  cand = annotations.joinpath(sampleName, f'{uid}.xml')
  return str(cand) if cand.is_file() else None

# # find corresponding dicom image to xml (testing)
# def find_dicom_for_slice(xml_path):
#   xml_name = xml_path.split('/')[-1]
#   dcm_path = dict[xml_name[:-4]]
#   print(dcm_path)


# get z offset for center cropping 
def calculate_series_z_offset(tumor_coords, og_spacing, target_spacing):
  # 1. Identify all tumor bounding boxes (exclude placeholders)
  # bbox format: [Zmin, Xmin, Ymin, Zmax, Xmax, Ymax, Label]
  tumor_bboxes = [b for b in tumor_coords if b[6] != 0]

  if not tumor_bboxes:
      # If no tumor is present, the offset is irrelevant (return 0 or handle separately)
      return 0

  # Get the Z-scaling factor (assuming original_spacing is X, Y, Z order)
  z_scale_factor = og_spacing[2] / target_spacing[2]

  # Find the ABSOLUTE Z-min and Z-max across the entire tumor volume (original physical coords)
  global_z_min_orig = np.min([b[0] for b in tumor_bboxes])
  global_z_max_orig = np.max([b[3] for b in tumor_bboxes])

  # 2. Rescale the global boundaries
  global_z_min_scaled = global_z_min_orig * z_scale_factor
  global_z_max_scaled = global_z_max_orig * z_scale_factor

  # 3. Calculate the single GLOBAL Z-Offset
  TARGET_CENTER = 128 / 2  # Assuming target shape is 128x128x128
  global_tumor_center_Z = (global_z_min_scaled + global_z_max_scaled) / 2

  # 4. Calculate the offset: Center of tumor minus center of target window
  global_offset_Z = global_tumor_center_Z - TARGET_CENTER

  return global_offset_Z

# getting ratios
def split_tf_dataset(params, dataset, total_size):

  # Define split ratios
  TRAIN_RATIO = params['train_ratio']
  VAL_RATIO = params['val_ratio']
  TEST_RATIO = params['test_ratio']

  # Calculate exact sizes
  train_size = math.floor(TRAIN_RATIO * total_size)
  val_size = math.floor(VAL_RATIO * total_size)
  # The remaining instances go to the test set to account for floor rounding
  test_size = total_size - train_size - val_size

  # Shuffle the ENTIRE dataset
  dataset = dataset.shuffle(buffer_size=params['buffer_size'])

  # Extract the splits
  train_dataset = dataset.take(train_size)
  temp_test_val = dataset.skip(train_size)

  val_dataset = temp_test_val.take(val_size)
  test_dataset = temp_test_val.skip(val_size)

  return dataset, train_dataset, val_dataset, test_dataset
