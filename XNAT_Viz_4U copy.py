import pickle
import os
import sys
import pydicom
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

import logging
from datetime import datetime
import streamlit as st

#-------- Configurable Paths --------#
script_path = os.path.abspath(sys.argv[0])
script_directory = os.path.dirname(script_path)
log_file_path = os.path.join(script_directory, "streamlit_logs.txt")
pickle_file_path = os.path.join(script_directory, "streamlit_cache.pkl")

#-------- Pickle Functions (Cache) -------------#
def load_pickle_file():
    """
    Loads and returns a pickle file. If the file does not exist, returns an empty dictionary.
    Updates the session state variable 'pickle_cache'.
    """
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}
    
    st.session_state = data
    return data

def save_pickle_cache():
    """
    Saves the 'pickle_cache' from the session state to a pickle file.
    """
    if 'pickle_cache' in st.session_state:
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(st.session_state['pickle_cache'], f)
    else:
        raise ValueError("Session state does not contain 'pickle_cache'.")

#-------- Logging Function --------#
def append_log(log_message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"-------\n{current_time} - {log_message}\n"
    try:
        with open(log_file_path, "a") as file:
            file.write(log_entry)
    except Exception as e:
        st.error(f"Failed to append log due to: {e}")

#-----------  Paths Function -----------#
def get_folders_in_directory(directory_path):
    """
    Returns a list of paths to folders in the given directory.

    Parameters:
    - directory_path: Path to the directory.

    Returns:
    - List of paths to folders in the directory.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The provided path '{directory_path}' is not a directory or does not exist.")

    folders = {name: os.path.join(directory_path, name) for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))}
    return folders

def assessors_path(patient_directory: str) -> Optional[str]:
    """
    This function takes the path to a patient directory and returns the path of the ASSESSORS directory if it exists.
    
    Parameters:
    - patient_directory (str): The directory path of the patient.
    
    Returns:
    - Optional[str]: The path to the ASSESSORS directory if it exists, otherwise None.
    """
    
    assessors_dir = os.path.join(patient_directory, 'ASSESSORS')
    if not os.path.exists(assessors_dir):
        logging.info("No segmentation exists")
        return None
    logging.info(f"'ASSESSORS' folder exists: {assessors_dir}")
    return assessors_dir


def get_segmentations_from_assessors_path(assessors_path: str) -> Dict[str, Dict[str, Any]]:
    """
    This function searches the ASSESSORS folder and creates a dictionary of all segmentations.
    
    Parameters:
    - assessors_path (str): The path to the ASSESSORS directory.
    
    Returns:
    - Dict[str, Dict[str, Any]]: A dictionary with the name of the segmentor and datetime of segmentation as keys. 
      Each key maps to another dictionary containing 'created_by', 'created_time', 'dicom_name', and 'dicom_fullpath'.
    """
    segmentation_paths = [d for d in os.listdir(assessors_path) if os.path.isdir(os.path.join(assessors_path, d))]
    segmentations = {}

    for seg in segmentation_paths:
        seg_dir = os.path.join(assessors_path, seg, 'SEG')
        if os.path.exists(seg_dir):
            files = os.listdir(seg_dir)
            
            # Find XML files
            xml_files = [f for f in files if f.endswith('.xml')]
            
            for xml_file in xml_files:
                xml_path = os.path.join(seg_dir, xml_file)
                try:
                    # Parse the XML file
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # Extract createdBy and createdTime
                    entry = root.find('.//cat:entry', namespaces={'cat': 'http://nrg.wustl.edu/catalog'})
                    if entry is not None:
                        created_by = entry.get('createdBy')
                        created_time = entry.get('createdTime')
                        dicom_name = entry.get('name')
                        dicom_fullpath = os.path.join(seg_dir, dicom_name)
                        
                        # Save to dictionary
                        key = f"{created_by}>>{created_time}"
                        segmentations[key] = {
                            'created_by': created_by,
                            'created_time': created_time,
                            'dicom_name': dicom_name,
                            'dicom_fullpath': dicom_fullpath
                        }
                        logging.info(segmentations[key])
                except ET.ParseError as e:
                    logging.error(f"Error parsing XML file {xml_file}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")

    return segmentations

def select_segmentation_from_valid(all_segmentation_dic):
    """
    This function allows the user to select a segmentation from the dictionary of segmentations.
    
    Parameters:
    - all_segmentation_dic (Dict[str, Dict[str, Any]]): Dictionary containing segmentations.
    - default_selected_segmentation (str, optional): Default selected segmentation. Defaults to None.
    
    Returns:
    - Optional[str]: The full path to the selected DICOM file, or None if not found.
    """
    valid_options_to_select = list(all_segmentation_dic.keys())
    len_options_to_select = len(valid_options_to_select)
    
    if len_options_to_select == 0:
        st.warning("No valid segmentations available.")
        return None

    show_options_to_select = "\n".join([f"{i}: {valid_options_to_select[i]}" for i in range(len_options_to_select)])
    selected_segmentation_name = st.radio("Choose a segmentation to proceed ", options=valid_options_to_select)
        
    segmentation_info = all_segmentation_dic.get(selected_segmentation_name)
    if segmentation_info:
        logging.info(f"Selected SEG: {selected_segmentation_name}")
        logging.info(f"Path to selected SEG: {segmentation_info.get('dicom_fullpath')}")
        st.session_state["selected_segmentation_path"] = segmentation_info.get('dicom_fullpath')
    else:
        st.session_state["selected_segmentation_path"] = None


#---------------DICOM functions--------------#
def read_dicom(dicom_path):
    """
    Reads a DICOM file from the given path.

    Parameters:
    - selected_SEGdicom_fullpath: The full path to the DICOM file.

    Returns:
    - The DICOM dataset if read successfully, otherwise None.
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        logging.info(f"Dicom data loaded from: {dicom_path}.")
        return dicom_data
    except Exception as e:
        logging.error(f"Error in reading '{dicom_path}': {e}")
        return None

def get_nested_element(dataset, tags):
    """
    Navigate through the DICOM dataset using a list of tags and return the final element.

    Parameters:
        dataset (pydicom.dataset.Dataset): The DICOM dataset.
        tags (list): A list of tuples representing the tags to navigate through.

    Returns:
        The final element in the DICOM dataset specified by the tags.
    """
    current_element = dataset
    for tag in tags:
        tag = pydicom.tag.Tag(tag)
        if tag in current_element:
            current_element = current_element[tag]
        else:
            raise KeyError(f"Tag {tag} not found in the dataset.")
        
        # If the current element is a sequence, assume we want the first item
        if isinstance(current_element, pydicom.sequence.Sequence):
            if len(current_element) > 0:
                current_element = current_element[0]
            else:
                raise ValueError(f"Sequence at tag {tag} is empty.")
    
    return current_element

#---------------Segmentation File functions--------------#
def get_referenced_series_UID(segmentation_dicom) -> str:
    """
    Extract a list of Referenced SOP Instance UIDs from a DICOM segmentation dataset.

    Parameters:
        segmentation_dicom (pydicom.dataset.Dataset): The DICOM dataset containing segmentation data.

    Returns:
        list: A list of Referenced SOP Instance UIDs.
    """
    try:
        referenced_series_sequence = get_nested_element(segmentation_dicom, [(0x0008, 0x1115)])
        for series_instance in referenced_series_sequence:
            if 'SeriesInstanceUID' in series_instance:
                Ref_series_UID = series_instance.SeriesInstanceUID
                logging.info(f"Referenced Series Instance UID: {Ref_series_UID}")
                return Ref_series_UID
    except (KeyError, ValueError) as e:
        logging.error(f"Error extracting Referenced Series Instance UID: {e}")


def create_segment_number_to_label_map(segmentation_dicom) -> dict:
    """
    Create a dictionary mapping Segment Number to Segment Label from a DICOM segmentation object.

    Parameters:
        segmentation_dicom (pydicom.dataset.Dataset): The DICOM dataset containing segmentation data.

    Returns:
        dict: A dictionary mapping Segment Number (int) to Segment Label (str).
    """
    segment_map = {}
    try:
        segment_sequence = get_nested_element(segmentation_dicom, [(0x0062, 0x0002)])
        for item in segment_sequence:
            if (0x0062, 0x0004) in item and (0x0062, 0x0005) in item:
                segment_number = item[(0x0062, 0x0004)].value
                segment_label = item[(0x0062, 0x0005)].value
                segment_map[segment_number] = segment_label
        logging.info(f"Segment map: {segment_map}")
    except (KeyError, ValueError) as e:
        logging.error(f"Error creating segment map: {e}")
    return segment_map

def get_segmentation_data_including_RefSOPUID_refSegNum_pixelData(segmentation_dicom, segment_map):
    """
    Reads a segmentation DICOM file and creates a dictionary for each slice.

    Parameters:
    - segmentation_dicom: The DICOM dataset for the segmentation.
    - segment_map: Dictionary to map segment numbers to segment labels.

    Returns:
    - Dictionary with segment labels as keys and lists of dictionaries as values. Each dictionary contains 'pixel_data' and 'sop_instance_uid'.
    """
    segmentation_data = {}

    try:
        # Get the number of frames
        num_frames = segmentation_dicom.NumberOfFrames
        logging.info(f"Number of frames in the DICOM: {num_frames}")

        # Get pixel data
        pixel_data = segmentation_dicom.pixel_array

        # Get referenced instance UID and segmentation number for each frame
        for frame_index in range(num_frames):
            try:
                frame = segmentation_dicom.PerFrameFunctionalGroupsSequence[frame_index]
                referenced_sop_instance_uid = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                segment_number = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                
                if segment_number in segment_map:
                    segment_label = segment_map[segment_number]
                    slice_info = {
                        'pixel_data': pixel_data[frame_index],
                        'sop_instance_uid': referenced_sop_instance_uid
                    }
                    if segment_label not in segmentation_data:
                        segmentation_data[segment_label] = []
                    segmentation_data[segment_label].append(slice_info)
                else:
                    logging.warning(f"Segment number {segment_number} not found in segment_map.")
                    
            except Exception as e:
                logging.error(f"Error processing frame {frame_index}: {e}")

        # Log the count of slices for each segment label
        for segment_label, slices in segmentation_data.items():
            logging.info(f"Segment '{segment_label}': {len(slices)} slices")

    except AttributeError as e:
        logging.error(f"Error reading DICOM attributes: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return segmentation_data

#--------------- Original CT Functions ----------#

def extract_series_data(patient_directory):
    """
    Extracts series data from XML files within the SCANS directory of the given patient directory.

    Parameters:
    - patient_directory: Path to the patient's directory containing the SCANS folder.

    Returns:
    - Dictionary containing series data.
    """
    scans_dir = os.path.join(patient_directory, 'SCANS')
    series_folders = [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    
    series_data = {}

    for series in series_folders:
        series_path = os.path.join(scans_dir, series, 'DICOM')
        xml_files = [f for f in os.listdir(series_path) if f.endswith('.xml')]

        if not xml_files:
            logging.warning(f"No XML files found in series directory: {series_path}")
            continue

        # Assuming there's only one XML file per series directory
        xml_file = xml_files[0]
        xml_path = os.path.join(series_path, xml_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract original_RefSeriesUID
            ref_series_uid = root.attrib.get('UID', None)
            if not ref_series_uid:
                logging.warning(f"No UID found in XML root for series {series}")
                continue

            # Extract original_RefInstanceUID_dict
            ref_instance_uid_dict = {}
            for entry in root.findall('.//cat:entry', namespaces={'cat': 'http://nrg.wustl.edu/catalog'}):
                uid = entry.attrib.get('UID')
                uri = entry.attrib.get('URI')
                if uid and uri:
                    ref_instance_uid_dict[uid] = os.path.join(series_path, uri)

            series_data[series] = {
                'original_RefSeriesUID': ref_series_uid,
                'original_RefInstanceUID_dict': ref_instance_uid_dict
            }
        except ET.ParseError as e:
            logging.error(f"Error parsing XML file {xml_file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing series {series}: {e}")
            
    logging.info(f"Successfully read the XML file for {patient_directory}. Series and slice counts:")
    for key, value in series_data.items():
        original_RefInstanceUID_dict = value.get('original_RefInstanceUID_dict',{})
        logging.info(f"    {key}: {len(original_RefInstanceUID_dict)}")    
        
    return series_data

def read_dicom_pixel_data_with_sop_instance_uid(directory_path):
    """
    Reads pixel data and referenced SOP Instance UID from all DICOM files in the given directory.

    Parameters:
    - directory_path: Path to the directory containing DICOM files.

    Returns:
    - List of dictionaries with 'pixel_data', 'sop_instance_uid', and 'instance_number'.
    """
    dicom_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.dcm')]
    dicom_data_list = []
    
    for dicom_file in dicom_files:
        try:
            dicom_data = pydicom.dcmread(dicom_file)
            sop_instance_uid = dicom_data.SOPInstanceUID
            pixel_data = dicom_data.pixel_array
            instance_number = dicom_data.InstanceNumber if hasattr(dicom_data, 'InstanceNumber') else 0
            
            dicom_data_list.append({
                'pixel_data': pixel_data,
                'sop_instance_uid': sop_instance_uid,
                'instance_number': instance_number
            })
        except Exception as e:
            logging.error(f"Error reading DICOM file '{dicom_file}': {e}")
    
    # Sort by InstanceNumber
    dicom_data_list.sort(key=lambda x: x['instance_number'])
    
    return dicom_data_list


def get_path_to_matched_original_series(original_series_data,Ref_series_UID):
    for original_folder, original_info_dic in original_series_data.items():
        if original_info_dic['original_RefSeriesUID']==Ref_series_UID:
            logging.info(f"Successfully find a match for series UIDs in folder: '{original_folder}'")
            matched_original_series_data = read_dicom_pixel_data_with_sop_instance_uid(os.path.join(st.session_state['selected_patient_directory'],"SCANS", original_folder,"DICOM"))
    if matched_original_series_data:
        logging.info(" The matched series of original image created successfully.")
        return matched_original_series_data
    else:
        logging.warning("No matched series UID was ")

#-----------Mergge Functions----------------------#
# Function to merge dictionaries
def merge_dictionaries(original_dic, segment_dic):
    merged_data = []
    segment_lookup = {item['sop_instance_uid']: item['pixel_data'] for item in segment_dic}
    
    for original_item in original_dic:
        sop_instance_uid = original_item['sop_instance_uid']
        segmentation_pixel = segment_lookup.get(sop_instance_uid, None)
        
        merged_data.append({
            'sop_instance_uid': sop_instance_uid,
            'original_pixel': original_item['pixel_data'],
            'segmentation_pixel': segmentation_pixel
        })
    
    return merged_data

# Function to prepare images
def prepare_images(merged_data, segment_overlay_transparency, segment_overlay_color, figsize=(10, 10), label_for_overlay_image=""):
    prepared_images = []
    overlay_rgba = to_rgba(segment_overlay_color, alpha=segment_overlay_transparency)
    
    for data in merged_data:
        original_pixel = data['original_pixel']
        segmentation_pixel = data['segmentation_pixel']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(original_pixel, cmap=plt.cm.gray)
        
        if segmentation_pixel is not None:
            overlay = np.zeros((*segmentation_pixel.shape, 4))
            overlay[..., :3] = overlay_rgba[:3]
            overlay[..., 3] = (segmentation_pixel > 0) * segment_overlay_transparency
            
            ax.imshow(overlay, cmap=None, alpha=segment_overlay_transparency)
        
        ax.set_title(f"SOP Instance UID: {data['sop_instance_uid']}\n{label_for_overlay_image}")
        ax.axis('off')
        fig.canvas.draw()
        
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        prepared_images.append(image)
        
        plt.close(fig)
    
    return prepared_images

# Function to show merged DICOM stack in Streamlit
def show_merged_dicom_stack_fast(prepared_images):
    try:
        i = st.slider("Slice scroll", min_value=0, max_value=len(prepared_images)-1, step=1)
        st.image(prepared_images[i], caption=f"Slice {i + 1}", use_column_width=True)
    except:
        print("nothing to show  ")

#-------- Streamlit App --------#
def main():
    st.title("XNAT Project Viewer")
    
    with st.sidebar:
        project_directory = st.text_input("Enter the path to the XNAT project with patients data in folders:")
        if project_directory:
            try:
                patient_folders = get_folders_in_directory(project_directory)
                if patient_folders:
                    patient_directory = st.selectbox("Select a patient folder", options=list(patient_folders.keys()))
                    st.session_state['selected_patient_directory'] = patient_folders[patient_directory]
                    if patient_folders:
                        st.write(patient_directory)
                        ass_path = assessors_path(os.path.join(project_directory, patient_directory))                    
                        all_segmentation_dic = get_segmentations_from_assessors_path(ass_path)
                        select_segmentation_from_valid(all_segmentation_dic)
                        if st.session_state["selected_segmentation_path"]:
                            selected_SEGdicom_data = read_dicom(st.session_state["selected_segmentation_path"])
                            if selected_SEGdicom_data:
                                st.session_state['Ref_series_UID'] = Ref_series_UID = get_referenced_series_UID(selected_SEGdicom_data)
                                st.session_state['segment_map'] = segment_map  = create_segment_number_to_label_map(selected_SEGdicom_data)
                                st.session_state['slice_info_list'] = slice_info_list = get_segmentation_data_including_RefSOPUID_refSegNum_pixelData(selected_SEGdicom_data,segment_map)
                                
                                segmentations_to_chose = "\n".join([f"Segment '{segment_label}': {len(slices)} slices" for segment_label, slices in slice_info_list.items()])
                                
                                selected_segmentation_label= st.radio(f"Select one of the segmentation labels ", options=list(slice_info_list.keys(),), key='segment_label')
                                st.write(f"Current segmentaions: {segmentations_to_chose}")
                                if selected_segmentation_label:
                                    st.session_state['selected_segmentation_data_dic'] = slice_info_list[selected_segmentation_label]
                                    st.write("Segmentation data successfully loaded")           
            except ValueError as e:
                st.error(e)
                
    if st.session_state['selected_patient_directory'] and st.session_state['selected_segmentation_data_dic'] and st.session_state['Ref_series_UID']:
        # Example usage
        original_series_data = extract_series_data(st.session_state['selected_patient_directory'])
        matched_series_original_data_dic = get_path_to_matched_original_series(original_series_data,st.session_state['Ref_series_UID'])
        
        merged_data = merge_dictionaries(matched_series_original_data_dic, st.session_state['selected_segmentation_data_dic'])
        st.session_state["prepared_images"] = prepared_images = prepare_images(merged_data, segment_overlay_transparency=0.4, segment_overlay_color='#FFA500', figsize=(10, 10), label_for_overlay_image=st.session_state['segment_label'])
    
    show_merged_dicom_stack_fast(prepared_images)

if __name__ == "__main__":
    # Load the cache at the start
    load_pickle_file()
    
    # Run the main app
    main()
    
    # Save the cache at the end
    save_pickle_cache()
