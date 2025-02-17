import argparse
import os
import random
import glob
import json
import sys
import time
from pathlib import Path
import numpy as np
import blenderproc as bproc
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.loader.ShapeNetLoader import _ShapeNetLoader
import bmesh
import bpy



def get_obj_min_width(obj):
    """
    Get the minimum width of an object.

    Parameters:
    -----------
    obj : bproc.object
        The object to measure

    Returns:
    --------
    float
        The minimum width of the object
    """
    bbox = obj.get_bound_box()
    bbox = np.array(bbox)
    min_corner = np.min(bbox, axis=0)
    max_corner = np.max(bbox, axis=0)
    dimensions = max_corner - min_corner
    min_dim_idx = np.argmin(dimensions)
    return dimensions[min_dim_idx]

def position_and_align_object(obj, obj_cv, robot):
    """
    Aligns an object to match the orientation of the robot's hand/gripper,
    with the longest dimension pointing upward and the smallest dimension
    between the fingers.
    
    Parameters:
    obj: blenderproc.object.Object
        The object to align
    robot: blenderproc.loader.URDFLoader
        The loaded robot model
    """

    # We scale the object to the gripper
    scale_object_to_hand(obj, robot)

    # Get the hand's transformation matrix
    joint_positions = robot.get_all_local2world_mats()
    
    # Get finger positions for gripper orientation
    l_finger_pos = joint_positions[35][:3, 3]  # Left finger position
    r_finger_pos = joint_positions[41][:3, 3]  # Right finger position
    
    # Calculate grasp direction (between fingers)
    grasp_direction = l_finger_pos - r_finger_pos
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    
    # Define world up vector
    world_up = np.array([0, 0, 1])
    
    # Get object's bounding box
    bbox = obj.get_bound_box()
    bbox_np = np.array(bbox)
    
    # Calculate bounding box dimensions
    min_corner = np.min(bbox_np, axis=0)
    max_corner = np.max(bbox_np, axis=0)
    dimensions = max_corner - min_corner
    
    # Find dimension indices sorted by size
    dim_indices = np.argsort(dimensions)
    min_dim_idx = dim_indices[0]  # Smallest (for grasping)
    mid_dim_idx = dim_indices[1]  # Middle
    max_dim_idx = dim_indices[2]  # Largest (should point up)

    # Step 1: First align the smallest dimension with grasp direction
    rotation_matrix = np.eye(3)
    rotation_matrix[:, min_dim_idx] = grasp_direction
    
    # Step 2: Project world up onto the plane perpendicular to grasp direction
    up_proj = world_up - np.dot(world_up, grasp_direction) * grasp_direction
    up_proj = up_proj / np.linalg.norm(up_proj)
    
    # This will be aligned with the longest dimension
    rotation_matrix[:, max_dim_idx] = up_proj
    
    # Step 3: Complete the right-handed coordinate system for the middle dimension
    middle_axis = np.cross(up_proj, grasp_direction)
    middle_axis = middle_axis / np.linalg.norm(middle_axis)
    rotation_matrix[:, mid_dim_idx] = middle_axis
    
    # Ensure the rotation matrix is orthogonal
    u, _, vh = np.linalg.svd(rotation_matrix)
    rotation_matrix = u @ vh
    
    # Position object between fingers
    center_grasp_pos = (l_finger_pos + r_finger_pos) / 2
    obj.set_location(center_grasp_pos)
    
    # Apply rotation to object
    obj.set_rotation_mat(rotation_matrix)
    obj.hide(False)

    local2world = obj.get_local2world_mat()
    # We set up the convex hull object
    obj_cv.hide(False)
    obj_cv.set_scale(obj.get_scale())
    obj_cv.set_location(local2world[:3,3])
    obj_cv.set_rotation_mat(local2world[:3,:3])

    # return obj.get_local2world_mat(),dimensions[min_dim_idx]

def scale_object_to_hand(obj, robot):
    """
        Given the object and the robot, it scales the object accordingly to the graps size. 
        To do so, it calculates the distance between the fingers and scales the object to match that distance multiplied by
        a random factor between 0.05 and 0.5.
        Furthermore, if the object's height is too big, it scales it accordingly. In this case the object does not preserve its aspect ratio.
    """
    max_height = 0.15
    bbox_3d = obj.get_bound_box()
    bbox_3d = np.array(bbox_3d)
    min_corner = np.min(bbox_3d, axis=0)
    max_corner = np.max(bbox_3d, axis=0)
    dimensions = max_corner - min_corner  # Size along each local axis
    
    joint_positions = robot.get_all_local2world_mats()
            

    # Middle of fingers
    l_finger_pos = joint_positions[35][:3, 3]
    r_finger_pos = joint_positions[41][:3, 3]

    maximum_width = np.linalg.norm(l_finger_pos - r_finger_pos)

    scale = [maximum_width/min(dimensions)*random.uniform(0.1,.5) for _ in range(3)]
    print(scale)
    if max(dimensions)>max_height:
        scale_max = max_height/max(dimensions)*random.uniform(0.5,1)
        scale[np.argmax(dimensions)] = scale_max

    obj.set_scale(scale)

def randomize_viewing(robot):
    """
        Randomizes some of the robot's joints to get a different view of the object.
    """
    # arm_roll_joint 25
    # get random roation between -.8 and .8
    rotation = np.random.uniform(-0.8, 0.8)
    robot.set_rotation_euler_fk(link=robot.links[25], rotation_euler=rotation, mode='absolute')

    
    # head_pan_joint
    rotation = np.random.uniform(-.25 , 0)
    robot.set_rotation_euler_fk(link=robot.links[13], rotation_euler=rotation, mode='absolute')


    # head_tilt_joint
    rotation = np.random.uniform(-.3, 0)
    robot.set_rotation_euler_fk(link=robot.links[14], rotation_euler=rotation, mode='absolute')

    # arm_flex_joint
    rotation = np.random.uniform(-.5, 0)
    robot.set_rotation_euler_fk(link=robot.links[24], rotation_euler=rotation, mode='absolute')

def convert_to_convex_hull(obj):
    """
        Given an object, it calculates a convex hull and replaces the object by it.
    """
    obj.edit_mode()
    me = obj.get_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    me = bpy.data.meshes.new("%s convexhull" % me.name)
    ch = bmesh.ops.convex_hull(bm, input=bm.verts)
    bmesh.ops.delete(bm, geom=ch["geom_interior"], context='VERTS')
    obj.object_mode()
    obj.update_from_bmesh(bm)
    obj.enable_rigidbody(False)
    return True

def get_obj_filenames(shapenet_path, cc_textures, obj_type='shapenet', categories_file = None):
    """
    Load objects from ShapeNet or BOP dataset, apply random textures, and set physics properties.
    Parameters:
    shapenet_path (str): Path to the ShapeNet dataset.
    cc_textures (list): List of textures to apply to the objects.
    type (str): Type of dataset to load ('shapenet' or other). Default is 'shapenet'.
    num_objects (int): Number of objects to load. Default is 1.
    categories_file (str): Path to the categories file containing ShapeNet IDs. Default is None.
    Returns:
    tuple: A tuple containing:
        - objs (list): List of loaded objects.
        - cc_textures (list): List of textures applied to the objects.
        - objs_bm (list): List of loaded objects for convex hull generation.
    """
    if obj_type == 'shapenet':
        # open categories file, which has format ID NAME and get list of ID
        with open(categories_file, 'r') as f:
            categories = f.readlines()
            categories = [category.split()[0] for category in categories]

        # intersect categories with the ones in the filed']
        shapenet_categories = os.listdir(shapenet_path)
        categories = list(set(categories).intersection(shapenet_categories))

        # get relevant obj files
        obj_files = []
        taxonomy_file = shapenet_path+"/taxonomy.json"
        taxonomy_file = json.load(open(taxonomy_file))
        for category in categories:
            # id_path = shapenet_path+"/"+category
            parent_id = _ShapeNetLoader.find_parent_synset_id(data_path=shapenet_path, synset_id=category,json_data=taxonomy_file)
            id_path = os.path.join(shapenet_path, parent_id )
            obj_files.extend(glob.glob(os.path.join(id_path, "*", "models", "model_normalized_scaled.ply")))
        # sample num_objects objects

    return obj_files


def get_obj(object_file, cc_textures):
    """
    Load objects from ShapeNet or BOP dataset, apply random textures, and set physics properties.
    Parameters:
    shapenet_path (str): Path to the ShapeNet dataset.
    cc_textures (list): List of textures to apply to the objects.
    type (str): Type of dataset to load ('shapenet' or other). Default is 'shapenet'.
    num_objects (int): Number of objects to load. Default is 1.
    categories_file (str): Path to the categories file containing ShapeNet IDs. Default is None.
    Returns:
    tuple: A tuple containing:
        - objs (list): List of loaded objects.
        - cc_textures (list): List of textures applied to the objects.
        - objs_bm (list): List of loaded objects for convex hull generation.
    """
    obj = bproc.loader.load_obj(object_file)[0]
    obj_bm=bproc.loader.load_obj(object_file)[0]
     
    success = convert_to_convex_hull(obj_bm)
    if not success:
        return None, None
    obj.set_shading_mode('auto')
    obj.hide(True)

    obj_bm.hide(True)        


    random_cc_texture = np.random.choice(cc_textures)
    obj.replace_materials(random_cc_texture)
    if not obj.has_uv_mapping():
        obj.add_uv_mapping("smart")
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Alpha", 1.0)
    obj.enable_rigidbody(False)
    obj.hide(True)
  

    return obj,obj_bm
def sample_pose_func(obj: bproc.types.MeshObject):
    # place it randomly at [x,y,1] with x in [0.6,1.7] AND Y IN [-1,1]
    x = np.random.uniform(0.8, 1.2)
    y = np.random.uniform(-.5, .8)
    z = np.random.uniform(0.4, 0.6)
    obj.set_location([x, y, z])


def get_distractor_objects(obj_files, cc_textures, num_objects = 1):
    objs = []
    i = 10
    for obj_file in np.random.choice(obj_files, num_objects, replace=False):
        obj = bproc.loader.load_obj(obj_file)[0]
        if obj is None:
            continue
        # set random texture
        random_cc_texture = np.random.choice(cc_textures)
        obj.replace_materials(random_cc_texture)
        if not obj.has_uv_mapping():
            obj.add_uv_mapping("smart")
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Alpha", 1.0)
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        obj.set_cp('category_id',i)
        i+=1
        obj.hide(True)

        # get larges bounding box side
        bbox = obj.get_bound_box()
        bbox = np.array(bbox)
        max_corner = np.max(bbox, axis=0)
        min_corner = np.min(bbox, axis=0)

        dimensions = max_corner - min_corner

        max_dim_idx = np.argmax(dimensions)
        max_dim = dimensions[max_dim_idx]

        # get target size between 0.1 and 0.5 randomly
        target_size = np.random.uniform(0.1, 0.35)

        # scale the object to the target size
        scale = target_size / max_dim
        obj.set_scale([scale, scale, scale])

        # scale to 0.001
        obj.hide(False)


        objs.append(obj)
        

    return objs


def get_robot(urdf_file):
    """
    Loads a robot from a URDF file, modifies its materials, and returns the robot object.

    Args:
        urdf_file (str): The file path to the URDF file of the robot.

    Returns:
        bproc.loader.URDF: The loaded and modified robot object.

    The function performs the following steps:
    1. Loads the robot from the specified URDF file.
    2. Removes the link at index 0.
    3. Sets ascending category IDs for the robot's links.
    4. Iterates through the robot's links and modifies the materials of the visuals:
        - Sets the "Metallic" shader value to a random value between 0 and 0.1.
        - Sets the "Roughness" shader value to a random value between 0 and 0.5.
    5. Modifies the materials of specific links (indices 32, 34, 38, 40) to have a random black color.
    """
    print(urdf_file)
    try:
        robot = bproc.loader.load_urdf(urdf_file)
    except:
        for _ in range(5):
            # wait 10s and try again
            time.sleep(10)
            robot = bproc.loader.load_urdf(urdf_file)
    robot.remove_link_by_index(index=0)
    robot.set_ascending_category_ids()
    for link in robot.links:
        if link.visuals and hasattr(link.visuals[0], "materials"):
            materials = link.visuals[0].get_materials()
            for mat in materials:
                mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.1))
                # mat.set_principled_shader_value("Specular", np.random.uniform(0, 0.5))
                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))

    # Specify the indices of the links you want to modify
    indices_to_modify = [32, 34, 38, 40]  # Replace with the desired indices

    for index in indices_to_modify:
        link = robot.links[index]
        materials = link.visuals[0].get_materials()
        for mat in materials:
            black_col = np.random.uniform(0.001, 0.02)
            mat.set_principled_shader_value("Base Color", [black_col, black_col, black_col, 1])
            # mat.set_principled_shader_value("Specular", np.random.uniform(0, 0.3))

    return robot


def prepare_room():
    """
    Prepares a room environment with lighting for a simulation.

    This function creates a ceiling light plane, a point light, and the walls of a room using 
    the bproc library. The ceiling light plane is assigned a material, and the point light's 
    energy is set. The room is constructed using planes positioned and rotated to form walls.

    Returns:
        tuple: A tuple containing the following elements:
            - light_plane (bproc.object): The ceiling light plane object.
            - light_plane_material (bproc.material): The material assigned to the ceiling light plane.
            - light_point (bproc.types.Light): The point light object.
            - room_planes (list of bproc.object): A list of plane objects representing the walls of the room.
    """
    # sample light color and strength from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(50)

    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    i = 4
    for plane in room_planes:
        plane.set_cp('category_id',i)
        i+=1
    # Set up lighting
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3,6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    ) 
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=2,
        radius_max=3,
        elevation_min=5,
        elevation_max=89
    )
    light_point.set_location(location)
    return room_planes
def is_point_inside_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.

    Parameters:
    -----------
    point : array-like
        The point to check
    bbox : array-like
        The bounding box to check againstshapen

    Returns:
    --------
    bool
        True if the point is inside the bounding box, False otherwise
    """
    bbox = np.array(bbox)
    min_corner = np.min(bbox, axis=0)
    max_corner = np.max(bbox, axis=0)
    return np.all(min_corner <= point) and np.all(point <= max_corner)


def move_object_away_from_camera(obj, obj_cv, robot, step_move=0.05):
    """
    Moves the object obj (and its corresponding bounding box) away from the robot grippr.
    """
    # We check if the object is inside the robot's hand camera and move it away from it
    center_grasp_pos = obj.get_location()
    joint_positions = robot.get_all_local2world_mats()
    joint_obj = robot.get_all_visual_objs()[21]

    joint_middle_vec = center_grasp_pos-joint_obj.get_location()
    step_move = 0.05
    iters = 1
    link_bvh_tree = joint_obj.create_bvh_tree()

    while CollisionUtility.check_mesh_intersection(joint_obj,obj) and CollisionUtility.is_point_inside_object(point=joint_obj.get_location(),obj_bvh_tree=link_bvh_tree,obj=obj) and  iters<1/step_move*1.5:
        obj.set_location(obj.get_location() + joint_middle_vec*step_move)
        obj_cv.set_location(obj_cv.get_location() + joint_middle_vec*step_move)
        iters+=1

    print(f'Moved object {iters*step_move} away from camera')


def close_grippers(obj, obj_cv, robot):
    """
    Close the robot's grippers around the object until a collision is detected.

    Parameters:
    -----------
    obj : bproc.object
        The object to grasp.
    obj_cv : bproc.object
        The convex hull of the object.
    robot : bproc.loader.URDFLoader
        The loaded robot model.
    """
    has_collided = {32: False, 38: False}
    rotations = {32: 0, 38: 0}
    bvh_tree_cv = obj_cv.create_bvh_tree()
    patience = 10
    patience_iter = {32: 0, 38: 0}

    # Progressively move each finger to check for collisions with the object
    while rotations[32] < 1.1 * 0.8 / 0.005 and rotations[38] < 1.1 * 0.8 / 0.005 and not (has_collided[32] and has_collided[38]):
        joint_positions = robot.get_all_local2world_mats()
        finger_positions = {32: joint_positions[35], 38: joint_positions[41]}
        
        for finger_idx, finger_pos in finger_positions.items():
            if has_collided[finger_idx]:
                continue
            
            point = finger_pos[:3, 3]
            collision = CollisionUtility.is_point_inside_object(obj=obj_cv, obj_bvh_tree=bvh_tree_cv, point=point)
            
            if collision:
                patience_iter[finger_idx] += 1
                if patience_iter[finger_idx] > patience:
                    rotations[finger_idx] -= patience
                    has_collided[finger_idx] = True
                    continue
            else:
                patience_iter[finger_idx] = 0

            with open(os.devnull, 'w') as fnull:
                sys.stdout = fnull
                robot.set_rotation_euler_fk(link=robot.links[finger_idx], rotation_euler=-0.005, mode='relative')
                sys.stdout = sys.__stdout__
            
            rotations[finger_idx] += 1

    
    if rotations[32] < 50 and rotations[38] > 11:
        robot.set_rotation_euler_fk(link=robot.links[32], rotation_euler=-0.005 * rotations[38], mode='relative')
    if rotations[38] < 50 and rotations[32] > 11:
        robot.set_rotation_euler_fk(link=robot.links[38], rotation_euler=-0.005 * rotations[32], mode='relative')
    
    obj_cv.hide(True)

def setup_camera(robot):
    # We set up the camera                
    joint_positions = robot.get_all_local2world_mats()

    # Sample camera pose
    location = joint_positions[21][:3, 3]

    poi = bproc.object.compute_poi(robot.links[42].get_visuals())
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

def render_scene(robot,output_dir):
    
    print("\n\n\nRENDERING\n-----------------------------------------")
    # Render data
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id","instance", "name"])    
    data = bproc.renderer.render()

    bproc.writer.write_coco_annotations(
        Path(output_dir),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors= data["colors"],
        color_file_format= "JPEG",
    )

def full_close_grippers(robot):
    """
    Close the robot's grippers fully.
    """   
    robot.set_rotation_euler_fk(link=robot.links[31], rotation_euler=-0.08, mode='absolute')
    robot.set_rotation_euler_fk(link=robot.links[37], rotation_euler=-0.08, mode='absolute')

def generate_scene(
    urdf_file, 
    output_dir, 
    positive_ratio, 
    shapenet_path,
    cc_textures_path,
    dataset_type,
    num_scenes = 1,
    num_samples_per_scene = 1,
    resolution_X = 640,
    resolution_Y = 480,
    render=False
):
    """
    Generate a scene with a robot, objects, and rendering.

    Parameters:
    -----------
    urdf_file : str
        Path to the URDF file for the robot
    output_dir : str
        Directory to save output files
    positive_ratio: float
        Ratio of positive samples, between 0 and 1
    shapenet_path : str, optional
        Path to ShapeNet dataset
    cc_textures_path : str, optional
        Path to CC Textures
    dataset_type : str, optional
        Type of dataset to load objects from ('shapenet' or 'bop')
    num_scenes = 1 : int, optional
        Number of scenes to generate
    num_samples_per_scene = 1 : int, optional
        Number of samples to generate per scene
    resolution_X = 640 : int, optional
        X resolution of the rendered images
    resolution_Y = 480 : int, optional
        Y resolution of the rendered images
    render = False : bool, optional
    
    """
    # Initialize BlenderProc

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    # Load CC Textures
    cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
    print("CC Textures loaded")
    # Set rendering parameters
    bproc.camera.set_resolution(resolution_X, resolution_Y)
    bproc.renderer.enable_depth_output(True)

    # Prepare room
    room_planes = prepare_room()

    # We get the robot
    robot = get_robot(urdf_file)
    robot.set_cp('category_id',0)
    for link_id in [23,24,26,27]:
        robot.get_all_visual_objs()[link_id].set_cp('category_id',1)
        mat= robot.get_all_visual_objs()[link_id].get_materials()[0]
        mat.set_principled_shader_value("Specular IOR Level", 0.1)
    print("Robot loaded")

    # Generate scenes
    obj_files = get_obj_filenames(shapenet_path, cc_textures, obj_type=dataset_type, categories_file='custom_scripts/categories.txt')
    for scene in range(num_scenes):
        
        # We get the distractor objects
        distractor_objs = get_distractor_objects(obj_files, cc_textures, num_objects=np.random.randint(2,15))
        print("Distractor objects loaded")
        # We place them in the scene
        distractor_objs = bproc.object.sample_poses_on_surface(objects_to_sample=distractor_objs,
                                                          surface=room_planes[0],
                                                          sample_pose_func=sample_pose_func,
                                                          min_distance=0.5, 
                                                          max_distance=3)
        
        
        for i in range(num_samples_per_scene):
            # we set a random texture to the background
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)

            # Flex wrist for better view
            robot.set_rotation_euler_fk(link=robot.links[26], rotation_euler=-1.0, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[32], rotation_euler=0.8, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[38], rotation_euler=0.8, mode='absolute')

            randomize_viewing(robot)
            
            # generate True or False for the object to be included following the ratio
            include_object = np.random.choice([True, False], p=[positive_ratio, 1-positive_ratio])

            if include_object:
                robot.set_rotation_euler_fk(link=robot.links[31], rotation_euler=0., mode='absolute')
                robot.set_rotation_euler_fk(link=robot.links[37], rotation_euler=0., mode='absolute')
                # We get the object for the gripper
                obj = np.random.choice(obj_files)
                obj, obj_cv = get_obj(obj, cc_textures)
                obj.set_cp('category_id',2)
                obj.set_name('object')
                if obj is None:
                    continue

                position_and_align_object(obj,obj_cv, robot)


                move_object_away_from_camera(obj, obj_cv, robot)
                close_grippers(obj, obj_cv, robot)
            else:
                robot.set_rotation_euler_fk(link=robot.links[31], rotation_euler=-1.08, mode='absolute')
                robot.set_rotation_euler_fk(link=robot.links[37], rotation_euler=-1.08, mode='absolute')

            setup_camera(robot)
            if render:
                sub_path = 'positive' if include_object else 'negative'
                render_scene(robot, Path(output_dir) / sub_path)
            
            print("--"*10)
            print("\n"*10)
            if include_object:
                obj.hide(True)

            bproc.utility.reset_keyframes()



# Optionally, keep the main block for direct script execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate data for the BOP challenge')
    parser.add_argument('--render', action='store_true', help='Render the scene')
    parser.add_argument('--num_scenes', type=int, default=1, help='Number of scenes')
    parser.add_argument('--num_per_scene', type=int, default=1, help='Number of samples per scene')
    parser.add_argument('--positive_ratio', type=float, default=0.5, help='Ratio of positive samples')
    args = parser.parse_args()

    # add the path to load the file
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))    
    # load config.json
    
    with open('config.json') as f:
        config = json.load(f)
    print("---"*10)
    print("STARTING")
    print("---"*10)
    print("Args:"+str(args))
    print("Config:")
    print(config)
    print("---"*10)
    bproc.init()


    
    generate_scene(
        urdf_file=config['urdf_file'], 
        output_dir=config['output_dir'],    
        positive_ratio = args.positive_ratio,
        num_scenes=args.num_scenes,
        num_samples_per_scene=args.num_per_scene,
        shapenet_path=config['objects_path'],
        cc_textures_path=config['textures_path'],
        dataset_type=config['dataset_type'],
        render = args.render
    )
