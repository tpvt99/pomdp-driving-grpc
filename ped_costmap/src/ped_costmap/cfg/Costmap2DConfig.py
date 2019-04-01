## *********************************************************
##
## File autogenerated for the ped_costmap package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'upper': 'DEFAULT', 'lower': 'groups', 'srcline': 246, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'cstate': 'true', 'parentname': 'Default', 'class': 'DEFAULT', 'field': 'default', 'state': True, 'parentclass': '', 'groups': [], 'parameters': [{'srcline': 274, 'description': 'Specifies the delay in transform (tf) data that is tolerable in seconds.', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'transform_tolerance', 'edit_method': '', 'default': 0.3, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The frequency in Hz for the map to be updated.', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'update_frequency', 'edit_method': '', 'default': 5.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The frequency in Hz for the map to be publish display information.', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'publish_frequency', 'edit_method': '', 'default': 0.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The maximum height of any obstacle to be inserted into the costmap in meters.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'max_obstacle_height', 'edit_method': '', 'default': 2.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The default maximum distance from the robot at which an obstacle will be inserted into the cost map in meters.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'max_obstacle_range', 'edit_method': '', 'default': 2.5, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The default range in meters at which to raytrace out obstacles from the map using sensor data.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'raytrace_range', 'edit_method': '', 'default': 3.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'A scaling factor to apply to cost values during inflation.', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'cost_scaling_factor', 'edit_method': '', 'default': 10.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The radius in meters to which the map inflates obstacle cost values.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'inflation_radius', 'edit_method': '', 'default': 0.55, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The footprint of the robot specified in the robot_base_frame coordinate frame as a list in the format: [ [x1, y1], [x2, y2], ...., [xn, yn] ].', 'max': '', 'cconsttype': 'const char * const', 'ctype': 'std::string', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'footprint', 'edit_method': '', 'default': '[]', 'level': 0, 'min': '', 'type': 'str'}, {'srcline': 274, 'description': 'The radius of the robot in meters, this parameter should only be set for circular robots, all others should use the footprint parameter described above.', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'robot_radius', 'edit_method': '', 'default': 0.46, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'Whether or not to use the static map to initialize the costmap.', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'static_map', 'edit_method': '', 'default': True, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 274, 'description': 'Whether or not to use a rolling window version of the costmap.', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'rolling_window', 'edit_method': '', 'default': False, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 274, 'description': 'The value for which a cost should be considered unknown when reading in a map from the map server.', 'max': 255, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'unknown_cost_value', 'edit_method': '', 'default': 0, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The width of the map in meters.', 'max': 20, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'width', 'edit_method': '', 'default': 0, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The height of the map in meters.', 'max': 20, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'height', 'edit_method': '', 'default': 10, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The resolution of the map in meters/cell.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'resolution', 'edit_method': '', 'default': 0.05, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The x origin of the map in the global frame in meters.', 'max': inf, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'origin_x', 'edit_method': '', 'default': 0.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The y origin of the map in the global frame in meters.', 'max': inf, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'origin_y', 'edit_method': '', 'default': 0.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'Whether or not to publish the underlying voxel grid for visualization purposes.', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'publish_voxel_map', 'edit_method': '', 'default': False, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 274, 'description': 'The threshold value at which to consider a cost lethal when reading in a map from the map server.', 'max': 255, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'lethal_cost_threshold', 'edit_method': '', 'default': 100, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The topic that the costmap subscribes to for the static map.', 'max': '', 'cconsttype': 'const char * const', 'ctype': 'std::string', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'map_topic', 'edit_method': '', 'default': 'map', 'level': 0, 'min': '', 'type': 'str'}, {'srcline': 274, 'description': 'What map type to use. voxel or costmap are the supported types', 'max': '', 'cconsttype': 'const char * const', 'ctype': 'std::string', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'map_type', 'edit_method': "{'enum_description': 'An enum to set the map type', 'enum': [{'srcline': 13, 'description': 'Use VoxelCostmap2D', 'srcfile': '/home/panpan/workspace/catkin_ws/src/ped_costmap/cfg/Costmap2D.cfg', 'cconsttype': 'const char * const', 'value': 'voxel', 'ctype': 'std::string', 'type': 'str', 'name': 'voxel_const'}, {'srcline': 13, 'description': 'Use Costmap2D', 'srcfile': '/home/panpan/workspace/catkin_ws/src/ped_costmap/cfg/Costmap2D.cfg', 'cconsttype': 'const char * const', 'value': 'costmap', 'ctype': 'std::string', 'type': 'str', 'name': 'costmap_const'}]}", 'default': 'costmap', 'level': 0, 'min': '', 'type': 'str'}, {'srcline': 274, 'description': 'The z origin of the map in meters.', 'max': inf, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'origin_z', 'edit_method': '', 'default': 0.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The z resolution of the map in meters/cell.', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'z_resolution', 'edit_method': '', 'default': 0.2, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 274, 'description': 'The number of voxels to in each vertical column.', 'max': 16, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'z_voxels', 'edit_method': '', 'default': 10, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The number of unknown cells allowed in a column considered to be known', 'max': 16, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'unknown_threshold', 'edit_method': '', 'default': 15, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'The maximum number of marked cells allowed in a column considered to be free', 'max': 16, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'mark_threshold', 'edit_method': '', 'default': 0, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 274, 'description': 'Specifies whether or not to track what space in the costmap is unknown', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'track_unknown_space', 'edit_method': '', 'default': False, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 274, 'description': 'Restore to the original configuration', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator.py', 'name': 'restore_defaults', 'edit_method': '', 'default': False, 'level': 0, 'min': False, 'type': 'bool'}], 'type': '', 'id': 0}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

Costmap2D_voxel_const = 'voxel'
Costmap2D_costmap_const = 'costmap'
