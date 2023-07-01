#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles
"""
from distutils.util import strtobool
import math
import weakref
import inspect
import numpy as np
import cv2

import pandas as pd
import os
import datetime
import sys

import glob
# import os
# import sys
import signal

try:
	sys.path.append(glob.glob(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass


import carla
# from agents.navigation.basic_agent import LocalPlanner
from agents.navigation.basic_agent import LocalPlanner
from agents.navigation.roaming_agent import RoamingAgent
from examples.manual_control import IMUSensor

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
# from srunner.scenariomanager.actorcontrols.rss_sensor import RssSensor
from examples.rss.rss_sensor import RssSensor

# @staticmethod
def change_mkz_physics(physics_control):
	"""
	Change Lincoln MKZ physics
	"""
	
	## Print vehicle physics control
	physics_control.gear_switch_time = 0.05
	physics_control.torque_curve = [carla.Vector2D(x=500, y=67.7908), 
													carla.Vector2D(x=2000, y=508.4317), 
													carla.Vector2D(x=2750, y=542.3271), 
													carla.Vector2D(x=5500, y=517.9224), 
													carla.Vector2D(x=7500, y=284.8573)]

	physics_control.forward_gears = [carla.GearPhysicsControl(ratio=4.580, down_ratio=0.200000, up_ratio=0.350000),
													carla.GearPhysicsControl(ratio=2.960000, down_ratio=0.200000, up_ratio=0.350000),
													carla.GearPhysicsControl(ratio=1.910000, down_ratio=0.200000, up_ratio=0.350000),
													carla.GearPhysicsControl(ratio=1.450000, down_ratio=0.200000, up_ratio=0.350000), 
													carla.GearPhysicsControl(ratio=1.00000, down_ratio=0.200000, up_ratio=0.350000), 
													carla.GearPhysicsControl(ratio=0.75000, down_ratio=0.200000, up_ratio=0.350000)]
	physics_control.max_rpm = 6099.8
	physics_control.moi =  0.7344025

class ContolLossWithPvaC(BasicControl):

	"""
	Controller class for vehicles derived from BasicControl.

	The controller makes use of the LocalPlanner implemented in CARLA.

	Args:
		actor (carla.Actor): Vehicle actor that should be controlled.
	"""

	def __init__(self, actor, args=None):
		super(ContolLossWithPvaC, self).__init__(actor)

		self._world = CarlaDataProvider.get_world()
		self._actor = actor
		self._restrictor = None
		self._cv_image = None
		self._camera = None
		
		self._draw_waypoints = False
		self._draw_bbox = False
		self._log_info_on_camera = False
		self._restrictor_enabled = True

		# routing_target = [carla.Transform(carla.Location(x=350.0, y=1.98, z=0.05), carla.Rotation())] ## LVA_10
		routing_target = [carla.Transform(carla.Location(x=632.0, y=45.2), carla.Rotation())] ## LVD_Town6
		# routing_target = [carla.Transform(carla.Location(x=-12.5, y=157.0), carla.Rotation())] ## LVD_Town4
		# routing_target = None

		## Sensors
		##  parent_actor, world, unstructured_scene_visualizer, bounding_box_visualizer, state_visualizer, routing_targets=Non
		self._rss_sensor = RssSensor(actor, self._world, None, None, None, routing_targets=routing_target)
		print(self._rss_sensor.get_default_parameters())
		max_brake = self._rss_sensor.get_default_parameters().alphaLon.brakeMax

		self._imu_sensor = IMUSensor(self._actor)
		
		self._physics_control_static = self._actor.get_physics_control()
		self._actor.apply_physics_control(self._physics_control_static)
		#print(self._physics_control_static)
		
		if self._restrictor_enabled:
			print("RSS Restrictor Enabled")
			#print(self._physic_control_static)
			print("X")
			self._restrictor = carla.RssRestrictor()


		client = carla.Client('127.0.0.1', 2000)
		client.set_timeout(2.0)
		
		world = client.get_world()

		# Find Trigger Friction Blueprint
		friction_bp = world.get_blueprint_library().find('static.trigger.friction')

		extent = carla.Location(400.0, 100.0, 200.0)

		friction_bp.set_attribute('friction', str(0.2))
		friction_bp.set_attribute('extent_x', str(extent.x))
		friction_bp.set_attribute('extent_y', str(extent.y))
		friction_bp.set_attribute('extent_z', str(extent.z))

		# Spawn Trigger Friction
		transform1 = carla.Transform()
		transform2 = carla.Transform()

		# Critical
		# one
	
		transform1.location = carla.Location(668.2, 33.0, 0.0) 

		# two
		transform2.location = carla.Location(668.2, 40.0, 0.0) 


		world.spawn_actor(friction_bp, transform1)
		world.debug.draw_box(box=carla.BoundingBox(transform1.location, extent * 1e-2), rotation=transform1.rotation, life_time=100, thickness=0.5, color=carla.Color(r=255,g=0,b=0))
		world.spawn_actor(friction_bp, transform2)
		world.debug.draw_box(box=carla.BoundingBox(transform2.location, extent * 1e-2), rotation=transform2.rotation, life_time=100, thickness=0.5, color=carla.Color(r=255,g=0,b=0))
			
			
			
			

		## Behavior
		self._agent_behavior = RoamingAgent(self._actor )
		self._agent_behavior._proximity_vehicle_threshold = 1  # meters

		self._local_planner = self._agent_behavior._local_planner
		
		#self._physics_control_static = self._actor.get_physics_control()
		#self._actor.apply_physics_control(self._physics_control_static)		
		#print(self._physics_control_static)

		if self._waypoints:
			print("waypoints", self._waypoints)
			self._update_plan()

		## Camera
		if args and 'attach_camera' in args and strtobool(args['attach_camera']):
			bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
			self._camera = self._world.spawn_actor(bp, carla.Transform(
				carla.Location(x=-3.0, z=9.0), carla.Rotation(pitch=-40)), attach_to=self._actor)
			self._camera.listen(lambda image: self._on_camera_update(image))  # pylint: disable=unnecessary-lambda
		
		## Safety Metrics Log
		self._safety_metrics_log_fn = os.path.join(os.environ['SCENARIO_RUNNER_ROOT'],  
													'safety_metrics_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')

		self._safety_metrics_df = pd.DataFrame(columns=["timestamp",
														"situation_type",
														"ego_id",
														"ego_type_id",
														"ego_yaw",
														"ego_length",
														"ego_width",
														"ego_center_x",
														"ego_center_y",
														"ego_speed",
														"ego_transform_location_x",
														"ego_transform_location_y",
														"ego_transform_location_z",
														"ego_transform_rotation_pitch",
														"ego_transform_rotation_yaw",
														"ego_transform_rotation_roll",
														"ego_rf_x",
														"ego_rf_y",
														"ego_rf_z",
														"ego_rb_x",
														"ego_rb_y",
														"ego_rb_z",
														"ego_lf_x",
														"ego_lf_y",
														"ego_lf_z",
														"ego_lb_x",
														"ego_lb_y",
														"ego_lb_z",
														"ego_vel_x",
														"ego_vel_y",
														"ego_vel_z",
														"ego_ang_vel_x",
														"ego_ang_vel_y",
														"ego_ang_vel_z",
														"ego_accel_x",
														"ego_accel_y",
														"ego_accel_z",
														"ego_lon_speed",
														"ego_lat_speed",
														"ego_lon_accel",
														"ego_lat_accel",
														"imu_sensor_accelerometer_x",
														"imu_sensor_accelerometer_y",
														"imu_sensor_accelerometer_z",
														"imu_sensor_gyroscope_x",
														"imu_sensor_gyroscope_y",
														"imu_sensor_gyroscope_z",
														"other_id",
														"other_type_id",
														"other_yaw",
														"other_length",
														"other_width",
														"other_center_x",
														"other_center_y",
														"other_speed",
														"other_transform_location_x",
														"other_transform_location_y",
														"other_transform_location_z",
														"other_transform_rotation_pitch",
														"other_transform_rotation_yaw",
														"other_transform_rotation_roll",
														"other_rf_x",
														"other_rf_y",
														"other_rf_z",
														"other_rb_x",
														"other_rb_y",
														"other_rb_z",
														"other_lf_x",
														"other_lf_y",
														"other_lf_z",
														"other_lb_x",
														"other_lb_y",
														"other_lb_z",
														"other_vel_x",
														"other_vel_y",
														"other_vel_z",
														"other_ang_vel_x",
														"other_ang_vel_y",
														"other_ang_vel_z",
														"other_accel_x",
														"other_accel_y",
														"other_accel_z",
														"other_lon_speed",
														"other_lat_speed",
														"other_lon_accel",
														"other_lat_accel",
														"d_lon",
														"d_lat_l",
														"d_lat_r",
														"d_min_lon",
														"d_min_lat_l",
														"d_min_lat_r",
														"ego_rss_parameters_alphaLon_accelMax",
														"ego_rss_parameters_alphaLon_brakeMax",
														"ego_rss_parameters_alphaLon_brakeMin",
														"ego_rss_parameters_alphaLon_brakeMinCorrect",
														"ego_rss_parameters_alphaLat_accelMax",
														"ego_rss_parameters_alphaLat_brakeMin",
														"ego_rss_parameters_lateralFluctuationMargin",
														"ego_rss_parameters_responseTime", 
														"proximity_threshold", 
														"ego_road_id",
														"ego_lane_id", 
														"other_road_id",
														"other_lane_id", 
														'ego_speed_limit', 
														'other_speed_limit'])


		
	def _draw_boundingbox_on_server(self):
		"""
		debug draw vehicle bounding box 

		"""
		### draw bboxes
		for vehicle in self._world.get_actors().filter('vehicle.*'):
			transform = vehicle.get_transform()
			bounding_box = vehicle.bounding_box
			bounding_box.location += transform.location
			self._world.debug.draw_box(bounding_box, transform.rotation, life_time=0.001)

	def _draw_world_coord_system(self):
		"""
		debug draw world coordinate system  

		"""
		self._world.debug.draw_arrow(carla.Location(0,0,0), carla.Location(10, 0, 0),  
										thickness=0.5, 
										arrow_size=1,
										color=carla.Color(r=255, g=0, b=0), 
										life_time=0.02,
										persistent_lines=False)	
												
		self._world.debug.draw_arrow(carla.Location(0,0,0), carla.Location(0, 10, 0),  
										thickness=0.5, 
										arrow_size=1,
										color=carla.Color(r=0, g=255, b=0), 
										life_time=0.02,
										persistent_lines=False)   

		self._world.debug.draw_arrow(carla.Location(0,0,0), carla.Location(0, 0, 10),  
										thickness=0.5, 
										arrow_size=1,
										color=carla.Color(r=0, g=0, b=255), 
										life_time=0.02,
										persistent_lines=False)			

	def _draw_vehicle_coord_system(self, actor, size = 5):
		"""
		debug draw vehicle coordinate system of each actor 

		"""
		transform = actor.get_transform()
		location = transform.location
		rot_matrix = transform.get_matrix()

		rot_matrix_aux = np.array(rot_matrix) * size

		thickness = size/10
		arrow_size = size/10

		self._world.debug.draw_arrow(location, location + carla.Location(rot_matrix_aux[0,0], rot_matrix_aux[1,0], rot_matrix_aux[2, 0]), 
										thickness=thickness, 
										arrow_size= arrow_size,
										color=carla.Color(r=255, g=0, b=0), 
										life_time=0.001,
										persistent_lines=False)	   

		self._world.debug.draw_arrow(location, location - carla.Location(rot_matrix_aux[0,1], rot_matrix_aux[1,1], rot_matrix_aux[2, 1]), 
										thickness=thickness, 
										arrow_size= arrow_size,
										color=carla.Color(r=0, g=255, b=0), 
										life_time=0.001,
										persistent_lines=False)	   

		self._world.debug.draw_arrow(location, location + carla.Location(rot_matrix_aux[0,2], rot_matrix_aux[1,2], rot_matrix_aux[2, 2]), 
										thickness=thickness, 
										arrow_size= arrow_size,
										color=carla.Color(r=0, g=0, b=255), 
										life_time=0.001,
										persistent_lines=False)

	def _get_bbox_corners(self, debug_draw = False):
		"""
		get the bounding box corner coordinates of each vehicle

		"""

		vehicle_corners = {}
		if debug_draw:
			self._draw_world_coord_system()

		### draw bboxes
		for vehicle in self._world.get_actors().filter('vehicle.*'):
			
			if debug_draw:
				self._draw_vehicle_coord_system(vehicle, 2)
			
			transform			   = vehicle.get_transform()
			location				= transform.location
			extent				  = vehicle.bounding_box.extent

			vehicle_world_matrix = np.array(transform.get_matrix())

			### Corner points
			RF = np.array([extent.x, extent.y, extent.z, 1]) ## Right Front
			RB = np.array([-extent.x, extent.y, extent.z, 1]) ## Right Back
			LF = np.array([extent.x, -extent.y, extent.z, 1]) ## Left Front
			LB = np.array([-extent.x, -extent.y, extent.z, 1]) ## Left Back

			### Transform bbox extent from vehicle coord frame to world coord frame
			### Homogeneous matrix (4x4) * (4x1)
			'''
			xx, yx, zx, tx  
			xy, yy, zy, tz
			xz, yz, zz, tz
			0, 0, 0, 1

			'''

			### Rotation and Translation of corner points
			rf = np.matmul(vehicle_world_matrix, RF.T)
			rb = np.matmul(vehicle_world_matrix, RB.T)
			lf = np.matmul(vehicle_world_matrix, LF.T)
			lb = np.matmul(vehicle_world_matrix, LB.T)


			if debug_draw:
				self._world.debug.draw_string(carla.Location(rf[0], rf[1], rf[2]), 'RF', draw_shadow=False,
												color=carla.Color(r=255, g=0, b=0), life_time=0.001,
												persistent_lines=False)			
				
				self._world.debug.draw_string(carla.Location(rb[0], rb[1], rb[2]), 'RB', draw_shadow=False,
												color=carla.Color(r=255, g=255, b=0), life_time=0.001,
												persistent_lines=False)			
				
				self._world.debug.draw_string(carla.Location(lf[0], lf[1], lf[2]), 'LF', draw_shadow=False,
												color=carla.Color(r=0, g=255, b=0), life_time=0.001,
												persistent_lines=False)			
				
				self._world.debug.draw_string(carla.Location(lb[0], lb[1], lb[2]), 'LB', draw_shadow=False,
												color=carla.Color(r=0, g=125, b=255), life_time=0.001,
												persistent_lines=False)

				self._world.debug.draw_string(carla.Location(vehicle_world_matrix[0,3], vehicle_world_matrix[1,3], vehicle_world_matrix[2,3]), 'X', draw_shadow=False,
												color=carla.Color(r=0, g=0, b=0), life_time=0.001,
												persistent_lines=False)
			### save as [RF, RB, LF, LB]
			vehicle_corners[vehicle.id] = [[rf[0], rf[1], rf[2]], 
					[rb[0], rb[1], rb[2]], 
					[lf[0], lf[1], lf[2]], 
					[lb[0], lb[1], lb[2]]]

		return vehicle_corners


	def _draw_waypoints_on_server(self):

		### draw waypoints
		waypoints = self._local_planner._waypoints_queue
		for waypoint in waypoints:
			self._world.debug.draw_point(waypoint[0].transform.location, life_time=0.01)
			self._world.debug.draw_string(waypoint[0].transform.location, str('*'), draw_shadow=False, color=(255,0,0), life_time=1)

	def _update_server_cam_transform(self):
		"""
		server camera transform to follow the ego vehicle 

		"""
		if self._actor:
			spectator = self._world.get_spectator()

			ego_vehicle_transform = self._actor.get_transform()
			
			pos_x = ego_vehicle_transform.location.x
			pos_y = ego_vehicle_transform.location.y
			pos_z = ego_vehicle_transform.location.z + 40
			yaw = 0 ## town06
			# yaw = -90 ## town04
			pitch = 270

			cam_transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw, pitch=pitch))
			
			spectator.set_transform(cam_transform)
	
	def _print_log_info_on_camera(self):
		"""
		debug print info on opencv frame 

		"""
		if len(self._safety_metrics_df) < 1:
			return

		safety_metrics = self._safety_metrics_df.iloc[-1]

		if safety_metrics is not None:
			vertical_pos_offset = 50
			color_offset = 255
			
			ego_speed = safety_metrics.ego_lon_speed
			ego_accel = safety_metrics.ego_lon_accel

			other_speed = safety_metrics.other_lon_speed
			other_accel = safety_metrics.other_lon_accel

			d_lon = safety_metrics.d_lon

			d_lon_min = safety_metrics.d_min_lon

			# other_id = safety_metrics.other_id

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(self._cv_image,'ego lon speed: ' + "%.2f" % ego_speed, (20, vertical_pos_offset), font, 0.5,(color_offset,color_offset,color_offset), 2)
			cv2.putText(self._cv_image,'ego lon accel: ' + "%.2f" % ego_accel, (20, vertical_pos_offset + 20), font, 0.5,(color_offset,color_offset,color_offset), 2)
			# cv2.putText(self._cv_image,'other lon speed: ' + "%.2f" % other_speed, (20, vertical_pos_offset + 40), font, 0.5,(0,0,0), 2)
			# cv2.putText(self._cv_image,'other lon accel: ' + "%.2f" % other_accel, (20, vertical_pos_offset + 60), font, 0.5,(0,0,0), 2)
			cv2.putText(self._cv_image,'lon distance: ' + "%.2f" % d_lon, (20, vertical_pos_offset + 80), font, 0.5,(color_offset,color_offset,color_offset), 2)
			cv2.putText(self._cv_image,'RSS lon distance: ' + "%.2f" % d_lon_min, (20, vertical_pos_offset + 100), font, 0.5,(color_offset,color_offset,color_offset), 2)


	def _on_camera_update(self, image, opencv_image = False):
		"""
		Callback for the camera sensor

		Sets the OpenCV image (_cv_image). Requires conversion from BGRA to RGB.
		"""

		if not image:
			return
		
		if self._actor is None:
			return

		### draw bbox 
		# if self._draw_bbox:
		#	 # self._draw_boundingbox_on_server()
		#	 self._draw_bbox_corners()
		
		
		if self._draw_waypoints:
			self._draw_waypoints_on_server()

		### update server camera transform
		self._update_server_cam_transform()

		if self._log_info_on_camera:
			self._print_log_info_on_camera()


		if opencv_image:
			### OpenCV image		
			image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
			np_image = np.reshape(image_data, (image.height, image.width, 4))
			np_image = np_image[:, :, :3]
			np_image = np_image[:, :, ::-1]
			self._cv_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

	
	def _update_plan(self):
		"""
		Update the plan (waypoint list) of the LocalPlanner
		"""
		plan = []
		for transform in self._waypoints:
			waypoint = CarlaDataProvider.get_map().get_waypoint(
				transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
			plan.append((waypoint, RoadOption.LANEFOLLOW))
		self._local_planner.set_global_plan(plan)

	def reset(self):
		"""
		Reset the controller
		"""
		print("Saving metrics to file")
		self._safety_metrics_df.to_csv(self._safety_metrics_log_fn, index=False)

		# if self._actor and self._actor.is_alive:
		#	 self._rss_sensor.destroy()
		#	 self._imu_sensor.sensor.destroy()
		#	 if self._local_planner:
		#		 self._local_planner.reset_vehicle()
		#		 self._local_planner = None
		#	 self._camera = None
		#	 self._actor = None

		if self._camera:
			self._camera.destroy()
			self._camera = None
		if self._imu_sensor:
			self._imu_sensor.sensor.destroy()
			self._imu_sensor = None
		if self._rss_sensor:
			self._rss_sensor.destroy()
			self._rss_sensor = None
		if self._local_planner:
			self._local_planner.reset_vehicle()
			self._local_planner = None
		if self._actor and self._actor.is_alive:
			self._actor = None





	def run_step(self):
		"""
		Execute on tick of the controller's control loop

		If _waypoints are provided, the vehicle moves towards the next waypoint
		with the given _target_speed, until reaching the final waypoint. Upon reaching
		the final waypoint, _reached_goal is set to True.

		If _waypoints is empty, the vehicle moves in its current direction with
		the given _target_speed.

		If _init_speed is True, the control command is post-processed to ensure that
		the initial actor velocity is maintained independent of physics.
		"""

		if self._rss_sensor:
			self._get_safety_metrics_logs()

		if self._cv_image is not None:
			cv2.imshow("", self._cv_image)
			cv2.waitKey(1)

		self._reached_goal = False
		self._local_planner.set_speed(self._target_speed * 3.6)

		if self._waypoints_updated:
			self._waypoints_updated = False
			self._update_plan()

		# control = self._local_planner.run_step(debug=True)
		control = self._agent_behavior.run_step(debug=True)

		rss_control_flag = False
		#-------------------------------------
		if self._restrictor:
			rss_proper_response = self._rss_sensor.proper_response if self._rss_sensor and self._rss_sensor.response_valid else None
			if rss_proper_response:
				print(self._rss_sensor.proper_response)
				print("Enabling RSS response")
				vehicle_control_rss = self._restrictor.restrict_vehicle_control(control, 
																			rss_proper_response, 
																			self._rss_sensor.ego_dynamics_on_route, 
																			self._physics_control_static)

				if not (control == vehicle_control_rss):
					rss_control_flag = True
					print('RSS restrictor is ON: brake=%.3f, steer=%.3f' % (vehicle_control_rss.brake, vehicle_control_rss.steer))

				control = vehicle_control_rss
		else:
			print("RSS Restrictor not working")

		# --- Raining scenario

		# actor_list = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')

		# for actor in actor_list:
		# 	wheel_phys = actor.get_physics_control().wheels[0]

		# 	print(f"Tire Friction: {wheel_phys.tire_friction}")
		# #---
		# #-------------------------------------

		if self._init_speed and not rss_control_flag:
			current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)
			print("RssRoaming speed", current_speed)
			
			if abs(self._target_speed - current_speed) > 3:
				yaw = self._actor.get_transform().rotation.yaw * (math.pi / 180)
				vx = math.cos(yaw) * self._target_speed
				vy = math.sin(yaw) * self._target_speed
				self._actor.set_velocity(carla.Vector3D(vx, vy, 0))

		# current_accel = math.sqrt(self._actor.get_acceleration().x**2 + self._actor.get_acceleration().y**2)
		# print("speed", current_speed, "acceleration", current_accel)
		self._actor.apply_control(control)

		# Check if the actor reached the end of the plan
		# @TODO replace access to private _waypoints_queue with public getter
		if not self._local_planner._waypoints_queue:  # pylint: disable=protected-access
			self._reached_goal = True

	def _get_road_lane_id(self, vehicle, world_map):
		"""
		:param vehicle: vehicle 
		:return: a list given by (road_id, lane_id)
		"""
	   
		vehicle = vehicle.get_location()
		vehicle_waypoint = world_map.get_waypoint(vehicle)

		return [vehicle_waypoint.road_id, vehicle_waypoint.lane_id]


	def _get_rss_state_by_id(self, rss_state_snapshot, id):
		for rss_state in rss_state_snapshot.individualResponses:
			if rss_state.objectId == id:
				return rss_state
		print("actor not found")


	def _get_safety_metrics_logs(self):

		world_model = self._rss_sensor.world_model
		world_map = self._world.get_map()
		
		if world_model is None:
			print("world model not ready")
			return

		if self._actor is None:
			return

		if self._rss_sensor is None:
			return

		rss_state_snapshot = self._rss_sensor.rss_state_snapshot
		ego_dynamics_on_route = self._rss_sensor.ego_dynamics_on_route

		time_index = world_model.timeIndex

		ego_actor = self._rss_sensor._parent
		ego_rss_parameters = self._rss_sensor.get_default_parameters() 

		for scene in world_model.scenes:
			# print('scene')
			# print(scene)

			ego_vehicle = scene.egoVehicle
			other_vehicle = scene.object

			situation_type = str(scene.situationType)

			## RSS sensor info
			ego_id		  = int(ego_vehicle.objectId)
			ego_type_id	 = str(ego_vehicle.objectType)
			ego_yaw		 = float(ego_vehicle.state.yaw)
			ego_length	  = float(ego_vehicle.state.dimension.length)
			ego_width	   = float(ego_vehicle.state.dimension.width)
			ego_center_x	= float(ego_vehicle.state.centerPoint.x)
			ego_center_y	= float(ego_vehicle.state.centerPoint.y)
			ego_speed	   = float(ego_vehicle.state.speed)
			
			other_id		= int(other_vehicle.objectId)
			other_type_id   = str(other_vehicle.objectType)
			other_yaw	   = float(other_vehicle.state.yaw)
			other_length	= float(other_vehicle.state.dimension.length)
			other_width	 = float(other_vehicle.state.dimension.width)
			other_center_x  = float(other_vehicle.state.centerPoint.x)
			other_center_y  = float(other_vehicle.state.centerPoint.y)
			other_speed	 = float(other_vehicle.state.speed)
			
			rss_state = self._get_rss_state_by_id(rss_state_snapshot, other_id)
			# print('rss state')
			# print(rss_state)
			if rss_state is None:
				print("empty RssState")
				return

			d_lon		   = float(rss_state.longitudinalState.rssStateInformation.currentDistance)
			d_lat_l		 = float(rss_state.lateralStateLeft.rssStateInformation.currentDistance)
			d_lat_r		 = float(rss_state.lateralStateRight.rssStateInformation.currentDistance)

			d_min_lon	   = float(rss_state.longitudinalState.rssStateInformation.safeDistance)
			d_min_lat_l	 = float(rss_state.lateralStateLeft.rssStateInformation.safeDistance)
			d_min_lat_r	 = float(rss_state.lateralStateRight.rssStateInformation.safeDistance)

			# print("d_lon: {}".format(d_lon))

			## CARLA info
			ego_transform   = ego_actor.get_transform()
			ego_accel	   = ego_actor.get_acceleration()
			ego_vel		 = ego_actor.get_velocity()
			ego_ang_vel	 = ego_actor.get_angular_velocity()
			ego_bbox_ext	= ego_actor.bounding_box.extent
			ego_speed_limit = ego_actor.get_speed_limit() /3.6

			ego_lon_speed   = self._get_longitudinal_speed(ego_actor)
			ego_lat_speed   = self._get_lateral_speed(ego_actor)
			ego_lon_accel   = self._get_longitudinal_acceleration(ego_actor)
			ego_lat_accel   = self._get_lateral_acceleration(ego_actor)

			other_actor	 = CarlaDataProvider.get_actor_by_id(other_id)
			if other_actor is not None:
				other_transform = other_actor.get_transform()
				other_accel	 = other_actor.get_acceleration()
				other_vel	   = other_actor.get_velocity()
				other_ang_vel   = other_actor.get_angular_velocity()
				other_speed_limit = other_actor.get_speed_limit() /3.6

				other_lon_speed   = self._get_longitudinal_speed(other_actor)
				other_lat_speed   = self._get_lateral_speed(other_actor)
				other_lon_accel   = self._get_longitudinal_acceleration(other_actor)
				other_lat_accel   = self._get_lateral_acceleration(other_actor)
			else: 
				print("can't get other actor's info")

			vehicle_corners = self._get_bbox_corners(debug_draw=False)
			# [RF, RB, LF, LB]
			ego_corners = vehicle_corners[ego_id]
			ego_rf = ego_corners[0]
			ego_rb = ego_corners[1]
			ego_lf = ego_corners[2]
			ego_lb = ego_corners[3]

			other_corners = vehicle_corners[other_id]
			other_rf = other_corners[0]
			other_rb = other_corners[1]
			other_lf = other_corners[2]
			other_lb = other_corners[3]

			map_desc = self._get_road_lane_id(ego_actor, world_map)

			ego_road_id = map_desc[0]
			ego_lane_id = map_desc[1]
			
			map_desc = self._get_road_lane_id(other_actor, world_map)
			other_road_id = map_desc[0]
			other_lane_id = map_desc[1]

			to_append = [self._rss_sensor.timestamp,
						situation_type,
						ego_id,
						ego_type_id,
						ego_yaw,
						ego_length,
						ego_width,
						ego_center_x,
						ego_center_y,
						ego_speed,
						ego_transform.location.x,
						ego_transform.location.y,
						ego_transform.location.z,
						ego_transform.rotation.pitch,
						ego_transform.rotation.yaw,
						ego_transform.rotation.roll,
						ego_rf[0],
						ego_rf[1],
						ego_rf[2],
						ego_rb[0],
						ego_rb[1],
						ego_rb[2],
						ego_lf[0],
						ego_lf[1],
						ego_lf[2],
						ego_lb[0],
						ego_lb[1],
						ego_lb[2],
						ego_vel.x,
						ego_vel.y,
						ego_vel.z,
						ego_ang_vel.x,
						ego_ang_vel.y,
						ego_ang_vel.z,
						ego_accel.x,
						ego_accel.y,
						ego_accel.z,
						ego_lon_speed,
						ego_lat_speed,
						ego_lon_accel,
						ego_lat_accel,
						self._imu_sensor.accelerometer[0],
						self._imu_sensor.accelerometer[1],
						self._imu_sensor.accelerometer[2],
						self._imu_sensor.gyroscope[0],
						self._imu_sensor.gyroscope[1],
						self._imu_sensor.gyroscope[2],
						other_id,
						other_type_id,
						other_yaw,
						other_length,
						other_width,
						other_center_x,
						other_center_y,
						other_speed,
						other_transform.location.x,
						other_transform.location.y,
						other_transform.location.z,
						other_transform.rotation.pitch,
						other_transform.rotation.yaw,
						other_transform.rotation.roll,
						other_rf[0],
						other_rf[1],
						other_rf[2],
						other_rb[0],
						other_rb[1],
						other_rb[2],
						other_lf[0],
						other_lf[1],
						other_lf[2],
						other_lb[0],
						other_lb[1],
						other_lb[2],
						other_vel.x,
						other_vel.y,
						other_vel.z,
						other_ang_vel.x,
						other_ang_vel.y,
						other_ang_vel.z,
						other_accel.x,
						other_accel.y,
						other_accel.z,
						other_lon_speed,
						other_lat_speed,
						other_lon_accel,
						other_lat_accel,
						d_lon,
						d_lat_l,
						d_lat_r,
						d_min_lon,
						d_min_lat_l,
						d_min_lat_r,
						ego_rss_parameters.alphaLon.accelMax,
						ego_rss_parameters.alphaLon.brakeMax,
						ego_rss_parameters.alphaLon.brakeMin,
						ego_rss_parameters.alphaLon.brakeMinCorrect,
						ego_rss_parameters.alphaLat.accelMax,
						ego_rss_parameters.alphaLat.brakeMin,
						ego_rss_parameters.lateralFluctuationMargin,
						ego_rss_parameters.responseTime, 
						self._agent_behavior._proximity_vehicle_threshold, 
						ego_road_id,
						ego_lane_id, 
						other_road_id,
						other_lane_id, 
						ego_speed_limit, 
						other_speed_limit]

			append_series = pd.Series(to_append, index = self._safety_metrics_df.columns)
			self._safety_metrics_df = self._safety_metrics_df.append(append_series, ignore_index=True)
		
	def _get_longitudinal_speed(self, vehicle):
		velocity	= vehicle.get_velocity()
		transform   = vehicle.get_transform()
		vel_np	  = np.array([velocity.x, velocity.y, velocity.z])
		pitch	   = np.deg2rad(-transform.rotation.pitch)
		yaw		 = np.deg2rad(transform.rotation.yaw)
		orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
		speed	   = np.dot(vel_np, orientation)
		
		return speed

	def _get_lateral_speed(self, vehicle):
		""" Convert the vehicle transform directly to forward and lateral speed
			Reference: https://i0.wp.com/slideplayer.com/4241728/14/images/34/Roll+Pitch+Yaw+The+rotation+matrix+for+the+following+operations%3A+X+Y+Z.jpg
		"""
		velocity	= vehicle.get_velocity()
		transform   = vehicle.get_transform()
		vel_np	  = np.array([velocity.x, velocity.y, velocity.z])
		roll		= np.deg2rad(-transform.rotation.roll)
		pitch	   = np.deg2rad(-transform.rotation.pitch)
		yaw		 = np.deg2rad(transform.rotation.yaw)
		orientationY  = np.array([-np.sin(yaw)*np.sin(roll) + np.cos(yaw)*np.sin(pitch)*np.sin(roll),
								  np.cos(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.sin(roll),
								  np.cos(pitch)*np.sin(roll)])
		lateral_speed   = np.dot(vel_np, orientationY)
		return lateral_speed

	def _get_longitudinal_acceleration(self, vehicle):
		velocity	= vehicle.get_acceleration()
		transform   = vehicle.get_transform()
		vel_np	  = np.array([velocity.x, velocity.y, velocity.z])
		pitch	   = np.deg2rad(-transform.rotation.pitch)
		yaw		 = np.deg2rad(transform.rotation.yaw)
		orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
		speed	   = np.dot(vel_np, orientation)
		
		return speed

	def _get_lateral_acceleration(self, vehicle):
		""" Convert the vehicle transform directly to forward and lateral speed
			Reference: https://i0.wp.com/slideplayer.com/4241728/14/images/34/Roll+Pitch+Yaw+The+rotation+matrix+for+the+following+operations%3A+X+Y+Z.jpg
		"""
		velocity	= vehicle.get_acceleration()
		transform   = vehicle.get_transform()
		vel_np	  = np.array([velocity.x, velocity.y, velocity.z])
		roll		= np.deg2rad(-transform.rotation.roll)
		pitch	   = np.deg2rad(-transform.rotation.pitch)
		yaw		 = np.deg2rad(transform.rotation.yaw)
		orientationY  = np.array([-np.sin(yaw)*np.sin(roll) + np.cos(yaw)*np.sin(pitch)*np.sin(roll),
								  np.cos(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.sin(roll),
								  np.cos(pitch)*np.sin(roll)])
		lateral_speed   = np.dot(vel_np, orientationY)
		return lateral_speed

