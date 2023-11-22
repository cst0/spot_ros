# StowState
uint8 STOWSTATE_UNKNOWN=0
uint8 STOWSTATE_STOWED=1
uint8 STOWSTATE_DEPLOYED=2

# CarryState
uint8 CARRY_STATE_UNKNOWN=0
uint8 CARRY_STATE_NOT_CARRIABLE=1
uint8 CARRY_STATE_CARRIABLE=2
uint8 CARRY_STATE_CARRIABLE_AND_STOWABLE=3

float64 gripper_open_percentage  # How open the gripper is, measured in percent. 0 = fully closed, 100 = fully open.
bool    is_gripper_holding_item	 # Will be true if the gripper is holding an item, false otherwise.
geometry_msgs/Vector3 estimated_end_effector_force_in_hand  # The estimated force on the end-effector expressed in the hand frame.
uint8 stow_state  # Information on if the arm is stowed, or deployed.
geometry_msgs/Twist velocity_of_hand_in_vision  # Velocity of the hand frame with respect to vision frame and expressed in vision frame. The linear velocity is applied at the origin of the hand frame.
geometry_msgs/Twist velocity_of_hand_in_odom  # ocity	Velocity of the hand frame with respect to odom frame and expressed in odom frame. Again, the linear velocity is applied at the origin of the hand frame.
uint8 carry_state # ManipulatorState.CarryState	

