# Robocasa Multiple Choice Problems

## robocasa - Problem 1 (Multiple Choice)
When designing the physics simulation for kitchen manipulation tasks, RoboCasa must balance multiple competing objectives. Which design decisions in the codebase reflect trade-offs between simulation fidelity and computational efficiency?

- (A) Using lite_physics=True in the Kitchen environment initialization to reduce computational overhead while maintaining sufficient contact dynamics for manipulation
- (B) Setting control_freq=20Hz as a compromise between responsive control and simulation stability, avoiding the need for smaller integration timesteps
- (C) Implementing a settling phase after object placement where the simulation steps through multiple iterations without robot actions to stabilize physics
- (D) Using single collision geometries for most objects rather than decomposed convex hulls to reduce collision detection complexity
- (E) Disabling self-collision checking for robot links to prevent numerical instabilities in the constraint solver

**Correct Answer: A, B, C**

**Explanation:** This tests understanding of physics simulation trade-offs across kitchen.py, placement_samplers.py, and object models. (A) is correct - lite_physics reduces solver iterations. (B) is correct - 20Hz balances control responsiveness with stable integration. (C) is correct - the settling phase in _reset_internal() prevents objects from falling through surfaces due to initial penetration. (D) is incorrect - the codebase uses detailed collision meshes from Objaverse/AI-generated models. (E) is incorrect - self-collision is typically enabled for articulated robots to prevent unrealistic configurations.

---

## robocasa - Problem 2 (Multiple Choice)
RoboCasa's procedural scene generation system must handle complex spatial constraints when placing objects and fixtures. Which architectural patterns enable the framework to generate valid, collision-free scenes while maintaining task-relevant spatial relationships?

- (A) The SequentialCompositeSampler chains multiple placement initializers, allowing objects to be placed relative to previously placed objects or fixtures
- (B) The UniformRandomSampler's obj_in_region() check ensures object bounding boxes remain fully within designated regions by testing against region boundary points
- (C) The placement system uses a two-phase approach: first placing fixtures with fxtr_placements, then placing objects with object_placements that can reference fixture positions
- (D) The objs_intersect() collision detection uses axis-aligned bounding box (AABB) tests for computational efficiency
- (E) Placement samplers support hierarchical containment where objects can be placed inside container objects, which are then added to the scene

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of the procedural generation architecture across placement_samplers.py, kitchen.py, and object_utils.py. (A) is correct - SequentialCompositeSampler enables chaining with placed_objects tracking. (B) is correct - obj_in_region() tests if all object boundary points lie within the region defined by p0, px, py. (C) is correct - _load_model() first samples fxtr_placements, then object_placements can reference them. (E) is correct - _create_objects() implements try_to_place_in logic for containers. (D) is incorrect - objs_intersect() uses oriented bounding box (OBB) tests with rotation, not simple AABB.

---

## robocasa - Problem 3 (Single Choice)
The framework supports multiple robot embodiments (mobile manipulators, humanoids) with different kinematic structures. How does the composite controller architecture in RoboCasa handle the coordination of heterogeneous control modalities (arm, gripper, base, torso)?

- (A) Each body part has an independent controller that publishes commands to a shared action buffer, which are then synchronized by a central arbitrator
- (B) The refactor_composite_controller_config() function restructures controller configurations to match RoboSuite v1.5's composite controller format, with body_part_ordering specifying the control sequence
- (C) A hierarchical state machine coordinates transitions between different control modes (navigation, manipulation, whole-body) based on task phase
- (D) All body parts share a single unified controller that computes joint commands using whole-body inverse kinematics

**Correct Answer: B**

**Explanation:** This tests understanding of the robot control architecture across kitchen.py and config_utils.py. (B) is correct - refactor_composite_controller_config() converts old controller configs to the new composite format, and for PandaOmron specifically sets body_part_ordering=['right', 'right_gripper', 'base', 'torso'] to define control structure. This enables RoboSuite's composite controller to properly coordinate different body parts. (A) is incorrect - there's no action buffer arbitration pattern. (C) is incorrect - no hierarchical state machine for mode switching exists. (D) is incorrect - body parts have separate controllers, not unified whole-body IK.

---

## robocasa - Problem 4 (Multiple Choice)
Computer vision systems for robot learning require careful camera placement and configuration. Which design principles guide RoboCasa's camera system architecture to support both policy learning and human interpretability?

- (A) The _cam_configs system in CamUtils provides robot-specific camera configurations that account for different robot morphologies and workspace geometries
- (B) Camera randomization applies Gaussian noise to agentview camera poses but not eye-in-hand cameras to maintain consistent end-effector observations
- (C) The set_cameras() method dynamically adjusts camera positions based on the kitchen layout_id to ensure relevant fixtures remain in view
- (D) Multiple camera viewpoints (agentview_center, agentview_left, agentview_right) provide overlapping coverage for depth estimation and occlusion handling
- (E) The translucent_robot rendering mode reduces visual occlusions by making the robot semi-transparent, improving visibility of manipulation targets

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of the vision system design across camera_utils.py, kitchen.py, and visualization code. (A) is correct - get_robot_cam_configs() returns robot-specific configurations. (B) is correct - _randomize_cameras() applies noise to agentview but sets pos_noise=np.zeros_like() for eye_in_hand. (C) is correct - set_cameras() uses LAYOUT_CAMS.get(layout_id) for layout-specific positioning. (E) is correct - visualize() sets alpha=0.10 for robot geoms when translucent_robot=True. (D) is incorrect - while multiple viewpoints exist, they're not specifically designed for stereo depth estimation; they provide different perspectives for policy learning.

---

## robocasa - Problem 5 (Single Choice)
The texture_swap system enables visual domain randomization by procedurally varying material appearances. What is the architectural relationship between texture randomization and the generative_textures parameter in scene generation?

- (A) When generative_textures='100p', all textures are replaced with AI-generated alternatives; when None/False, human-designed textures from the asset library are used
- (B) The texture swap system operates independently of generative_textures, which only controls object geometry generation
- (C) Texture randomization is applied during every reset() call to maximize visual diversity, while generative_textures is a one-time scene initialization parameter
- (D) The system uses a hybrid approach where generative_textures controls fixture textures while object textures are always randomized

**Correct Answer: A**

**Explanation:** This tests understanding of the texture system architecture across texture_swap.py and kitchen.py. (A) is correct - the _curr_gen_fixtures variable stores generative texture info from _ep_meta, and edit_model_xml() uses this to determine whether to apply AI-generated textures (when '100p') or use standard textures. The replace_*_texture() functions in texture_swap.py handle the actual swapping. (B) is incorrect - generative_textures affects visual appearance, not geometry. (C) is incorrect - texture selection happens during _load_model(), not every reset(). (D) is incorrect - the parameter affects both fixtures and objects uniformly.

---

## robocasa - Problem 6 (Multiple Choice)
The object sampling system must handle diverse physical properties and task requirements. Which mechanisms enable the framework to select task-appropriate objects while maintaining physical realism?

- (A) The OBJ_CATEGORIES dictionary encodes physical affordances (graspable, washable, microwavable, cookable, freezable) that constrain object selection based on task requirements
- (B) Object scaling factors (aigen vs objaverse) compensate for different asset sources having inconsistent real-world dimensions
- (C) The obj_instance_split parameter enables train/test dataset splitting by partitioning object instances, with 'A' using most instances and 'B' using the remainder
- (D) The exclude list in object categories removes specific instances with mesh defects (holes, inconsistent normals) that cause physics instabilities
- (E) Object groups (fruit, vegetable, utensil, etc.) are hierarchically organized to support semantic task specifications

**Correct Answer: A, B, C, D, E**

**Explanation:** This tests understanding of the object system design across kitchen_objects.py, kitchen.py, and sample_object(). (A) is correct - OBJ_CATEGORIES defines affordances used in sample_object() filtering. (B) is correct - different scale values for aigen vs objaverse compensate for asset inconsistencies. (C) is correct - obj_instance_split implements train/test splitting as documented. (D) is correct - exclude lists remove problematic meshes (e.g., 'bagel_8', 'bar_1' with holes). (E) is correct - types tuples create hierarchical groupings (e.g., apple has types=('fruit',)).

---

## robocasa - Problem 7 (Single Choice)
Scene generation must handle the challenge of creating diverse kitchen layouts while ensuring robot reachability. How does the layout and style system balance environmental diversity with task feasibility?

- (A) The scene_registry maintains a fixed set of 120 pre-designed kitchen scenes that are loaded deterministically based on layout_id and style_id combinations
- (B) Layouts are procedurally generated using grammar-based rules, while styles are selected from a fixed palette of material/color schemes
- (C) The EXCLUDE_LAYOUTS mechanism allows tasks to filter out layouts where spatial constraints make the task infeasible, such as fixtures being too far apart
- (D) All layouts are guaranteed to be reachable by all robot types through workspace analysis during scene generation

**Correct Answer: C**

**Explanation:** This tests understanding of scene generation constraints across scene_registry.py, kitchen.py, and task implementations. (C) is correct - tasks can set EXCLUDE_LAYOUTS (e.g., PnPCounterToMicrowave excludes layout 8 because the microwave is far from counters), and layout_and_style_ids are filtered to remove excluded layouts. This balances diversity (many layouts) with feasibility (task-specific filtering). (A) is incorrect - while there are 120 scenes, they're combinations of layouts and styles, not fully pre-designed. (B) is incorrect - layouts are pre-designed, not procedurally generated. (D) is incorrect - reachability is not guaranteed; tasks must handle this via exclusions.

---

## robocasa - Problem 8 (Multiple Choice)
The framework must bridge the gap between high-level task specifications and low-level control. Which architectural components facilitate this hierarchical task decomposition?

- (A) The distinction between single_stage (atomic) and multi_stage (composite) tasks reflects a two-level hierarchy where composite tasks sequence multiple atomic skills
- (B) The _get_obj_cfgs() method in task classes specifies object configurations that define the initial scene state required for task execution
- (C) The staged() decorator system enables tasks to define sequential subtasks with automatic state transitions and reward shaping
- (D) Task classes inherit from Kitchen which provides fixture and object management, while task-specific logic is implemented in _check_success() and reward computation
- (E) The ManipulationTask model from RoboSuite provides the underlying MJCF composition that merges arena, robots, and objects

**Correct Answer: A, B, D, E**

**Explanation:** This tests understanding of task architecture across task implementations, kitchen.py, and RoboSuite integration. (A) is correct - the directory structure and task organization explicitly separate atomic (single_stage/) from composite (multi_stage/) tasks. (B) is correct - _get_obj_cfgs() defines object requirements for tasks. (D) is correct - Kitchen provides infrastructure while tasks implement success conditions. (E) is correct - ManipulationTask in _load_model() handles MJCF merging. (C) is incorrect - there's no staged() decorator system; multi-stage tasks are implemented as monolithic classes, not with automatic subtask transitions.

---

## robocasa - Problem 9 (Single Choice)
Contact-rich manipulation tasks like opening drawers and doors require careful modeling of friction and contact dynamics. What design choice in RoboCasa's physics configuration most directly impacts the stability and realism of these interactions?

- (A) Using MuJoCo's implicit contact model with complementarity constraints to handle stiff contacts without requiring extremely small timesteps
- (B) Implementing custom contact callbacks that detect drawer/door contacts and apply corrective forces to prevent penetration
- (C) Setting lite_physics=True which reduces solver iterations but maintains sufficient contact resolution for manipulation tasks
- (D) Using a hybrid physics approach where contacts are detected in MuJoCo but resolved using a separate constraint-based solver

**Correct Answer: A**

**Explanation:** This tests understanding of contact dynamics and physics solver configuration across kitchen.py and MuJoCo integration. (A) is correct - MuJoCo's implicit contact model (the default) uses complementarity constraints to handle stiff contacts efficiently, which is crucial for drawer/door manipulation. The framework relies on this by using mujoco==3.2.6 with its default contact settings. (C) is partially relevant but not the most direct factor - lite_physics affects overall simulation speed but doesn't fundamentally change contact handling. (B) is incorrect - no custom contact callbacks exist. (D) is incorrect - MuJoCo handles all contact resolution internally.

---

## robocasa - Problem 10 (Multiple Choice)
Demonstration collection and dataset generation require careful coordination between simulation, control, and data recording. Which components work together to enable high-quality trajectory collection?

- (A) The collect_demos.py script integrates with teleoperation devices (keyboard, spacemouse) to record human demonstrations with synchronized state and observation data
- (B) The dataset_registry.py maintains mappings between task names and dataset paths, supporting multiple dataset types (human_raw, human_im, mg_im)
- (C) Episode metadata (get_ep_meta()) captures scene configuration (layout_id, style_id, object_cfgs) to enable deterministic replay
- (D) The h5py format stores trajectories with hierarchical organization: states, actions, observations, and rewards at each timestep
- (E) MimicGen integration enables automatic data augmentation by generating synthetic demonstrations from human seeds

**Correct Answer: A, B, C, D, E**

**Explanation:** This tests understanding of the data collection pipeline across collect_demos.py, dataset_registry.py, kitchen.py, and MimicGen integration. (A) is correct - collect_demos.py handles teleoperation and recording. (B) is correct - dataset_registry.py maps tasks to dataset paths and types. (C) is correct - get_ep_meta() stores scene config for replay. (D) is correct - h5py is used for trajectory storage (referenced in scripts). (E) is correct - the codebase imports mimicgen and supports mg_im datasets.

---

## robocasa - Problem 11 (Multiple Choice)
Articulated object manipulation (cabinets, drawers, microwaves) presents unique challenges for both physics simulation and control. Which aspects of the framework's design specifically address these challenges?

- (A) Fixture classes (SingleCabinet, Drawer, Microwave) encapsulate joint definitions and kinematic constraints for articulated objects
- (B) The get_fixture() method with FixtureType enum provides type-safe access to specific fixture instances in the scene
- (C) Joint position initialization in _reset_internal() sets articulated objects to specific configurations (open/closed) based on task requirements
- (D) Impedance control is used for all articulated object interactions to handle contact forces safely
- (E) The fixture placement system ensures articulated objects are positioned to avoid kinematic singularities during operation

**Correct Answer: A, B, C**

**Explanation:** This tests understanding of articulated object handling across fixture implementations, kitchen.py, and task code. (A) is correct - fixture classes in models/fixtures/ define joints and constraints. (B) is correct - get_fixture() with FixtureType provides structured access. (C) is correct - _reset_internal() in tasks sets joint positions via set_joint_qpos(). (D) is incorrect - the framework uses position/velocity control, not specifically impedance control for fixtures. (E) is incorrect - placement focuses on collision avoidance and reachability, not kinematic singularities.

---

## robocasa - Problem 12 (Multiple Choice)
Collision detection and avoidance are critical for safe manipulation in cluttered kitchen environments. Which mechanisms contribute to collision handling in the simulation?

- (A) The objs_intersect() function uses oriented bounding box (OBB) intersection tests to detect collisions between objects during placement
- (B) MuJoCo's built-in collision detection handles runtime collisions between robot, objects, and fixtures using the defined collision geometries
- (C) The ensure_valid_placement flag in samplers enables/disables collision checking during object placement to balance scene validity with generation speed
- (D) A motion planning module computes collision-free trajectories for all robot movements
- (E) The placement system uses iterative sampling with collision checking, retrying up to 5000 times to find valid object positions

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of collision handling across placement_samplers.py, object_utils.py, and physics simulation. (A) is correct - objs_intersect() in object_utils.py performs OBB tests. (B) is correct - MuJoCo handles runtime collision detection using collision meshes. (C) is correct - ensure_valid_placement in UniformRandomSampler controls whether collision checking is performed. (E) is correct - the sample() method in UniformRandomSampler uses a loop with range(5000) for retries. (D) is incorrect - no motion planning module exists; the framework relies on learned policies or teleoperation.

---

## robocasa - Problem 13 (Single Choice)
The framework must handle the challenge of generating realistic object interactions (grasping, placing, stacking). What physics modeling choice most directly enables stable grasping of diverse object geometries?

- (A) Using soft contact models with compliance parameters tuned for each object material type
- (B) Implementing suction-based grasping that doesn't require precise contact modeling
- (C) Relying on MuJoCo's friction pyramid model with appropriate friction coefficients defined in object MJCF files to generate realistic contact forces
- (D) Using constraint-based grasping where objects are rigidly attached to the gripper upon contact

**Correct Answer: C**

**Explanation:** This tests understanding of grasping physics across object models and MuJoCo integration. (C) is correct - MuJoCo uses friction pyramid models for contacts, and object MJCF files (from Objaverse/AI-generated assets) define friction coefficients. The framework relies on this standard contact model rather than implementing custom grasping physics. Gripper models from RoboSuite provide the gripper geometry, and MuJoCo's contact solver handles the grasping dynamics. (A) is incorrect - while MuJoCo supports soft contacts, the framework doesn't tune per-material compliance. (B) is incorrect - parallel-jaw grippers are used, not suction. (D) is incorrect - no rigid attachment mechanism exists; grasping emerges from contact forces.

---

## robocasa - Problem 14 (Multiple Choice)
The framework's scene representation must support both simulation and policy learning. Which design decisions enable efficient scene representation while maintaining the information needed for learning?

- (A) The MJCF (MuJoCo XML) format provides a hierarchical scene description that separates kinematic structure, visual appearance, and collision geometry
- (B) The edit_model_xml() method post-processes generated MJCF to apply texture swaps, fix naming conventions, and adjust rendering properties
- (C) Scene state is represented as a flat vector of joint positions and velocities for efficient neural network processing
- (D) The KitchenArena class encapsulates scene construction, providing methods to query fixture positions and spatial relationships
- (E) Object and fixture models are loaded lazily during simulation to reduce memory footprint

**Correct Answer: A, B, D**

**Explanation:** This tests understanding of scene representation across MJCF handling, kitchen.py, and scene construction. (A) is correct - MJCF separates structure (body/joint), visuals (geom with visual properties), and collision (geom with collision properties). (B) is correct - edit_model_xml() in kitchen.py modifies MJCF for textures, naming, and rendering. (D) is correct - KitchenArena in models/scenes/ handles scene construction and provides fixture access. (C) is incorrect - while state can be vectorized, the framework maintains structured state representation through MuJoCo's data structures. (E) is incorrect - models are loaded during _load_model(), not lazily.

---

## robocasa - Problem 15 (Single Choice)
Multi-stage tasks require coordinating multiple subtasks with different success conditions. How does the framework's architecture support compositional task specification without requiring explicit state machines?

- (A) Multi-stage tasks use a blackboard architecture where subtasks communicate through shared memory
- (B) Multi-stage tasks are implemented as monolithic classes that check all subtask conditions in a single _check_success() method, relying on the task designer to implement the coordination logic
- (C) The framework automatically infers task structure from the directory hierarchy and generates state machines
- (D) Multi-stage tasks use behavior trees to compose atomic skills with fallback and sequence nodes

**Correct Answer: B**

**Explanation:** This tests understanding of multi-stage task architecture across multi_stage/ task implementations. (B) is correct - examining multi-stage tasks (e.g., PrepareCoffee, FillKettle) shows they implement all logic in a single class with _check_success() checking multiple conditions. There's no automatic state machine or behavior tree system; task designers manually implement the coordination. This is simpler but less modular than hierarchical approaches. (A) is incorrect - no blackboard pattern exists. (C) is incorrect - no automatic inference from directory structure. (D) is incorrect - no behavior tree system is implemented.

---

## robocasa - Problem 16 (Multiple Choice)
The framework must handle different robot morphologies with varying kinematic capabilities. Which design patterns enable cross-embodiment support?

- (A) The _ROBOT_POS_OFFSETS dictionary provides robot-specific base position offsets to account for different robot heights and base geometries
- (B) Robot models are defined in RoboSuite with standardized interfaces (robot_model, gripper_model, base_model) that Kitchen can interact with uniformly
- (C) The compute_robot_base_placement_pose() method applies robot-specific offsets when positioning robots relative to fixtures
- (D) All robots must have identical joint configurations to ensure policy transferability
- (E) The refactor_composite_controller_config() function adapts controller configurations to different robot control structures

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of cross-embodiment support across kitchen.py, config_utils.py, and RoboSuite integration. (A) is correct - _ROBOT_POS_OFFSETS defines per-robot offsets for GR1, G1, GoogleRobot, etc. (B) is correct - RoboSuite's robot abstraction provides uniform interfaces. (C) is correct - compute_robot_base_placement_pose() uses _ROBOT_POS_OFFSETS for robot-specific positioning. (E) is correct - refactor_composite_controller_config() adapts configs to different control structures. (D) is incorrect - robots have different joint configurations; the framework handles this through abstraction, not requiring identical kinematics.

---

## robocasa - Problem 17 (Multiple Choice)
Visual rendering quality impacts both policy learning and human interpretability. Which rendering techniques does the framework employ to balance realism with computational efficiency?

- (A) The framework supports both onscreen (mjviewer) and offscreen rendering modes for different use cases (interactive visualization vs. headless training)
- (B) Photo-realistic rendering is achieved through MuJoCo's built-in renderer with configurable lighting and shadows
- (C) The render_gpu_device_id parameter enables distributed rendering across multiple GPUs for parallel data collection
- (D) Texture resolution is dynamically adjusted based on object distance from camera to optimize memory usage
- (E) The renderer_config parameter allows customization of camera settings (position, orientation, field of view) for different layouts

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of rendering architecture across kitchen.py and RoboSuite's rendering system. (A) is correct - has_renderer and has_offscreen_renderer control rendering modes. (B) is correct - MuJoCo 3.2.6 provides the rendering with lighting/shadows. (C) is correct - render_gpu_device_id in __init__ enables GPU selection. (E) is correct - renderer_config with cam_config is used in set_cameras() for layout-specific camera settings. (D) is incorrect - no dynamic texture LOD system exists; textures are fixed resolution.

---

## robocasa - Problem 18 (Multiple Choice)
The framework's integration with policy learning frameworks requires careful interface design. Which architectural decisions facilitate this integration?

- (A) The environment follows the OpenAI Gym interface with reset(), step(), and observation/action spaces
- (B) The tianshou dependency provides reinforcement learning algorithm implementations that work with the environment
- (C) Episode metadata and camera configurations are included in the info dictionary returned by step() to support off-policy learning
- (D) The framework provides pre-trained policy checkpoints that can be fine-tuned for new tasks
- (E) Integration with robomimic enables imitation learning from demonstration datasets

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of policy learning integration across kitchen.py, setup.py, and documentation. (A) is correct - Kitchen inherits from ManipulationEnv which follows Gym interface. (B) is correct - setup.py includes tianshou==0.4.10 dependency. (C) is correct - get_ep_meta() data is available for learning algorithms. (E) is correct - policy_learning.md documents robomimic integration for BC-Transformer and other algorithms. (D) is incorrect - the framework provides datasets and environments, but pre-trained checkpoints are not part of the core codebase.

---

## robocasa - Problem 19 (Single Choice)
The framework must handle the computational challenge of simulating complex scenes with many objects. What is the primary strategy for managing simulation performance?

- (A) Using level-of-detail (LOD) systems that simplify distant objects
- (B) Implementing spatial partitioning (octrees/BVH) for efficient collision detection
- (C) Setting lite_physics=True to reduce MuJoCo solver iterations while maintaining sufficient accuracy for manipulation, combined with careful control frequency selection (20Hz)
- (D) Offloading physics computation to GPU using MuJoCo's CUDA backend

**Correct Answer: C**

**Explanation:** This tests understanding of performance optimization across kitchen.py and MuJoCo configuration. (C) is correct - Kitchen.__init__ explicitly sets lite_physics=True and control_freq=20. lite_physics reduces solver iterations (fewer constraint solver passes), trading some accuracy for speed. The 20Hz control frequency balances responsiveness with computational cost. This is the primary performance strategy visible in the codebase. (A) is incorrect - no LOD system exists. (B) is incorrect - MuJoCo handles collision detection internally; no custom spatial partitioning is implemented. (D) is incorrect - MuJoCo 3.2.6 doesn't use GPU for physics (only rendering).

---

## robocasa - Problem 20 (Multiple Choice)
The object asset pipeline must handle diverse sources (Objaverse, AI-generated) with varying quality. Which quality control mechanisms are implemented?

- (A) The exclude lists in OBJ_CATEGORIES remove specific object instances with known defects (holes, inconsistent normals, inertia violations)
- (B) Automated mesh validation checks for watertight geometry, manifold edges, and proper UV mapping
- (C) Different scale factors for aigen vs objaverse assets compensate for inconsistent real-world sizing across sources
- (D) Object instances are manually curated and tested before inclusion in the asset library
- (E) The framework automatically repairs mesh defects using hole-filling and remeshing algorithms

**Correct Answer: A, C, D**

**Explanation:** This tests understanding of asset quality control across kitchen_objects.py and asset management. (A) is correct - OBJ_CATEGORIES contains exclude lists with comments explaining defects (e.g., 'bagel_8' excluded, 'bar_1' has holes, 'can_5' has inconsistent orientation). (C) is correct - different scale values for aigen vs objaverse indicate manual calibration for size consistency. (D) is correct - the existence of curated exclude lists and specific scale factors indicates manual curation. (B) is incorrect - no automated validation code exists in the codebase. (E) is incorrect - no automatic repair algorithms are implemented; defective assets are excluded instead.

---

## robocasa - Problem 21 (Multiple Choice)
The framework's scene generation must handle spatial reasoning about fixture relationships. Which mechanisms enable the system to reason about spatial configurations?

- (A) The get_fixture() method with FixtureType and fixture_id enables querying specific fixtures by type and instance
- (B) The OU.point_in_fixture() function tests whether a point lies within a fixture's 2D or 3D bounds
- (C) The get_rel_transform() function computes relative transformations between fixtures for positioning objects
- (D) A scene graph data structure maintains parent-child relationships between all scene elements
- (E) The fixture placement system uses constraint satisfaction to ensure fixtures don't overlap

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of spatial reasoning across kitchen.py, object_utils.py, and fixture management. (A) is correct - get_fixture() provides structured access to fixtures. (B) is correct - OU.point_in_fixture() is used in compute_robot_base_placement_pose() for spatial queries. (C) is correct - OU.get_rel_transform() computes relative poses between fixtures. (E) is correct - fixture placement uses SequentialCompositeSampler with collision checking. (D) is incorrect - while MuJoCo has a kinematic tree, there's no explicit scene graph abstraction in the Python code for spatial reasoning.

---

## robocasa - Problem 22 (Multiple Choice)
The framework's MJCF post-processing system must handle various model adjustments. Which transformations are applied during edit_model_xml()?

- (A) Renaming elements starting with 'base0_' to 'mobilebase0_' for backward compatibility with old PandaOmron demonstrations
- (B) Applying texture swaps by replacing texture file paths in visual geometry elements
- (C) Adjusting collision geometry resolution based on object importance for performance optimization
- (D) Setting rendering properties like reflectance and specularity for visual realism
- (E) Removing unused joints and bodies to reduce model complexity

**Correct Answer: A, B, D**

**Explanation:** This tests understanding of MJCF post-processing in kitchen.py's edit_model_xml(). (A) is correct - the code explicitly renames 'base0_' to 'mobilebase0_' with a comment about PandaOmron demos. (B) is correct - texture swap functions are called within edit_model_xml() to replace textures. (D) is correct - the method adjusts rendering properties for fixtures and objects. (C) is incorrect - no adaptive collision resolution exists. (E) is incorrect - no pruning of unused elements is performed.

---

