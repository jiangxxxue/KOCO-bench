# Trackerlab Multiple Choice Problems

## trackerLab - Problem 1 (Multiple Choice)
In the context of transferring policies from IsaacLab to MuJoCo via Sim2SimLib, which architectural differences between the two simulators necessitate the observation and action mapping layers?

- (A) IsaacLab uses GPU-accelerated parallel environments while MuJoCo runs single-threaded CPU simulation
- (B) Joint ordering and naming conventions differ between IsaacLab URDF imports and MuJoCo XML definitions
- (C) IsaacLab provides observations in world frame while MuJoCo uses body-centric frames by default
- (D) The two simulators use different contact models (penalty-based vs constraint-based) affecting ground reaction forces
- (E) IsaacLab's articulation API returns data in different tensor shapes than MuJoCo's mjData structure

**Correct Answer: B, C, E**

**Explanation:** (B), (C), and (E) are correct. This tests deep understanding of sim-to-sim transfer challenges across `sim2simlib/model/sim2sim_base.py`, `sim2simlib/model/config.py`, and the joint mapping system. (B) is critical - the `policy_joint_names` configuration and joint reordering logic in Sim2SimLib directly address this. (C) reflects the frame convention differences requiring gravity orientation transformations. (E) captures the data structure mismatch requiring observation reshaping. (A) is a performance difference, not an architectural mapping requirement. (D) affects physics fidelity but doesn't necessitate observation/action remapping - both simulators can produce similar observations despite different contact solvers.

---

## trackerLab - Problem 2 (Multiple Choice)
In the DC motor model implementation within Sim2SimLib, which physical phenomena are captured by the velocity-dependent torque limit formulation?

- (A) Back-EMF reducing available voltage as motor speed increases
- (B) Coulomb friction transitioning to viscous friction at higher velocities
- (C) Thermal limitations causing torque derating at sustained high speeds
- (D) The fundamental torque-speed tradeoff in DC motors where maximum torque occurs at stall
- (E) Gear backlash effects becoming more pronounced at higher angular velocities

**Correct Answer: A, D**

**Explanation:** (A) and (D) are correct. This tests motor modeling domain knowledge from `sim2simlib/model/actuator_motor.py`. The DC motor model uses: τ_max(q̇) = τ_sat * (1 - |v_ratio|) where v_ratio = q̇/q̇_max. This captures (A) back-EMF: as motor spins faster, back-EMF opposes applied voltage, reducing current and thus torque. (D) is the macroscopic manifestation: maximum torque at zero speed (stall), decreasing linearly to zero at maximum speed. (B) is incorrect - friction is separate from the torque-speed curve. (C) thermal effects aren't modeled in this formulation. (E) backlash is a position-domain phenomenon, not captured by velocity-dependent limits.

---

## trackerLab - Problem 3 (Multiple Choice)
The SkillGraph builds a patch-level transition graph where intra-skill transitions have zero cost while inter-skill transitions have computed costs. What design principles does this cost structure enable?

- (A) Hierarchical motion planning where high-level policies select skills and low-level tracking follows patches
- (B) Automatic motion segmentation based on natural motion boundaries
- (C) Preference for completing initiated skills before transitioning, reducing motion fragmentation
- (D) Gradient-based optimization of skill sequences using differentiable graph traversal
- (E) Efficient shortest-path planning for skill composition using graph search algorithms

**Correct Answer: A, C, E**

**Explanation:** (A), (C), and (E) are correct. This tests skill composition architecture understanding from `trackerLab/managers/skill_manager/skill_graph.py` and `skill_manager.py`. (A) is enabled because zero intra-skill costs create natural skill boundaries for hierarchical decomposition. (C) follows directly - once in a skill, zero-cost sequential patches encourage completion before expensive inter-skill transitions. (E) is correct - the cost structure enables Dijkstra or A* for finding optimal skill sequences. (B) is incorrect - segmentation is predefined, not automatic from costs. (D) is wrong - the graph uses discrete costs, not differentiable for gradient-based optimization.

---

## trackerLab - Problem 4 (Multiple Choice)
When computing body-frame normalized motion terms (lvbs, ltbs, avbs) in MotionLib, which coordinate transformations are applied?

- (A) Inverse quaternion rotation to transform global velocities into the root body frame
- (B) Subtraction of the root position to obtain relative translations
- (C) Cross product with the root orientation to compute angular momentum
- (D) Quaternion conjugate multiplication to express vectors in local coordinates
- (E) Projection onto the ground plane to remove vertical components

**Correct Answer: A, D**

**Explanation:** (A) and (D) are correct (they describe the same operation). This tests transformation mathematics from `trackerLab/motion_buffer/motion_lib/motion_lib.py` and `poselib/core/rotation3d.py`. The code uses quat_apply_inverse(root_rot, global_vector) which applies the inverse rotation (quaternion conjugate for unit quaternions) to transform global-frame vectors into the body frame. (B) is incorrect - translation normalization uses inverse rotation, not subtraction. (C) is wrong - angular momentum isn't computed here. (E) is false - no ground plane projection occurs in these transformations.

---

## trackerLab - Problem 5 (Multiple Choice)
The MotionManager's calc_current_pose method constructs a state dictionary for environment resets. Which design decisions reflect the integration between motion tracking and IsaacLab's articulation system?

- (A) Joint positions are filled using id_caster.fill_2lab to handle partial joint coverage between motion and simulation
- (B) Root pose combines position and quaternion rotation in a 7D vector matching IsaacLab's articulation state format
- (C) Velocities are scaled by the simulation timestep to convert from per-frame to per-second units
- (D) The state dictionary uses nested keys matching IsaacLab's scene graph structure (articulation/robot)
- (E) Initial poses are cloned and stored separately to enable delta-based observations

**Correct Answer: A, B, D, E**

**Explanation:** (A), (B), (D), and (E) are correct. This tests integration architecture from `trackerLab/managers/motion_manager/motion_manager.py` and IsaacLab's articulation API. (A) is critical - motion data may not cover all simulation joints (e.g., hands), requiring selective filling. (B) matches IsaacLab's root_pose format: [x,y,z, qx,qy,qz,qw]. (D) reflects IsaacLab's hierarchical state structure. (E) enables computing relative observations (current - initial). (C) is incorrect - velocities are already in per-second units from motion data.

---

## trackerLab - Problem 6 (Multiple Choice)
When the SkillManager updates skills based on patch_time exceeding patch_time_curr, which state management strategies are employed?

- (A) Patch time is decremented by patch_time rather than reset to zero, preserving sub-patch timing precision
- (B) Environments with target skills set (env_update_target_skill > 0) are updated deterministically before random transitions
- (C) The skill transition policy is only invoked for environments without explicit target skills
- (D) All environments are synchronized to transition simultaneously when any environment's timer expires
- (E) Target skill indicators are cleared after successful transitions to prevent repeated forced transitions

**Correct Answer: A, B, C, E**

**Explanation:** (A), (B), (C), and (E) are correct. This tests skill transition logic from `trackerLab/managers/skill_manager/skill_manager.py`. (A) preserves timing accuracy: 'self.patch_time_curr[update_envs] -= self.patch_time' maintains fractional time. (B) and (C) implement a priority system: explicit targets override random policy. (E) prevents state corruption: 'self.env_update_target_skill[env_ids] = -1' after transition. (D) is wrong - environments transition independently based on individual timers.

---

## trackerLab - Problem 7 (Multiple Choice)
In IsaacLab's GPU-accelerated simulation, which aspects of the MotionManager's design enable efficient parallel motion tracking across thousands of environments?

- (A) All motion state tensors (loc_root_pos, loc_dof_pos, etc.) are batched with environment dimension first
- (B) Frame interpolation using SLERP is implemented as vectorized operations on entire batches
- (C) The MotionBuffer pre-computes motion library indices for all environments simultaneously
- (D) Joint ID casting uses tensor indexing operations instead of loops over individual environments
- (E) Motion time updates are conditional per-environment but executed as masked tensor operations

**Correct Answer: A, B, D, E**

**Explanation:** (A), (B), (D), and (E) are correct. This tests GPU parallelization understanding from `trackerLab/managers/motion_manager/motion_manager.py` and `trackerLab/motion_buffer/`. (A) enables SIMD operations: tensors are [num_envs, ...]. (B) SLERP operates on batched quaternions simultaneously. (D) id_caster uses tensor indexing: 'dof_pos[:, self.gym2lab_dof_ids]' processes all envs at once. (E) static_motion check uses masked updates. (C) is incorrect - MotionBuffer doesn't pre-compute indices; it queries MotionLib dynamically based on current motion_ids and motion_times.

---

## trackerLab - Problem 8 (Single Choice)
In Sim2SimLib's observation history mechanism, why is temporal observation stacking critical for policy transfer from IsaacLab to MuJoCo?

- (A) MuJoCo's single-threaded execution requires buffering observations for batch processing
- (B) Policies trained with observation history implicitly learn velocity and acceleration from finite differences
- (C) Observation history compensates for MuJoCo's lower simulation frequency compared to IsaacLab
- (D) Historical observations enable the policy to detect and correct for simulation divergence

**Correct Answer: B**

**Explanation:** (B) is correct. This tests policy architecture understanding from `sim2simlib/model/config.py` and observation processing. When policies are trained with observation history (e.g., last 3 timesteps of joint positions), they learn to extract derivatives: velocity ≈ (pos_t - pos_{t-1})/dt, acceleration ≈ (vel_t - vel_{t-1})/dt. This is crucial because if IsaacLab training used history but MuJoCo deployment doesn't provide it, the policy loses critical dynamic information. The history must be maintained for transfer. (A) is wrong - batching isn't the reason. (C) misunderstands - both can run at same frequency. (D) is incorrect - history doesn't detect sim differences.

---

## trackerLab - Problem 9 (Multiple Choice)
The SkeletonTree class represents skeleton topology as a tree structure. Which kinematic computations does this tree representation enable?

- (A) Forward kinematics: computing global joint positions from local rotations by traversing parent-to-child
- (B) Inverse kinematics: solving for joint angles given desired end-effector positions
- (C) Jacobian computation: calculating the relationship between joint velocities and end-effector velocities
- (D) Joint velocity propagation: computing child joint velocities from parent velocities and local angular velocities
- (E) Center of mass calculation: aggregating link masses weighted by global positions

**Correct Answer: A, D, E**

**Explanation:** (A), (D), and (E) are correct. This tests skeleton kinematics from `poselib/skeleton/skeleton3d.py`. (A) is fundamental - tree traversal enables FK: child_global_pos = parent_global_pos + parent_global_rot * bone_vector. (D) follows from the tree: child velocity includes parent velocity plus the contribution from parent's angular velocity. (E) uses tree structure to aggregate: CoM = Σ(mass_i * global_pos_i) / total_mass. (B) is incorrect - IK requires optimization/search, not just tree structure. (C) is wrong - Jacobian needs explicit derivative computation, not provided by tree alone.

---

## trackerLab - Problem 10 (Multiple Choice)
The MotionManager's loc_gen_state method computes motion state by interpolating between two frames. Which numerical considerations affect the interpolation quality?

- (A) Quaternion normalization after SLERP to maintain unit norm despite floating-point errors
- (B) Handling quaternion double-cover (q and -q represent the same rotation) by choosing the shorter path
- (C) Clamping the blend factor to [0,1] to prevent extrapolation beyond frame boundaries
- (D) Using double precision for rotation computations to minimize accumulation errors
- (E) Applying temporal smoothing filters to reduce high-frequency jitter in interpolated states

**Correct Answer: B, C**

**Explanation:** (B) and (C) are correct. This tests numerical methods from `trackerLab/managers/motion_manager/motion_manager.py` and `trackerLab/utils/torch_utils/slerp`. (B) is critical - quaternions have double-cover: q and -q represent the same rotation. SLERP must choose the shorter arc (dot product check) to avoid 360° rotations. (C) prevents extrapolation: blend = (motion_time - f0_time) / dt should be clamped. (A) is unnecessary - proper SLERP maintains unit norm. (D) is not implemented - single precision is used. (E) is not part of loc_gen_state.

---

## trackerLab - Problem 11 (Single Choice)
Why does Sim2SimLib implement control_decimation to run policy inference less frequently than the simulation timestep?

- (A) To reduce computational cost by avoiding redundant policy evaluations
- (B) To match the control frequency used during IsaacLab training, maintaining temporal consistency
- (C) To simulate communication delays between high-level planning and low-level control
- (D) To allow the PID controller time to stabilize between policy updates

**Correct Answer: B**

**Explanation:** (B) is correct. This tests control frequency matching from `sim2simlib/model/config.py` and `sim2sim_base.py`. If IsaacLab training used control_decimation=4 (policy runs every 4 sim steps), the policy learned to produce actions assuming this frequency. Running at a different frequency in MuJoCo would change the temporal dynamics - actions would be held for different durations, affecting the policy's behavior. Matching the training frequency is critical for transfer. (A) is a side benefit, not the primary reason. (C) mischaracterizes the purpose. (D) is incorrect - PID operates at simulation frequency regardless.

---

## trackerLab - Problem 12 (Multiple Choice)
In the SkeletonMotion class, which temporal operations are supported for motion sequence manipulation?

- (A) Computing joint velocities from position sequences using finite differences
- (B) Time warping to change motion speed while preserving spatial characteristics
- (C) Temporal alignment of multiple motion sequences using dynamic time warping
- (D) Frame interpolation to increase motion framerate
- (E) Motion blending to create smooth transitions between different motion clips

**Correct Answer: A, D**

**Explanation:** (A) is correct, (D) is supported through interpolation methods. This tests SkeletonMotion capabilities from `poselib/skeleton/skeleton3d.py`. (A) is explicitly mentioned: 'provides utilities for computing joint velocities' from position sequences. (D) is supported via the interpolation infrastructure. (B) time warping isn't explicitly implemented. (C) DTW is not provided. (E) motion blending between clips isn't a SkeletonMotion feature (though frame-level interpolation exists).

---

## trackerLab - Problem 13 (Multiple Choice)
When designing the skill transition cost function in SkillGraph, which motion similarity metrics could be used to compute inter-skill transition costs?

- (A) L2 distance between root body translations at patch boundaries
- (B) Quaternion distance between root body orientations at patch boundaries
- (C) Joint configuration space distance using weighted joint angle differences
- (D) Dynamic Time Warping distance between velocity profiles
- (E) Frechet distance between full-body trajectory curves

**Correct Answer: A, B, C**

**Explanation:** (A), (B), and (C) are correct. This tests skill graph construction from `trackerLab/managers/skill_manager/skill_graph/skill_edge.py`. The BUILD_GRAPH_METHODS use metrics like 'trans_base' (A - root translation), orientation differences (B), and joint space distances (C) with configurable norms (L2) and windows. These are computationally efficient boundary-based metrics. (D) DTW is too expensive for real-time graph construction. (E) Frechet distance over full trajectories is computationally prohibitive and not used in the implementation.

---

## trackerLab - Problem 14 (Single Choice)
Why does the retargeting pipeline require both source and target T-poses to be in the same canonical pose (e.g., both in T-pose)?

- (A) To ensure both skeletons have the same number of joints
- (B) To establish a common reference frame where bone direction vectors can be compared for computing rotation offsets
- (C) To normalize bone lengths to unit vectors for scale-invariant comparison
- (D) To simplify the joint mapping by ensuring corresponding joints have similar names

**Correct Answer: B**

**Explanation:** (B) is correct. This tests retargeting mathematics from `poselib/retarget/retargeting_processor.py`. The T-pose serves as a canonical configuration where we can compute the rotation offset needed to align corresponding bones. For example, if source's upper arm points in direction v_s and target's points in direction v_t (both in their respective T-poses), the rotation offset R satisfies: R * v_s = v_t. This offset is then applied to all motion frames. Both must be in the same pose type (T-pose) so the bone directions are comparable. (A) is wrong - joint counts can differ. (C) misunderstands - bone lengths aren't normalized. (D) is incorrect - naming is separate from pose matching.

---

## trackerLab - Problem 15 (Multiple Choice)
In MuJoCo's constraint-based contact solver versus IsaacLab's penalty-based contact model, which behavioral differences affect policy transfer?

- (A) Constraint-based solvers produce hard contacts with no penetration, while penalty-based allow small penetrations
- (B) Constraint-based solvers can generate instantaneous velocity changes, while penalty-based produce gradual force responses
- (C) Constraint-based solvers are deterministic, while penalty-based introduce stochastic contact dynamics
- (D) Constraint-based solvers better preserve energy conservation in collisions
- (E) Constraint-based solvers require smaller timesteps for stability

**Correct Answer: A, B, D**

**Explanation:** (A), (B), and (D) are correct. This tests physics simulation understanding relevant to `sim2simlib/` transfer. (A) constraint solvers enforce non-penetration exactly via Lagrange multipliers, while penalty methods use spring-damper forces allowing small penetration. (B) constraints can produce discontinuous velocity changes (impulses), while penalty forces integrate smoothly. (D) constraint solvers can exactly enforce restitution coefficients, better preserving energy. (C) is wrong - both can be deterministic. (E) is backwards - penalty methods often need smaller timesteps due to stiff springs.

---

## trackerLab - Problem 16 (Multiple Choice)
The JointIdCaster maintains mappings between gym and lab joint orderings. Which scenarios require this bidirectional mapping?

- (A) Reading motion data (gym order) and writing to IsaacLab simulation state (lab order)
- (B) Reading IsaacLab observations (lab order) and computing motion tracking errors (gym order)
- (C) Applying retargeted joint positions (gym order) to robot actuators (lab order)
- (D) Converting policy actions (lab order) to motion library format for logging (gym order)
- (E) Synchronizing joint states between distributed training workers

**Correct Answer: A, B, C**

**Explanation:** (A), (B), and (C) are correct. This tests bidirectional mapping use cases from `trackerLab/joint_id_caster.py` and its usage in managers. (A) is the primary use: motion data → simulation. (B) is needed for computing tracking rewards: simulation state → motion reference frame. (C) applies retargeted data to simulation. (D) is incorrect - policy actions are already in lab order and don't need conversion to gym for logging. (E) is unrelated - distributed training doesn't involve gym/lab conversion.

---

## trackerLab - Problem 17 (Multiple Choice)
When computing body-frame observations (lvbs, avbs) from global motion data, which physical interpretations do these transformed quantities represent?

- (A) lvbs represents the velocity of the root body as perceived by an observer fixed to the body
- (B) avbs represents the angular velocity vector in the body's local coordinate frame
- (C) lvbs eliminates the effect of global translation, isolating relative motion
- (D) avbs is invariant to the body's global orientation, representing intrinsic rotation rate
- (E) Both quantities are expressed in the inertial frame but relative to the body's position

**Correct Answer: A, B, D**

**Explanation:** (A), (B), and (D) are correct. This tests reference frame understanding from `trackerLab/motion_buffer/motion_lib/motion_lib.py` and `poselib/core/rotation3d.py`. (A) is correct - quat_apply_inverse(root_rot, global_vel) transforms global velocity into body frame, giving velocity as seen by body-fixed observer. (B) is correct - angular velocity is rotated into body frame. (D) is a key property - angular velocity magnitude and direction (in body frame) are independent of global orientation. (C) is wrong - lvbs doesn't eliminate translation effects, it changes reference frame. (E) is backwards - they're in body frame, not inertial frame.

---

