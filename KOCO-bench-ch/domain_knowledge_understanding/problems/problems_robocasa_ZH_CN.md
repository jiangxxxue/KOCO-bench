# Robocasa 选择题

## robocasa - 问题 1 (多选题)
在为厨房操作任务设计物理仿真时，RoboCasa必须平衡多个相互竞争的目标。代码库中的哪些设计决策反映了仿真保真度与计算效率之间的权衡？

- (A) 在Kitchen环境初始化中使用lite_physics=True来减少计算开销，同时保持足够的接触动力学用于操作
- (B) 将control_freq设置为20Hz作为响应控制和仿真稳定性之间的折衷，避免需要更小的积分时间步长
- (C) 在对象放置后实现稳定阶段，仿真在没有机器人动作的情况下经过多次迭代以稳定物理
- (D) 对大多数对象使用单一碰撞几何体而不是分解的凸包，以降低碰撞检测复杂度
- (E) 禁用机器人连杆的自碰撞检查以防止约束求解器中的数值不稳定性

**正确答案：A, B, C**

**解析：** 这测试对kitchen.py、placement_samplers.py和对象模型中物理仿真权衡的理解。(A)正确 - lite_physics减少求解器迭代次数。(B)正确 - 20Hz平衡控制响应性与稳定积分。(C)正确 - _reset_internal()中的稳定阶段防止对象由于初始穿透而掉落。(D)不正确 - 代码库使用来自Objaverse/AI生成模型的详细碰撞网格。(E)不正确 - 自碰撞通常为关节机器人启用以防止不现实的配置。

---

## robocasa - 问题 2 (多选题)
RoboCasa的程序化场景生成系统在放置对象和固定装置时必须处理复杂的空间约束。哪些架构模式使框架能够生成有效的、无碰撞的场景，同时保持任务相关的空间关系？

- (A) SequentialCompositeSampler链接多个放置初始化器，允许对象相对于先前放置的对象或固定装置进行放置
- (B) UniformRandomSampler的obj_in_region()检查通过测试区域边界点来确保对象边界框完全保持在指定区域内
- (C) 放置系统使用两阶段方法：首先使用fxtr_placements放置固定装置，然后使用可以引用固定装置位置的object_placements放置对象
- (D) objs_intersect()碰撞检测使用轴对齐边界框(AABB)测试以提高计算效率
- (E) 放置采样器支持层次包含，其中对象可以放置在容器对象内部，然后将容器添加到场景中

**正确答案：A, B, C, E**

**解析：** 这测试对placement_samplers.py、kitchen.py和object_utils.py中程序化生成架构的理解。(A)正确 - SequentialCompositeSampler通过placed_objects跟踪实现链接。(B)正确 - obj_in_region()测试所有对象边界点是否位于由p0、px、py定义的区域内。(C)正确 - _load_model()首先采样fxtr_placements，然后object_placements可以引用它们。(E)正确 - _create_objects()实现try_to_place_in逻辑用于容器。(D)不正确 - objs_intersect()使用带旋转的定向边界框(OBB)测试，而非简单的AABB。

---

## robocasa - 问题 3 (单选题)
框架支持具有不同运动学结构的多种机器人具身形态(移动操作臂、人形机器人)。RoboCasa中的复合控制器架构如何处理异构控制模态(手臂、夹爪、基座、躯干)的协调？

- (A) 每个身体部位都有一个独立的控制器，将命令发布到共享动作缓冲区，然后由中央仲裁器同步
- (B) refactor_composite_controller_config()函数重构控制器配置以匹配RoboSuite v1.5的复合控制器格式，body_part_ordering指定控制序列
- (C) 分层状态机根据任务阶段协调不同控制模式(导航、操作、全身)之间的转换
- (D) 所有身体部位共享一个统一的控制器，使用全身逆运动学计算关节命令

**正确答案：B**

**解析：** 这测试对kitchen.py和config_utils.py中机器人控制架构的理解。(B)正确 - refactor_composite_controller_config()将旧控制器配置转换为新的复合格式，对于PandaOmron特别设置body_part_ordering=['right', 'right_gripper', 'base', 'torso']来定义控制结构。这使RoboSuite的复合控制器能够正确协调不同的身体部位。(A)不正确 - 不存在动作缓冲区仲裁模式。(C)不正确 - 不存在用于模式切换的分层状态机。(D)不正确 - 身体部位有独立的控制器，而非统一的全身IK。

---

## robocasa - 问题 4 (多选题)
机器人学习的计算机视觉系统需要仔细的相机放置和配置。哪些设计原则指导RoboCasa的相机系统架构以支持策略学习和人类可解释性？

- (A) CamUtils中的_cam_configs系统提供机器人特定的相机配置，考虑不同的机器人形态和工作空间几何
- (B) 相机随机化对agentview相机姿态应用高斯噪声，但不对eye-in-hand相机应用，以保持一致的末端执行器观察
- (C) set_cameras()方法根据厨房layout_id动态调整相机位置，以确保相关固定装置保持在视野中
- (D) 多个相机视点(agentview_center、agentview_left、agentview_right)提供重叠覆盖以进行深度估计和遮挡处理
- (E) translucent_robot渲染模式通过使机器人半透明来减少视觉遮挡，提高操作目标的可见性

**正确答案：A, B, C, E**

**解析：** 这测试对camera_utils.py、kitchen.py和可视化代码中视觉系统设计的理解。(A)正确 - get_robot_cam_configs()返回机器人特定配置。(B)正确 - _randomize_cameras()对agentview应用噪声，但对eye_in_hand设置pos_noise=np.zeros_like()。(C)正确 - set_cameras()使用LAYOUT_CAMS.get(layout_id)进行布局特定定位。(E)正确 - visualize()在translucent_robot=True时为机器人几何体设置alpha=0.10。(D)不正确 - 虽然存在多个视点，但它们不是专门为立体深度估计设计的；它们为策略学习提供不同的视角。

---

## robocasa - 问题 5 (单选题)
texture_swap系统通过程序化改变材质外观来实现视觉域随机化。纹理随机化与场景生成中的generative_textures参数之间的架构关系是什么？

- (A) 当generative_textures='100p'时，所有纹理都被AI生成的替代品替换；当为None/False时，使用资产库中的人工设计纹理
- (B) 纹理交换系统独立于generative_textures运行，后者仅控制对象几何生成
- (C) 纹理随机化在每次reset()调用时应用以最大化视觉多样性，而generative_textures是一次性场景初始化参数
- (D) 系统使用混合方法，其中generative_textures控制固定装置纹理，而对象纹理总是随机化的

**正确答案：A**

**解析：** 这测试对texture_swap.py和kitchen.py中纹理系统架构的理解。(A)正确 - _curr_gen_fixtures变量存储来自_ep_meta的生成纹理信息，edit_model_xml()使用它来确定是应用AI生成的纹理(当为'100p'时)还是使用标准纹理。texture_swap.py中的replace_*_texture()函数处理实际交换。(B)不正确 - generative_textures影响视觉外观，而非几何。(C)不正确 - 纹理选择发生在_load_model()期间，而非每次reset()。(D)不正确 - 该参数统一影响固定装置和对象。

---

## robocasa - 问题 6 (多选题)
对象采样系统必须处理多样的物理属性和任务需求。哪些机制使框架能够选择任务适当的对象同时保持物理真实性？

- (A) OBJ_CATEGORIES字典编码物理功能(可抓取、可清洗、可微波、可烹饪、可冷冻)，根据任务需求约束对象选择
- (B) 对象缩放因子(aigen vs objaverse)补偿不同资产源具有不一致的真实世界尺寸
- (C) obj_instance_split参数通过分区对象实例实现训练/测试数据集分割，'A'使用大多数实例，'B'使用其余实例
- (D) 对象类别中的排除列表删除具有网格缺陷(孔洞、不一致法线)的特定实例，这些缺陷会导致物理不稳定性
- (E) 对象组(水果、蔬菜、餐具等)按层次组织以支持语义任务规范

**正确答案：A, B, C, D, E**

**解析：** 这测试对kitchen_objects.py、kitchen.py和sample_object()中对象系统设计的理解。(A)正确 - OBJ_CATEGORIES定义在sample_object()过滤中使用的功能。(B)正确 - aigen与objaverse的不同缩放值补偿资产不一致性。(C)正确 - obj_instance_split实现如文档所述的训练/测试分割。(D)正确 - 排除列表删除有问题的网格(例如，'bagel_8'、'bar_1'有孔洞)。(E)正确 - types元组创建层次分组(例如，apple有types=('fruit',))。

---

## robocasa - 问题 7 (单选题)
场景生成必须处理创建多样化厨房布局同时确保机器人可达性的挑战。布局和风格系统如何平衡环境多样性与任务可行性？

- (A) scene_registry维护120个预设计厨房场景的固定集合，这些场景根据layout_id和style_id组合确定性加载
- (B) 布局使用基于语法的规则程序化生成，而风格从固定的材质/颜色方案调色板中选择
- (C) EXCLUDE_LAYOUTS机制允许任务过滤掉空间约束使任务不可行的布局，例如固定装置相距太远
- (D) 通过场景生成期间的工作空间分析，保证所有布局对所有机器人类型都是可达的

**正确答案：C**

**解析：** 这测试对scene_registry.py、kitchen.py和任务实现中场景生成约束的理解。(C)正确 - 任务可以设置EXCLUDE_LAYOUTS(例如，PnPCounterToMicrowave排除布局8因为微波炉离柜台太远)，layout_and_style_ids被过滤以删除排除的布局。这平衡了多样性(许多布局)与可行性(任务特定过滤)。(A)不正确 - 虽然有120个场景，但它们是布局和风格的组合，而非完全预设计。(B)不正确 - 布局是预设计的，非程序化生成。(D)不正确 - 不保证可达性；任务必须通过排除来处理这个问题。

---

## robocasa - 问题 8 (多选题)
框架必须在高级任务规范和低级控制之间架起桥梁。哪些架构组件促进了这种分层任务分解？

- (A) single_stage(原子)和multi_stage(复合)任务之间的区别反映了两级层次结构，其中复合任务对多个原子技能进行排序
- (B) 任务类中的_get_obj_cfgs()方法指定对象配置，定义任务执行所需的初始场景状态
- (C) staged()装饰器系统使任务能够定义顺序子任务，具有自动状态转换和奖励塑造
- (D) 任务类继承自Kitchen，后者提供固定装置和对象管理，而任务特定逻辑在_check_success()和奖励计算中实现
- (E) 来自RoboSuite的ManipulationTask模型提供底层MJCF组合，合并场地、机器人和对象

**正确答案：A, B, D, E**

**解析：** 这测试对任务实现、kitchen.py和RoboSuite集成中任务架构的理解。(A)正确 - 目录结构和任务组织明确分离原子(single_stage/)和复合(multi_stage/)任务。(B)正确 - _get_obj_cfgs()定义任务的对象需求。(D)正确 - Kitchen提供基础设施，而任务实现成功条件。(E)正确 - _load_model()中的ManipulationTask处理MJCF合并。(C)不正确 - 不存在staged()装饰器系统；多阶段任务作为单体类实现，而非具有自动子任务转换。

---

## robocasa - 问题 9 (单选题)
接触丰富的操作任务(如打开抽屉和门)需要仔细建模摩擦和接触动力学。RoboCasa物理配置中的哪个设计选择最直接影响这些交互的稳定性和真实性？

- (A) 使用MuJoCo的隐式接触模型与互补约束来处理刚性接触，而不需要极小的时间步长
- (B) 实现自定义接触回调，检测抽屉/门接触并应用校正力以防止穿透
- (C) 设置lite_physics=True，这减少了求解器迭代但保持了足够的接触分辨率用于操作任务
- (D) 使用混合物理方法，其中接触在MuJoCo中检测但使用单独的基于约束的求解器解决

**正确答案：A**

**解析：** 这测试对kitchen.py和MuJoCo集成中接触动力学和物理求解器配置的理解。(A)正确 - MuJoCo的隐式接触模型(默认)使用互补约束有效处理刚性接触，这对抽屉/门操作至关重要。框架通过使用mujoco==3.2.6及其默认接触设置来依赖这一点。(C)部分相关但不是最直接的因素 - lite_physics影响整体仿真速度但不从根本上改变接触处理。(B)不正确 - 不存在自定义接触回调。(D)不正确 - MuJoCo内部处理所有接触解决。

---

## robocasa - 问题 10 (多选题)
演示收集和数据集生成需要仿真、控制和数据记录之间的仔细协调。哪些组件协同工作以实现高质量的轨迹收集？

- (A) collect_demos.py脚本与遥操作设备(键盘、spacemouse)集成，记录具有同步状态和观察数据的人类演示
- (B) dataset_registry.py维护任务名称和数据集路径之间的映射，支持多种数据集类型(human_raw、human_im、mg_im)
- (C) 剧集元数据(get_ep_meta())捕获场景配置(layout_id、style_id、object_cfgs)以实现确定性重放
- (D) h5py格式以分层组织存储轨迹：每个时间步的状态、动作、观察和奖励
- (E) MimicGen集成通过从人类种子生成合成演示来实现自动数据增强

**正确答案：A, B, C, D, E**

**解析：** 这测试对collect_demos.py、dataset_registry.py、kitchen.py和MimicGen集成中数据收集管道的理解。(A)正确 - collect_demos.py处理遥操作和记录。(B)正确 - dataset_registry.py将任务映射到数据集路径和类型。(C)正确 - get_ep_meta()存储场景配置用于重放。(D)正确 - h5py用于轨迹存储(在脚本中引用)。(E)正确 - 代码库导入mimicgen并支持mg_im数据集。

---

## robocasa - 问题 11 (多选题)
关节对象操作(橱柜、抽屉、微波炉)对物理仿真和控制都提出了独特的挑战。框架设计的哪些方面专门解决这些挑战？

- (A) Fixture类(SingleCabinet、Drawer、Microwave)封装关节对象的关节定义和运动学约束
- (B) 带有FixtureType枚举的get_fixture()方法提供对场景中特定固定装置实例的类型安全访问
- (C) _reset_internal()中的关节位置初始化根据任务需求将关节对象设置为特定配置(打开/关闭)
- (D) 阻抗控制用于所有关节对象交互以安全处理接触力
- (E) 固定装置放置系统确保关节对象定位以避免操作期间的运动学奇点

**正确答案：A, B, C**

**解析：** 这测试对固定装置实现、kitchen.py和任务代码中关节对象处理的理解。(A)正确 - models/fixtures/中的固定装置类定义关节和约束。(B)正确 - 带有FixtureType的get_fixture()提供结构化访问。(C)正确 - 任务中的_reset_internal()通过set_joint_qpos()设置关节位置。(D)不正确 - 框架使用位置/速度控制，而非专门为固定装置使用阻抗控制。(E)不正确 - 放置关注碰撞避免和可达性，而非运动学奇点。

---

## robocasa - 问题 12 (多选题)
碰撞检测和避免对于在杂乱的厨房环境中安全操作至关重要。哪些机制有助于仿真中的碰撞处理？

- (A) objs_intersect()函数使用定向边界框(OBB)相交测试来检测放置期间对象之间的碰撞
- (B) MuJoCo的内置碰撞检测使用定义的碰撞几何体处理机器人、对象和固定装置之间的运行时碰撞
- (C) 采样器中的ensure_valid_placement标志启用/禁用对象放置期间的碰撞检查，以平衡场景有效性与生成速度
- (D) 运动规划模块为所有机器人运动计算无碰撞轨迹
- (E) 放置系统使用带碰撞检查的迭代采样，重试最多5000次以找到有效的对象位置

**正确答案：A, B, C, E**

**解析：** 这测试对placement_samplers.py、object_utils.py和物理仿真中碰撞处理的理解。(A)正确 - object_utils.py中的objs_intersect()执行OBB测试。(B)正确 - MuJoCo使用碰撞网格处理运行时碰撞检测。(C)正确 - UniformRandomSampler中的ensure_valid_placement控制是否执行碰撞检查。(E)正确 - UniformRandomSampler中的sample()方法使用range(5000)的循环进行重试。(D)不正确 - 不存在运动规划模块；框架依赖学习策略或遥操作。

---

## robocasa - 问题 13 (单选题)
框架必须处理生成真实对象交互(抓取、放置、堆叠)的挑战。什么物理建模选择最直接使多样对象几何体的稳定抓取成为可能？

- (A) 使用软接触模型，为每种对象材质类型调整柔顺参数
- (B) 实现基于吸力的抓取，不需要精确的接触建模
- (C) 依赖MuJoCo的摩擦金字塔模型，在对象MJCF文件中定义适当的摩擦系数以生成真实的接触力
- (D) 使用基于约束的抓取，其中对象在接触时刚性附着到夹爪

**正确答案：C**

**解析：** 这测试对对象模型和MuJoCo集成中抓取物理的理解。(C)正确 - MuJoCo使用摩擦金字塔模型进行接触，对象MJCF文件(来自Objaverse/AI生成资产)定义摩擦系数。框架依赖这种标准接触模型而不是实现自定义抓取物理。来自RoboSuite的夹爪模型提供夹爪几何体，MuJoCo的接触求解器处理抓取动力学。(A)不正确 - 虽然MuJoCo支持软接触，但框架不调整每种材质的柔顺性。(B)不正确 - 使用平行夹爪，而非吸力。(D)不正确 - 不存在刚性附着机制；抓取从接触力中产生。

---

## robocasa - 问题 14 (多选题)
框架的场景表示必须同时支持仿真和策略学习。哪些设计决策实现了高效的场景表示，同时保持学习所需的信息？

- (A) MJCF(MuJoCo XML)格式提供分层场景描述，分离运动学结构、视觉外观和碰撞几何体
- (B) edit_model_xml()方法后处理生成的MJCF以应用纹理交换、修复命名约定和调整渲染属性
- (C) 场景状态表示为关节位置和速度的平面向量，以实现高效的神经网络处理
- (D) KitchenArena类封装场景构建，提供查询固定装置位置和空间关系的方法
- (E) 对象和固定装置模型在仿真期间延迟加载以减少内存占用

**正确答案：A, B, D**

**解析：** 这测试对MJCF处理、kitchen.py和场景构建中场景表示的理解。(A)正确 - MJCF分离结构(body/joint)、视觉(带视觉属性的geom)和碰撞(带碰撞属性的geom)。(B)正确 - kitchen.py中的edit_model_xml()修改MJCF用于纹理、命名和渲染。(D)正确 - models/scenes/中的KitchenArena处理场景构建并提供固定装置访问。(C)不正确 - 虽然状态可以向量化，但框架通过MuJoCo的数据结构维护结构化状态表示。(E)不正确 - 模型在_load_model()期间加载，而非延迟加载。

---

## robocasa - 问题 15 (单选题)
多阶段任务需要协调具有不同成功条件的多个子任务。框架的架构如何支持组合任务规范而不需要显式状态机？

- (A) 多阶段任务使用黑板架构，其中子任务通过共享内存通信
- (B) 多阶段任务作为单体类实现，在单个_check_success()方法中检查所有子任务条件，依赖任务设计者实现协调逻辑
- (C) 框架从目录层次结构自动推断任务结构并生成状态机
- (D) 多阶段任务使用行为树来组合原子技能，具有回退和序列节点

**正确答案：B**

**解析：** 这测试对multi_stage/任务实现中多阶段任务架构的理解。(B)正确 - 检查多阶段任务(例如，PrepareCoffee、FillKettle)显示它们在单个类中实现所有逻辑，_check_success()检查多个条件。不存在自动状态机或行为树系统；任务设计者手动实现协调。这比分层方法更简单但模块化程度较低。(A)不正确 - 不存在黑板模式。(C)不正确 - 没有从目录结构自动推断。(D)不正确 - 未实现行为树系统。

---

## robocasa - 问题 16 (多选题)
框架必须处理具有不同运动学能力的不同机器人形态。哪些设计模式实现跨具身支持？

- (A) _ROBOT_POS_OFFSETS字典提供机器人特定的基座位置偏移，以考虑不同的机器人高度和基座几何体
- (B) 机器人模型在RoboSuite中定义，具有标准化接口(robot_model、gripper_model、base_model)，Kitchen可以统一交互
- (C) compute_robot_base_placement_pose()方法在相对于固定装置定位机器人时应用机器人特定偏移
- (D) 所有机器人必须具有相同的关节配置以确保策略可迁移性
- (E) refactor_composite_controller_config()函数使控制器配置适应不同的机器人控制结构

**正确答案：A, B, C, E**

**解析：** 这测试对kitchen.py、config_utils.py和RoboSuite集成中跨具身支持的理解。(A)正确 - _ROBOT_POS_OFFSETS为GR1、G1、GoogleRobot等定义每个机器人的偏移。(B)正确 - RoboSuite的机器人抽象提供统一接口。(C)正确 - compute_robot_base_placement_pose()使用_ROBOT_POS_OFFSETS进行机器人特定定位。(E)正确 - refactor_composite_controller_config()使配置适应不同的控制结构。(D)不正确 - 机器人具有不同的关节配置；框架通过抽象处理这个问题，而不要求相同的运动学。

---

## robocasa - 问题 17 (多选题)
视觉渲染质量影响策略学习和人类可解释性。框架采用哪些渲染技术来平衡真实性与计算效率？

- (A) 框架支持屏幕上(mjviewer)和离屏渲染模式，用于不同用例(交互式可视化vs无头训练)
- (B) 通过MuJoCo的内置渲染器实现照片级真实渲染，具有可配置的光照和阴影
- (C) render_gpu_device_id参数启用跨多个GPU的分布式渲染以进行并行数据收集
- (D) 纹理分辨率根据对象与相机的距离动态调整以优化内存使用
- (E) renderer_config参数允许为不同布局自定义相机设置(位置、方向、视场)

**正确答案：A, B, C, E**

**解析：** 这测试对kitchen.py和RoboSuite渲染系统中渲染架构的理解。(A)正确 - has_renderer和has_offscreen_renderer控制渲染模式。(B)正确 - MuJoCo 3.2.6提供带光照/阴影的渲染。(C)正确 - __init__中的render_gpu_device_id启用GPU选择。(E)正确 - 带有cam_config的renderer_config在set_cameras()中用于布局特定的相机设置。(D)不正确 - 不存在动态纹理LOD系统；纹理是固定分辨率。

---

## robocasa - 问题 18 (多选题)
框架与策略学习框架的集成需要仔细的接口设计。哪些架构决策促进了这种集成？

- (A) 环境遵循OpenAI Gym接口，具有reset()、step()和观察/动作空间
- (B) tianshou依赖提供与环境配合使用的强化学习算法实现
- (C) 剧集元数据和相机配置包含在step()返回的info字典中以支持离策略学习
- (D) 框架提供可以为新任务微调的预训练策略检查点
- (E) 与robomimic的集成使从演示数据集进行模仿学习成为可能

**正确答案：A, B, C, E**

**解析：** 这测试对kitchen.py、setup.py和文档中策略学习集成的理解。(A)正确 - Kitchen继承自遵循Gym接口的ManipulationEnv。(B)正确 - setup.py包含tianshou==0.4.10依赖。(C)正确 - get_ep_meta()数据可用于学习算法。(E)正确 - policy_learning.md记录了BC-Transformer和其他算法的robomimic集成。(D)不正确 - 框架提供数据集和环境，但预训练检查点不是核心代码库的一部分。

---

## robocasa - 问题 19 (单选题)
框架必须处理模拟具有许多对象的复杂场景的计算挑战。管理仿真性能的主要策略是什么？

- (A) 使用细节层次(LOD)系统简化远处的对象
- (B) 实现空间分区(八叉树/BVH)以实现高效的碰撞检测
- (C) 设置lite_physics=True以减少MuJoCo求解器迭代，同时保持足够的操作精度，结合仔细的控制频率选择(20Hz)
- (D) 使用MuJoCo的CUDA后端将物理计算卸载到GPU

**正确答案：C**

**解析：** 这测试对kitchen.py和MuJoCo配置中性能优化的理解。(C)正确 - Kitchen.__init__明确设置lite_physics=True和control_freq=20。lite_physics减少求解器迭代(更少的约束求解器通过)，以速度换取一些精度。20Hz控制频率平衡响应性与计算成本。这是代码库中可见的主要性能策略。(A)不正确 - 不存在LOD系统。(B)不正确 - MuJoCo内部处理碰撞检测；未实现自定义空间分区。(D)不正确 - MuJoCo 3.2.6不使用GPU进行物理(仅渲染)。

---

## robocasa - 问题 20 (多选题)
对象资产管道必须处理具有不同质量的多样来源(Objaverse、AI生成)。实现了哪些质量控制机制？

- (A) OBJ_CATEGORIES中的排除列表删除具有已知缺陷(孔洞、不一致法线、惯性违规)的特定对象实例
- (B) 自动网格验证检查水密几何体、流形边缘和正确的UV映射
- (C) aigen与objaverse资产的不同缩放因子补偿跨来源不一致的真实世界尺寸
- (D) 对象实例在包含在资产库之前经过手动策划和测试
- (E) 框架使用孔洞填充和重新网格化算法自动修复网格缺陷

**正确答案：A, C, D**

**解析：** 这测试对kitchen_objects.py和资产管理中资产质量控制的理解。(A)正确 - OBJ_CATEGORIES包含带有解释缺陷的注释的排除列表(例如，'bagel_8'被排除，'bar_1'有孔洞，'can_5'有不一致的方向)。(C)正确 - aigen与objaverse的不同缩放值表明手动校准以实现尺寸一致性。(D)正确 - 策划的排除列表和特定缩放因子的存在表明手动策划。(B)不正确 - 代码库中不存在自动验证代码。(E)不正确 - 未实现自动修复算法；有缺陷的资产被排除。

---

## robocasa - 问题 21 (多选题)
框架的场景生成必须处理关于固定装置关系的空间推理。哪些机制使系统能够推理空间配置？

- (A) 带有FixtureType和fixture_id的get_fixture()方法启用按类型和实例查询特定固定装置
- (B) OU.point_in_fixture()函数测试点是否位于固定装置的2D或3D边界内
- (C) get_rel_transform()函数计算固定装置之间的相对变换以定位对象
- (D) 场景图数据结构维护所有场景元素之间的父子关系
- (E) 固定装置放置系统使用约束满足来确保固定装置不重叠

**正确答案：A, B, C, E**

**解析：** 这测试对kitchen.py、object_utils.py和固定装置管理中空间推理的理解。(A)正确 - get_fixture()提供对固定装置的结构化访问。(B)正确 - OU.point_in_fixture()在compute_robot_base_placement_pose()中用于空间查询。(C)正确 - OU.get_rel_transform()计算固定装置之间的相对姿态。(E)正确 - 固定装置放置使用带碰撞检查的SequentialCompositeSampler。(D)不正确 - 虽然MuJoCo有运动学树，但Python代码中没有用于空间推理的显式场景图抽象。

---

## robocasa - 问题 22 (多选题)
框架的MJCF后处理系统必须处理各种模型调整。在edit_model_xml()期间应用哪些转换？

- (A) 将以'base0_'开头的元素重命名为'mobilebase0_'以实现与旧PandaOmron演示的向后兼容性
- (B) 通过替换视觉几何元素中的纹理文件路径来应用纹理交换
- (C) 根据对象重要性调整碰撞几何分辨率以优化性能
- (D) 设置渲染属性，如反射率和镜面反射以实现视觉真实性
- (E) 删除未使用的关节和主体以降低模型复杂性

**正确答案：A, B, D**

**解析：** 这测试对kitchen.py的edit_model_xml()中MJCF后处理的理解。(A)正确 - 代码明确将'base0_'重命名为'mobilebase0_'，并带有关于PandaOmron演示的注释。(B)正确 - 在edit_model_xml()内调用纹理交换函数以替换纹理。(D)正确 - 该方法调整固定装置和对象的渲染属性。(C)不正确 - 不存在自适应碰撞分辨率。(E)不正确 - 不执行未使用元素的修剪。

---

