from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class KuavoMimicCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        # 目标运动的时间步列表，用于获取未来多个时间点的目标姿态 / 运动数据
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        num_envs = 4096
        num_actions = 28 # 6*2 (legs) + 7*2 (arms) + 2 (head) = 28
        n_priv = 0
        n_mimic_obs = 3*4 + 28 # 28 for dof pos 根节点位置、姿态、速度、角速度 + 28 关节
        n_proprio = len(tar_motion_steps_priv) * n_mimic_obs + 3 + 2 + 3*num_actions # 基座角速度3+IMU2+关节位置偏差+关节速度+上一时刻速度
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv + extra_critic_obs 
        num_privileged_obs = None

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        root_tracking_termination_dist = 0.8
        rand_reset = True
        track_root = False
        
        # 关节误差权重，用于计算模仿奖励
        # 顺序: Left Leg (6), Right Leg (6), Left Arm (7), Right Arm (7), Head (2)
        # 假设关节顺序为: L_Leg, R_Leg, L_Arm, R_Arm, Head (需根据URDF加载顺序确认，通常是按URDF定义顺序)
        # URDF顺序: leg_l1..l6, leg_r1..r6, zarm_l1..l7, zarm_r1..r7, zhead_1..2
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, # Left Leg (Hip*3, Knee, Ankle*2)
                     1.0, 1.0, 1.0, 1.0, 0.1, 0.1, # Right Leg
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Arm
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Arm
                     1.0, 1.0 # Head
                     ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0] # 初始位置
        
        # 默认关节角度，定义站立姿态
        default_joint_angles = {
            # Left Leg
            'leg_l1_joint': 0.0,  # Hip Roll
            'leg_l2_joint': 0.0,  # Hip Yaw
            'leg_l3_joint': -0.2, # Hip Pitch
            'leg_l4_joint': 0.4,  # Knee
            'leg_l5_joint': -0.2, # Ankle Pitch
            'leg_l6_joint': 0.0,  # Ankle Roll
            
            # Right Leg
            'leg_r1_joint': 0.0,
            'leg_r2_joint': 0.0,
            'leg_r3_joint': -0.2,
            'leg_r4_joint': 0.4,
            'leg_r5_joint': -0.2,
            'leg_r6_joint': 0.0,
            
            # Left Arm (假设姿态)
            'zarm_l1_joint': 0.0,
            'zarm_l2_joint': 0.2,
            'zarm_l3_joint': 0.0,
            'zarm_l4_joint': 0.5,
            'zarm_l5_joint': 0.0,
            'zarm_l6_joint': 0.0,
            'zarm_l7_joint': 0.0,
            
            # Right Arm
            'zarm_r1_joint': 0.0,
            'zarm_r2_joint': -0.2,
            'zarm_r3_joint': 0.0,
            'zarm_r4_joint': 0.5,
            'zarm_r5_joint': 0.0,
            'zarm_r6_joint': 0.0,
            'zarm_r7_joint': 0.0,
            
            # Head
            'zhead_1_joint': 0.0,
            'zhead_2_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        # PD控制器参数
        stiffness = {
                     'leg_l1': 100, 'leg_l2': 100, 'leg_l3': 100, # Hip
                     'leg_l4': 150, # Knee
                     'leg_l5': 40, 'leg_l6': 40, # Ankle
                     
                     'leg_r1': 100, 'leg_r2': 100, 'leg_r3': 100,
                     'leg_r4': 150,
                     'leg_r5': 40, 'leg_r6': 40,
                     
                     'zarm': 40, # Arms
                     'zhead': 40, # Head
                     }  # [N*m/rad]
        
        damping = {  
                     'leg_l1': 2, 'leg_l2': 2, 'leg_l3': 2,
                     'leg_l4': 4,
                     'leg_l5': 2, 'leg_l6': 2,
                     
                     'leg_r1': 2, 'leg_r2': 2, 'leg_r3': 2,
                     'leg_r4': 4,
                     'leg_r5': 2, 'leg_r6': 2,
                     
                     'zarm': 5,
                     'zhead': 2,
                     }  # [N*m*s/rad]
        
        action_scale = 0.5
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/../assets/kuavo_s45/biped_s45.urdf'
        
        # 身体部位名称映射，用于代码中查找对应的 Body Index
        torso_name: str = 'base_link'
        chest_name: str = 'base_link' # Kuavo 没有独立的胸部 Link，使用 base_link

        # Link 名称
        thigh_name: str = 'leg_l3_link' # 大腿
        shank_name: str = 'leg_l4_link' # 小腿
        foot_name: str = 'leg_l6_link'  # 足部 (Ankle Roll Link)
        
        # Kuavo 没有腰部关节，这里留空或指向 base_link
        waist_name: list = ['base_link'] 
        
        upper_arm_name: str = 'zarm_l3_link' # 上臂
        lower_arm_name: str = 'zarm_l5_link' # 前臂
        hand_name: str = 'zarm_l7_end_effector' # 手部

        feet_bodies = ['l_foot_toe', 'r_foot_toe'] # 用于检测接触的足部 Body
        n_lower_body_dofs: int = 12 # 双腿自由度总和

        # 接触惩罚部位
        penalize_contacts_on = ["zarm", "leg_l1", "leg_l2", "leg_l3", "leg_l4"]
        terminate_after_contacts_on = ['base_link']
        
        # ========================= Inertia =========================
        # 关节转子惯量，这里暂时使用默认值或参考 G1
        dof_armature = [0.01] * 28
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        "feet_stumble",
                        "feet_contact_forces",
                        "lin_vel_z",
                        "ang_vel_xy",
                        "orientation",
                        "dof_pos_limits",
                        "dof_torque_limits",
                        "collision",
                        "torque_penalty",
                        # "thigh_torque_roll_yaw", # Kuavo 关节命名不同，可能需要调整
                        # "thigh_roll_yaw_acc",
                        "dof_acc",
                        "dof_vel",
                        "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_translation = 0.6
            tracking_root_rotation = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0
            
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            feet_air_time = 5.0
            
            ang_vel_xy = -0.01
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        target_feet_height = 0.07
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 350
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        termination_roll = 1.5
        termination_pitch = 1.5
        root_height_diff_threshold = 0.3

    class domain_rand:
        domain_rand_general = True
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
    
    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        # 关键身体部位，用于计算跟踪误差
        key_bodies = ["zarm_l7_end_effector", "zarm_r7_end_effector", "l_foot_toe", "r_foot_toe", "leg_l4_link", "leg_r4_link", "zarm_l4_link", "zarm_r4_link", "head_link"]
        
        # 动作文件路径，需替换为适配 Kuavo 的动作数据
        motion_file = f"../../../../motion_data/LAFAN1_g1_gmr/dance1_subject2.pkl"

        reset_consec_frames = 30


class KuavoMimicCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 30002

        save_interval = 500
        experiment_name = 'kuavo_mimic_test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
    
    class policy(HumanoidMimicCfgPPO.policy):
        # 动作标准差初始值，需与 num_actions (28) 匹配
        # Legs (12) + Arms (14) + Head (2)
        action_std = [0.7] * 12 + [0.5] * 14 + [0.5] * 2
        init_noise_std = 0.8
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
