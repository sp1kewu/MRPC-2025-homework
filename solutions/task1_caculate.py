import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

omega = 0.5           #  rad/s
alpha_deg = 180/12    # 角度
alpha = math.radians(alpha_deg)

# 工具函数
def quat_to_rotmat(x, y, z, w):
    """四元数 -> 旋转矩阵, 输入需为 (x, y, z, w)"""
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    R = np.array([
        [1.0 - (yy + zz), xy - wz,       xz + wy      ],
        [xy + wz,         1.0 - (xx+zz), yz - wx      ],
        [xz - wy,         yz + wx,       1.0 - (xx+yy)]
    ])
    return R

def rotmat_to_quat(R):
    """旋转矩阵 -> 四元数 (x, y, z, w)"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w])
    # 归一化
    q = q / np.linalg.norm(q)
    return q

def R_BD(t, omega, alpha):
    """题目给定的 B->D 旋转矩阵（式(1)）"""
    s = math.sin(omega * t)
    c = math.cos(omega * t)
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    return np.array([
        [c,  -s * ca,  s * sa],
        [s,   c * ca, -c * sa],
        [0.,  sa,      ca    ]
    ])

#读取 tracking.csv
df = pd.read_csv("documents\\tracking.csv") 

t_array  = df["t"].values
qx_body  = df["qx"].values
qy_body  = df["qy"].values
qz_body  = df["qz"].values
qw_body  = df["qw"].values

qxd_list, qyd_list, qzd_list, qwd_list = [], [], [], []

for t, qx, qy, qz, qw in zip(t_array, qx_body, qy_body, qz_body, qw_body):
    # 1. 机体姿态四元数 -> 旋转矩阵 ^W R_B
    R_WB = quat_to_rotmat(qx, qy, qz, qw)

    # 2. 末端相对机体旋转矩阵 ^B R_D(t)
    R_BD_t = R_BD(t, omega, alpha)

    # 3. 合成世界系下末端姿态: ^W R_D = ^W R_B * ^B R_D
    R_WD = R_WB @ R_BD_t

    # 4. 转回四元数 (x, y, z, w)
    qx_d, qy_d, qz_d, qw_d = rotmat_to_quat(R_WD)

    # 5. 符号统一：保证 w >= 0
    if qw_d < 0:
        qx_d, qy_d, qz_d, qw_d = -qx_d, -qy_d, -qz_d, -qw_d

    qxd_list.append(qx_d)
    qyd_list.append(qy_d)
    qzd_list.append(qz_d)
    qwd_list.append(qw_d)

# 保存结果到 CSV 文件
out_df = pd.DataFrame({
    "t": t_array,
    "qx": qxd_list,
    "qy": qyd_list,
    "qz": qzd_list,
    "qw": qwd_list
})
out_df.to_csv("end_effector_quaternion.csv", index=False)

# 绘制四元数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(t_array, qxd_list, label='qx')
plt.plot(t_array, qyd_list, label='qy')
plt.plot(t_array, qzd_list, label='qz')
plt.plot(t_array, qwd_list, label='qw')
plt.xlabel("Time [s]")
plt.ylabel("Quaternion components")
plt.title("End Effector Quaternion in World Frame")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
